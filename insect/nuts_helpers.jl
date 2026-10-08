# NUTS chains on an `OrdinaryDiffEqLDP` with AdvancedHMC, replicating `Turing.NUTS(δ, metricT=UnitEuclideanMetric)`
# (multinomial sampling, generalised no-U-turn criterion, dual averaging of the step size, warmup discarded).
using AdvancedHMC, Distributed, LogDensityProblems, MCMCChains, Random, StableRNGs

const NUTS_INTERNALS = [
    :lp, :acceptance_rate, :step_size, :nom_step_size, :n_steps, :tree_depth, :numerical_error, :hamiltonian_energy
]

function run_nuts_chain(rng, target, init, n_sample, nadapts; δ=0.9, max_depth=10, Δ_max=1000.)
    metric = UnitEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target)
    ϵ = find_good_stepsize(rng, h, init)
    integrator = Leapfrog(ϵ)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth, Δ_max)))
    adaptor = StepSizeAdaptor(δ, integrator)
    start_time = time()
    θs, stats = AdvancedHMC.sample(
        rng, h, κ, init, n_sample + nadapts, adaptor, nadapts; drop_warmup=true, verbose=false, progress=false
    )
    stop_time = time()
    vals = hcat(
        stack(θs; dims=1),
        [Float64(getproperty(stat, s === :lp ? :log_density : s)) for stat in stats, s in NUTS_INTERNALS],
    )
    return vals, start_time, stop_time
end

# Starts `n` worker processes for `parallel = :distributed` and includes `files` on each of them, in order. A target sent
# to a worker carries its closures, but the globals they read (`logpdf`, `n_u`, ...) must exist there, so `files`
# must define everything the target uses (for the insect models: setup.jl, ldp_setup.jl and this file). Unlike
# `start_SMC_workers`, the workers do not run the calling script. Call this before pinning threads: worker processes
# inherit the CPU affinity of the master thread that starts them. It does nothing when `n == 0` or when workers already
# exist. `n` defaults to the environment variable MCMC_NWORKERS (unset: 0) and `heap_size_hint` (per worker) to
# MCMC_WORKER_HEAP (unset: 2G).
function start_nuts_workers(
    files; n=parse(Int, get(ENV, "MCMC_NWORKERS", "0")), heap_size_hint=get(ENV, "MCMC_WORKER_HEAP", "2G"),
)
    (myid() == 1 && n > 0 && nprocs() == 1) || return nothing
    paths = abspath.(files)
    t = @elapsed begin
        addprocs(n; exeflags=`--project=$(Base.active_project()) --threads=1 --heap-size-hint=$heap_size_hint`)
        @everywhere workers() foreach(include, $paths)
    end
    @info "Started $n worker processes in $(round(t; digits=1)) s"
    flush(stdout)
    flush(stderr)
    return nothing
end

# Runs one chain per initial point, each with its own copy of `target` and its own RNG seeded from `rng`. Returns an
# `MCMCChains.Chains` with parameters θ[1], ..., θ[d] followed by the sampler internals.
# `parallel` chooses where the chains run:
#   :threads     - one chain per thread (a `customcopy` of `target` each).
#   :distributed - one chain per worker process (see `start_nuts_workers`), with `pmap`; `target` is serialised to the
#                  worker for each chain. Each process has its own heap, so a garbage collection on one worker does not
#                  stop the others.
# Both give the same output for the same `rng`.
function run_nuts_chains(rng, target, inits, n_sample, nadapts; parallel=:threads, kwargs...)
    parallel in (:threads, :distributed) ||
        throw(ArgumentError("parallel must be :threads or :distributed (got $(repr(parallel)))"))
    parallel === :distributed && nprocs() == 1 &&
        throw(ArgumentError("parallel = :distributed, but there are no worker processes (see start_nuts_workers)"))
    n_chains = length(inits)
    d = LogDensityProblems.dimension(target)
    seeds = rand(rng, UInt64, n_chains)
    if parallel === :threads
        results = Vector{Any}(undef, n_chains)
        Threads.@threads :dynamic for c in 1:n_chains
            results[c] = run_nuts_chain(StableRNG(seeds[c]), customcopy(target), inits[c], n_sample, nadapts; kwargs...)
        end
    else
        results = pmap(1:n_chains) do c
            run_nuts_chain(StableRNG(seeds[c]), target, inits[c], n_sample, nadapts; kwargs...)
        end
    end
    vals = stack(first.(results); dims=3) # n_sample x (d + n_internals) x n_chains
    names = [[Symbol("θ[$i]") for i in 1:d]; NUTS_INTERNALS]
    return Chains(
        vals, names, (parameters = names[1:d], internals = NUTS_INTERNALS);
        info = (start_time = [r[2] for r in results], stop_time = [r[3] for r in results]),
    )
end
