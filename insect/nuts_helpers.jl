# NUTS chains on an `OrdinaryDiffEqLDP` with AdvancedHMC, replicating `Turing.NUTS(δ, metricT=UnitEuclideanMetric)`
# (multinomial sampling, generalised no-U-turn criterion, dual averaging of the step size, warmup discarded).
using AdvancedHMC, LogDensityProblems, MCMCChains, Random, StableRNGs

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

# Runs one chain per initial point on separate threads, each with its own copy of `target` and its own RNG seeded
# from `rng`. Returns an `MCMCChains.Chains` with parameters θ[1], ..., θ[d] followed by the sampler internals.
function run_nuts_chains(rng, target, inits, n_sample, nadapts; kwargs...)
    n_chains = length(inits)
    d = LogDensityProblems.dimension(target)
    seeds = rand(rng, UInt64, n_chains)
    results = Vector{Any}(undef, n_chains)
    Threads.@threads :dynamic for c in 1:n_chains
        results[c] = run_nuts_chain(StableRNG(seeds[c]), customcopy(target), inits[c], n_sample, nadapts; kwargs...)
    end
    vals = stack(first.(results); dims=3) # n_sample x (d + n_internals) x n_chains
    names = [[Symbol("θ[$i]") for i in 1:d]; NUTS_INTERNALS]
    return Chains(
        vals, names, (parameters = names[1:d], internals = NUTS_INTERNALS);
        info = (start_time = [r[2] for r in results], stop_time = [r[3] for r in results]),
    )
end
