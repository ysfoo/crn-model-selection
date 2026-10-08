include(joinpath(@__DIR__, "stats_helpers.jl"));
include(joinpath(@__DIR__, "plot_helpers.jl"));
include(joinpath(@__DIR__, "transform_maps.jl"));

using Distributions, Random, StableRNGs, Statistics, StatsBase
using Distances, LinearAlgebra, LogExpFunctions, NearestNeighbors
using AdvancedHMC, Bijectors, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, PreallocationTools
using Accessors, DiffResults, JLD2, ProgressMeter
using Distributed: @everywhere, addprocs, myid, nprocs, nworkers, remotecall_fetch, workers
using VideoIO


struct SMCParticle
    state::AbstractVector{Float64}
    loglike::Float64   # log likelihood
    logtarget::Float64 # log target density
    stepsize::Float64  # NUTS step size
    n_nuts::Int64      # number of NUTS iterations
    info::NamedTuple
end


function nuts_move(rng, particle, target, n_nuts, pop_info; adapt_stepsize_func=adapt_using_curr)
    metric = DiagEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target, ForwardDiff)

    stepsize = particle.stepsize
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, NoAdaptation(), 0; verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
        n_grad = 0
    else
        n_grad = particle.info.n_grad
        sum_sqjdist = particle.info.sum_sqjdist
        sum_acc_rate = particle.info.agg_acc_rate * particle.n_nuts
    end
    delta = θs[end] .- particle.state

    agg_n_nuts = particle.n_nuts + n_nuts
    acc_rate_incr = sum(
        begin
            a = s.acceptance_rate
            isfinite(a) ? a : 0.
        end for s in stats
    )
    sum_acc_rate += acc_rate_incr

    # Each NUTS transition costs `n_steps` leapfrog steps, i.e. gradient evaluations.
    curr_sqjdist = sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        curr_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end
    curr_n_grad = sum(stat.n_steps for stat in stats)
    sum_sqjdist += curr_sqjdist
    n_grad += curr_n_grad

    n_diverge = particle.n_nuts > 0 ? particle.info.n_diverge : 0
    n_diverge += sum(stat.numerical_error for stat in stats)

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        sum_sqjdist = sum_sqjdist,
        curr_esjd_per_grad = curr_sqjdist / curr_n_grad,
        curr_n_grad = curr_n_grad,
        n_grad = n_grad,
        delta = delta,
        n_diverge = n_diverge,
    )

    # adapt stepsize after NUTS using most recent acceptance rate
    stepsize_factor = adapt_stepsize_func(info)
    if isfinite(stepsize_factor)
        stepsize *= stepsize_factor
    end
    logtarget = stats[end].log_density

    return SMCParticle(θs[end], NaN, isfinite(logtarget) ? logtarget : -Inf, stepsize, agg_n_nuts, info)
end

# NUTS move with a population-level diagonal mass matrix and a step size that only depends on the population,
# not on the state of the particle being moved. The step size is jittered multiplicatively, exp(jitter * randn()).
# `pop_info` provides `invmass` and `jitter`, plus either
#   - `stepsize`: the step size solved from a regression of acceptance on step size (see `compute_QB_info`), or
#   - `stepsizes` and `cum_probs`: a pool of step sizes from which one is drawn with probability `cum_probs`
#     (Fearnhead & Taylor, 2013; see `compute_FT_info`).
function nuts_invmass_move(rng, particle, target, n_nuts, pop_info)
    metric = DiagEuclideanMetric(pop_info.invmass)
    h = Hamiltonian(metric, target, ForwardDiff)

    if hasproperty(pop_info, :stepsize)
        base_stepsize = pop_info.stepsize
    else
        k = min(searchsortedfirst(pop_info.cum_probs, rand(rng)), length(pop_info.stepsizes))
        base_stepsize = pop_info.stepsizes[k]
    end
    stepsize = base_stepsize * exp(pop_info.jitter * randn(rng))
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, NoAdaptation(), 0; verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
        n_grad = 0
    else
        n_grad = particle.info.n_grad
        sum_sqjdist = particle.info.sum_sqjdist
        sum_acc_rate = particle.info.agg_acc_rate * particle.n_nuts
    end
    delta = θs[end] .- particle.state

    agg_n_nuts = particle.n_nuts + n_nuts
    acc_rate_incr = sum(
        begin
            a = s.acceptance_rate
            isfinite(a) ? a : 0.
        end for s in stats
    )
    sum_acc_rate += acc_rate_incr

    # Each NUTS transition costs `n_steps` leapfrog steps, i.e. gradient evaluations.
    curr_sqjdist = sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        curr_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end
    curr_n_grad = sum(stat.n_steps for stat in stats)
    sum_sqjdist += curr_sqjdist
    n_grad += curr_n_grad

    n_diverge = particle.n_nuts > 0 ? particle.info.n_diverge : 0
    n_diverge += sum(stat.numerical_error for stat in stats)

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        sum_sqjdist = sum_sqjdist,
        curr_esjd_per_grad = curr_sqjdist / curr_n_grad,
        curr_n_grad = curr_n_grad,
        n_grad = n_grad,
        delta = delta,
        n_diverge = n_diverge,
    )

    logtarget = stats[end].log_density

    return SMCParticle(θs[end], NaN, isfinite(logtarget) ? logtarget : -Inf, stepsize, agg_n_nuts, info)
end

# NUTS on the transformed target θ_k = F_k(u_k) (see transform_maps.jl), with the per-coordinate maps in `pop_info.maps`
# (fitted to the population, e.g. by `compute_TQB_info`) and the identity mass matrix in
# `pop_info.invmass`. The particle is mapped to u, moved by `nuts_invmass_move` on `TransformedTarget`, and mapped back.
# The stored log target is that of θ (the latent density minus Σ log F'), `info.delta` is the θ-space change, and the
# step size and the jump statistics (`sum_sqjdist`, `curr_esjd_per_grad`) are in latent units.
function nuts_transformed_move(rng, particle, target, n_nuts, pop_info)
    maps = pop_info.maps
    u0 = to_latent(maps, particle.state)
    moved = nuts_invmass_move(rng, @set(particle.state = u0), TransformedTarget(target, maps), n_nuts, pop_info)
    θ = to_theta(maps, moved.state)
    logtarget = isfinite(moved.logtarget) ? moved.logtarget - logjac(maps, moved.state) : -Inf
    info = merge(moved.info, (delta = θ .- particle.state,))
    return SMCParticle(θ, NaN, logtarget, moved.stepsize, moved.n_nuts, info)
end

## Step size and maps shared by the problem drivers (insect/SMC.jl, pinene/SMC.jl)

## Population info for `nuts_invmass_move` with a regression-based step size.
# Regress acceptance rates from previous population against log step size. Fit model for E[a | log ε] using quasi-binomial
# regression a(ε) = logistic(β0 + β1 (log ε - mean log ε)) by Fisher scoring. Returns the step size solution to a(ε)=`acc_target`.
# `nuts_invmass_move` multiplies this stepsize by exp(jitter * randn()).
# - `weights`: if `use_ngrad`, particles are weighted by `info.curr_n_grad`, default on.
# - `max_dlog`: cap on |log new ε - mean log ε| (default: no cap), to limit extrapolation beyond the spread of the step sizes.
# - Fallback (no acceptance information, too few particles, no spread in step sizes, or a non-negative slope): the
#   geometric mean of the particles' step sizes is used, i.e. the step size is left unchanged.
function fit_acc_stepsize(stepsizes, accs; acc_target=0.8, weights=nothing, min_obs=20, max_dlog=Inf)
    ok = findall(eachindex(stepsizes)) do i
        isfinite(stepsizes[i]) && stepsizes[i] > 0 && isfinite(accs[i]) &&
            (isnothing(weights) || (isfinite(weights[i]) && weights[i] > 0))
    end
    length(ok) < min_obs && return nothing
    x = log.(stepsizes[ok]); y = clamp.(accs[ok], 0., 1.)
    w = isnothing(weights) ? ones(length(ok)) : Float64.(weights[ok])
    x̄ = sum(w .* x) / sum(w); xc = x .- x̄
    maximum(abs, xc) > 1e-8 || return nothing

    β = [logit(clamp(sum(w .* y) / sum(w), 0.01, 0.99)), 0.]
    # Fisher scoring
    for _ in 1:100
        η = β[1] .+ β[2] .* xc; μ = logistic.(η)
        v = max.(μ .* (1 .- μ), 1e-6)
        z = η .+ (y .- μ) ./ v                  # working response
        W = w .* v
        S = [sum(W) sum(W .* xc); sum(W .* xc) sum(W .* xc .^ 2)]
        β_new = S \ [sum(W .* z), sum(W .* xc .* z)]
        all(isfinite, β_new) || return nothing
        converged = maximum(abs, β_new .- β) < 1e-8
        β = β_new
        converged && break
    end
    β[2] < 0 || return nothing
    dlog = clamp((logit(acc_target) - β[1]) / β[2], -max_dlog, max_dlog)
    return (stepsize = exp(x̄ + dlog), slope = β[2], centre = exp(x̄))
end

# Step size for the next round: quasi-binomial fit of acceptance on log step size (target `acc_target`) in pass 1; in later
# passes of the same SMC iteration the previous pass's base step size (`prev_pop_info.stepsize`, before jitter) times
# 0.95, capped at the fit, so it never increases. Without `prev_pop_info` (pass 1), the nominal step size, used when the
# fit fails, is the geometric mean of the particles' (jittered) step sizes.
function qb_stepsize(
    particles, npass, prev_pop_info=nothing; acc_target=0.8, max_dlog=Inf, use_ngrad=false, min_obs=20,
)
    stepsizes = [p.stepsize for p in particles]
    if isnothing(prev_pop_info)
        valid = filter(s -> isfinite(s) && s > 0, stepsizes)
        stepsize = exp(mean(log.(valid))) # nominal stepsize is previous stepsize
    else
        stepsize = prev_pop_info.stepsize  # base step size of the previous pass
    end

    accs = [hasproperty(p.info, :curr_acc_rate) ? p.info.curr_acc_rate : NaN for p in particles]
    weights = use_ngrad ? [hasproperty(p.info, :curr_n_grad) ? p.info.curr_n_grad : NaN for p in particles] : nothing
    fit = fit_acc_stepsize(stepsizes, accs; acc_target, weights, min_obs, max_dlog)
    if npass > 1
        stepsize *= 0.95 # decrease if multiple NUTS rounds
         if !isnothing(fit)
            stepsize = min(stepsize, fit.stepsize) # don't increase stepsize
         end
    elseif !isnothing(fit)
        stepsize = fit.stepsize
    end
    return stepsize
end

## Population info for `nuts_transformed_move`: per-coordinate two-slope maps (transform_maps.jl) and the step size
# Before the first round of every SMC iteration, fit a two-slope map to every coordinate by minimising the sample Fisher
# divergence with the log-slope penalty λ = λN / N (default 100 / N); the fit uses no spike-and-slab threshold. As in
# `compute_invmass`, the 1% of particles with the lowest log target are left out of the fit. A fit whose asymptotes are
# far from the slopes the particles see (`check_asymptotes`) falls back to the equal-slope optimum (the affine nutpie
# map). NUTS then runs in u with the identity mass matrix, since the D-optimal maps already set the scale of each
# coordinate. The maps are reused from `prev_pop_info` in later rounds of the same iteration (`refit_each_pass=true`
# refits them every round). The step size follows `qb_stepsize`, updated every round; particles carry their step size
# from the previous round, in latent units.
function compute_TQB_info(
    particles, target, npass, prev_pop_info=nothing; acc_target=0.8, jitter=0.2, max_dlog=Inf, use_ngrad=false,
    min_obs=20, λN=DEFAULT_λN, ζ_min=0.2, refit_each_pass=false, verbose=1,
)
    stepsize = qb_stepsize(particles, npass, prev_pop_info; acc_target, max_dlog, use_ngrad, min_obs)
    maps = (isnothing(prev_pop_info) || refit_each_pass) ? fit_maps(particles, target, npass; λN, ζ_min, verbose) :
        prev_pop_info.maps
    if verbose > 0
        @info "Step size (pass $npass)" stepsize
        flush(stdout)
    end
    return (invmass = ones(length(maps)), stepsize = stepsize, jitter = jitter, maps = maps)
end

# Two-slope maps for every coordinate, fitted to the population (see `compute_TQB_info`).
function fit_maps(particles, target, npass; λN=DEFAULT_λN, ζ_min=0.2, verbose=1)
    # scores at the current states, as in `compute_invmass`
    logp_func = Base.Fix1(LogDensityProblems.logdensity, target)
    x1 = particles[1].state
    d, n = length(x1), length(particles)
    X, G, logp = Matrix{Float64}(undef, d, n), Matrix{Float64}(undef, d, n), Vector{Float64}(undef, n)
    result = DiffResults.GradientResult(x1)
    for (i, p) in enumerate(particles)
        X[:, i] .= p.state
        result = ForwardDiff.gradient!(result, logp_func, p.state)
        logp[i] = DiffResults.value(result)
        G[:, i] .= DiffResults.gradient(result)
    end
    keep = logp .>= quantile(logp, 0.01)   # drop bottom 1% by log target density

    maps = Vector{TwoSlope}(undef, d)
    t_fit = @elapsed for k in 1:d
        θ, α = X[k, keep], G[k, keep]
        f = fit_twoslope(θ, α; λ=λN / length(θ), ζ_min)
        maps[k], reason = check_asymptotes(f.m, f.u, θ, α)
        isempty(reason) || @warn "Two-slope fit of coordinate $k replaced by the equal-slope map" reason f.m
    end
    if verbose > 0
        @info "Two-slope maps fitted (pass $npass)" t_fit
        flush(stdout)
    end
    return maps
end


# Squared jump distance per NUTS iteration; runs saved before `sum_sqjdist` was introduced store it as `info.esjd`.
esjd_per_nuts(p::SMCParticle) = hasproperty(p.info, :esjd) ? p.info.esjd : p.info.sum_sqjdist / p.n_nuts

no_adapt_func(info) = 1.
adapt_using_agg(info) = 1.5^((info.agg_acc_rate-0.8)/0.2)
adapt_using_curr(info) = 1.5^((info.curr_acc_rate-0.8)/0.2)


## Parallel backends
# `parallel` in `run_SMC` and `resume_SMC` selects how the per-particle work (moves and log target evaluations) is spread:
#   :serial      - one loop on the calling thread;
#   :threads     - one task per Julia thread, each with its own copy of the target, taking particles one at a time from a
#                  shared counter (so slow particles do not hold up a fixed block of others);
#   :distributed - Distributed.jl worker processes (see `start_SMC_workers`). Each remote call rebuilds the target from
#                  `(ldp_builder, prior_path, γ)` on the worker and processes a batch of particles. Every process has its
#                  own heap, so a garbage collection on one worker does not stop the others.
# Each particle is moved with its own RNG, StableRNG(seed), with the seeds drawn from the master RNG before each round,
# so results do not depend on the backend, on the number of threads or workers, or on which particle finishes first.

function resolve_backend(parallel)
    parallel isa Bool && throw(ArgumentError(
        "parallel = $parallel is no longer accepted; use :serial, :threads or :distributed"))
    parallel in (:serial, :threads, :distributed) ||
        throw(ArgumentError("parallel must be :serial, :threads or :distributed (got $(repr(parallel)))"))
    parallel === :distributed && nprocs() == 1 &&
        throw(ArgumentError("parallel = :distributed, but there are no worker processes (see start_SMC_workers)"))
    return parallel
end

# Starts `n` worker processes for `parallel = :distributed` and makes each of them run `script` (by default the script
# the master is running) with the master's command-line arguments, so that every definition in the problem script (prior
# path, likelihood builder, move function, ...) also exists on the workers. `run_SMC` and `resume_SMC` return at once on
# a worker, so only the master runs SMC; anything else the script does at top level runs on every process, so it must be
# safe to repeat (e.g. `mkpath`, not copying or deleting files). Workers have one thread each.
# Call this before pinning threads: worker processes inherit the CPU affinity of the master thread that starts them.
# It does nothing on a worker, when `n == 0` or when workers already exist. `n` defaults to the environment variable
# SMC_NWORKERS (unset: 0) and `heap_size_hint` (per worker) to SMC_WORKER_HEAP (unset: 2G).
function start_SMC_workers(;
    n=parse(Int, get(ENV, "SMC_NWORKERS", "0")), heap_size_hint=get(ENV, "SMC_WORKER_HEAP", "2G"),
    script=PROGRAM_FILE,
)
    (myid() == 1 && n > 0 && nprocs() == 1) || return nothing
    isempty(script) && error("start_SMC_workers needs the path of the problem script (there is none in an interactive session)")
    script_path = abspath(script)
    args = copy(ARGS)
    t = @elapsed begin
        addprocs(n; exeflags=`--project=$(Base.active_project()) --threads=1 --heap-size-hint=$heap_size_hint`)
        @everywhere workers() begin
            empty!(ARGS)
            append!(ARGS, $args)
            include($script_path)
        end
    end
    @info "Started $n worker processes running $script_path in $(round(t; digits=1)) s"
    flush(stdout)
    flush(stderr)
    return nothing
end

# Applies `f(target, args, item)` to every element of `items` with the given backend and returns the results in order.
# `target` is used on the master (a copy per task with :threads); with :distributed, `target_spec = (ldp_builder,
# prior_path, γ)` rebuilds it on the worker, so `target` must equal `ldp_builder(prior_path, γ)`. `f` may only modify its
# target (`args` and the items are shared between tasks). `on_done(n_done)` is called on the master after each item
# (:serial, :threads) or batch (:distributed), never concurrently. `batch_size` (:distributed only) defaults to about
# 25 batches per worker.
function particle_map(
    f, args, items, target, target_spec, backend; batch_size=nothing, on_done=(n_done)->nothing,
)
    n = length(items)
    results = Vector{Any}(undef, n)
    if backend === :serial
        for i in 1:n
            results[i] = f(target, args, items[i])
            on_done(i)
        end
    elseif backend === :threads
        next_i = Threads.Atomic{Int}(1)
        failed = Threads.Atomic{Bool}(false)
        n_done = Ref(0)  # only changed while holding `done_lock`
        done_lock = ReentrantLock()
        @sync for _ in 1:min(n, Threads.nthreads())
            Threads.@spawn begin
                local_target = customcopy(target)
                while !failed[]
                    i = Threads.atomic_add!(next_i, 1)  # returns the value before the increment
                    i > n && break
                    try
                        results[i] = f(local_target, args, items[i])  # each index is written by one task only
                    catch
                        failed[] = true
                        rethrow()
                    end
                    lock(done_lock) do
                        n_done[] += 1
                        on_done(n_done[])
                    end
                end
            end
        end
    elseif backend === :distributed
        bs = isnothing(batch_size) ? max(1, cld(n, 25 * nworkers())) : batch_size
        batches = collect(Iterators.partition(1:n, bs))
        # One feeding task per worker, so a worker never has two calls in flight. The tasks run on the master thread
        # and only switch at `remotecall_fetch`, so reading and advancing `next_b` (and `n_done`) cannot interleave.
        next_b = Ref(1)
        n_done = Ref(0)
        failed = Ref(false)
        @sync for w in workers()
            @async while !failed[] && next_b[] <= length(batches)
                idxs = batches[next_b[]]
                next_b[] += 1
                try
                    results[idxs] = remotecall_fetch(_particle_batch, w, f, args, target_spec, items[idxs])
                catch
                    failed[] = true
                    rethrow()
                end
                n_done[] += length(idxs)
                on_done(n_done[])
            end
        end
    else
        throw(ArgumentError("unknown backend $(repr(backend))"))
    end
    return identity.(results)  # narrows the element type
end

# Runs on a worker: builds the target from `target_spec`, then applies `f` to a batch of items. The target is local to
# the call, so nothing is shared between calls.
function _particle_batch(f, args, target_spec, items)
    ldp_builder, prior_path, γ = target_spec
    target = ldp_builder(prior_path, γ)
    return [f(target, args, item) for item in items]
end

# Moves one particle with its own RNG; returns the moved particle and the time the move took.
function _move_particle(target, args, item)
    move_func, n_nuts, pop_info = args
    particle, seed = item
    t = @elapsed moved = move_func(StableRNG(seed), particle, target, n_nuts, pop_info)
    return (moved, t)
end

_logdensity_at(target, _, state) = LogDensityProblems.logdensity(target, state)


# `target_spec = (ldp_builder, prior_path, γ)` rebuilds `target` (see `particle_map`).
function perform_moves!(
    particles, target, target_spec, move_func, n_nuts_func, states_before_move, compute_pop_info,
    iter, targetinfo, rng;
    max_npass, pop_size, pbar_lines, parallel, verbose
)
    npass = 0
    move_time = 0.0
    last_n_nuts = nothing
    # `compute_pop_info(particles, target, npass, prev_pop_info)` sees the previous round's result in this SMC iteration
    # (`nothing` before the first round), e.g. to reuse its mass matrix or maps and only update the step size
    pop_info = nothing
    while npass < max_npass
        n_nuts = n_nuts_func(particles, last_n_nuts, npass, states_before_move; verbose, rng)
        n_nuts == 0 && break

        npass += 1
        last_n_nuts = n_nuts
        pop_info = compute_pop_info(particles, target, npass, pop_info)
        seeds = rand(rng, UInt64, length(particles))

        pbar = Progress(pop_size; desc="Iter $iter, pass $(npass)")
        pbar_step = max(1, pop_size ÷ pbar_lines)
        last_shown = Ref(0)
        on_done = function (n_done)
            if n_done - last_shown[] >= pbar_step
                ProgressMeter.update!(pbar, n_done)
                last_shown[] = n_done
            end
        end
        parallel === :threads && GC.gc(false)
        results = particle_map(
            _move_particle, (move_func, n_nuts, pop_info), collect(zip(particles, seeds)), target, target_spec, parallel;
            on_done,
        )
        for (i, (moved, t)) in enumerate(results)
            particles[i] = moved
            move_time += t
        end
        ProgressMeter.finish!(pbar)
    end
    return npass, move_time
end


# Re-evaluates the log target (and hence log likelihood) of a population with the same plain Float64 code path that is
# used to evaluate the next target, instead of trusting the `logtarget` stored by the move function. The latter is the
# density that NUTS saw, i.e. computed through ForwardDiff, and for an adaptive ODE solver the two can disagree by orders
# of magnitude at extreme states (loose tolerances). The importance weights are (plain density of the new target) /
# (stored density of the previous target), so a single such particle can receive a huge weight and collapse the
# resampled population onto one ancestor. `γ` is the path parameter of the population's own target.
# Particles whose density is not finite when re-evaluated get log target and log likelihood -Inf (zero weight).
function refresh_logtargets(particles, ldp_builder, prior_path, γ; parallel=:serial)
    target = ldp_builder(prior_path, γ)
    logtargets = particle_map(
        _logdensity_at, nothing, [p.state for p in particles], target, (ldp_builder, prior_path, γ),
        resolve_backend(parallel),
    )
    return [
        begin
            lt = isfinite(logtargets[i]) ? logtargets[i] : -Inf
            ll = isfinite(lt) ? (lt - prior_path(p.state, γ)) / γ : -Inf
            SMCParticle(p.state, ll, lt, p.stepsize, p.n_nuts, p.info)
        end for (i, p) in enumerate(particles)
    ]
end


# Seconds since this process started (Linux only, from /proc; NaN elsewhere). Used for the start-up time of a run: Julia
# start, package loading, problem setup, worker start-up and the initial population.
function process_uptime()
    try
        stat = read("/proc/self/stat", String)
        fields = split(stat[findlast(')', stat)+2:end])   # fields 3 onwards (the command name may contain spaces)
        start_ticks = parse(Float64, fields[20])           # field 22: start time, in clock ticks after boot
        clk_tck = ccall(:sysconf, Clong, (Cint,), 2)       # sysconf(_SC_CLK_TCK)
        uptime = parse(Float64, split(read("/proc/uptime", String))[1])
        return uptime - start_ticks / clk_tck
    catch
        return NaN
    end
end

# Start-up time of a run_SMC / resume_SMC call: since the process started for the first call, and since the previous
# call ended for later calls in the same process (e.g. a test script running several backends).
const _SMC_LAST_END = Ref(NaN)
startup_time() = isnan(_SMC_LAST_END[]) ? process_uptime() : time() - _SMC_LAST_END[]

# Runs one SMC iteration: reweight → resample → move.
# Modifies all_particles, targetinfos, npass_vec, figs, smc_times, wall_times, clock.
# `smc_times` holds the master's wall time plus the summed per-particle move times (comparable within a backend only:
# with :threads, every GC pause also stalls the other threads' moves). `wall_times` holds the wall time of each
# iteration, measured from `clock[]` (the previous iteration's mark) to just before the checkpoint, so that a checkpoint
# write counts towards the next iteration. `startup_times` (one entry per run or resume) is only saved here.
# Returns whether to terminate SMC.
function SMC_iteration!(
    iter, pop_size, target_ess,
    all_particles, targetinfos, npass_vec, figs, smc_times, wall_times, startup_times, clock,
    prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rng;
    max_npass=10, refresh_logtarget=true,
    parallel=:serial, pbar_lines=pop_size, vid_path=nothing, make_fig=nothing, verbose=0
)
    t_start = time()

    # Make the stored log targets of the previous population consistent with the plain evaluation used below. The initial
    # population (γ = 0) was already evaluated this way. Set `refresh_logtarget=false` for priors that depend on SMC state
    # (e.g. `FRPath`), whose previous target cannot be rebuilt from `prior_path` and the previous γ alone.
    if refresh_logtarget && targetinfos[end].γ > 0
        old = all_particles[end]
        all_particles[end] = refresh_logtargets(old, ldp_builder, prior_path, targetinfos[end].γ; parallel)
        if verbose > 0
            shifts = [abs(a.logtarget - b.logtarget) for (a, b) in zip(old, all_particles[end]) if isfinite(a.logtarget) && isfinite(b.logtarget)]
            n_lost = count(p -> !isfinite(p.logtarget), all_particles[end])
            max_shift = isempty(shifts) ? NaN : maximum(shifts)
            @info "Iter $iter refreshed log targets" max_shift n_lost
            flush(stdout)
        end
    end

    # Reweight    
    particles = all_particles[end]
    prev_targetinfo = targetinfos[end]
    target, targetinfo = build_target_SMC!(
        iter, all_particles, prev_targetinfo, target_ess, prior_path, ldp_builder; 
        verbose
    )
    if isnothing(target)
        return true
    end
    
    γ = targetinfo.γ
    push!(targetinfos, targetinfo)
    # `build_target_SMC!` returns `ldp_builder(prior_path, γ)`, which is how workers rebuild the target
    target_spec = (ldp_builder, prior_path, γ)

    prev_logtarget_vec = getproperty.(particles, :logtarget)
    logtarget_vec = particle_map(_logdensity_at, nothing, getproperty.(particles, :state), target, target_spec, parallel)
    logws = logtarget_vec .- prev_logtarget_vec
    logws[.!isfinite.(logws)] .= -Inf

    # Resample  
    ws = exp.(logws .- maximum(logws))      
    idxs = stratified_sampling(ws, pop_size; rng=rng)
    particles = [
        begin
            p = particles[idx]
            # keep the ancestor's info so that `compute_FT_info` can weight step sizes by its ESJD per gradient evaluation;
            # move functions ignore it (apart from stale copies of counters) when `n_nuts == 0`
            SMCParticle(p.state, NaN, NaN, p.stepsize, 0, deepcopy(p.info))
        end for idx in idxs
    ]
    states_before_move = getproperty.(particles, :state)

    # Move
    smc_time = time() - t_start
    npass, move_time = perform_moves!(
        particles, target, target_spec, move_func, n_nuts_func, states_before_move, compute_pop_info,
        iter, targetinfo, rng;
        max_npass, pop_size, pbar_lines, parallel, verbose
    )
    smc_time += move_time
    
    particles = [@set p.loglike = (p.logtarget - prior_path(p.state, γ)) / γ for p in particles]
    push!(all_particles, particles)

    agg_acc_rate = mean(filter(isfinite, [p.info.agg_acc_rate for p in particles]))
    median_esjd = median(esjd_per_nuts.(particles))

    if verbose > 0
        total_n_nuts = particles[1].n_nuts
        @info "Iter $iter post-MCMC" total_n_nuts agg_acc_rate median_esjd
        flush(stdout)
        flush(stderr)
    end    

    fig = nothing
    if !isnothing(vid_path)
        fig = make_fig(particles, iter)
        save(joinpath(vid_path, "iter$(iter).png"), fig, px_per_unit=4)
        push!(figs, fig)
    end
    push!(npass_vec, npass)
    push!(smc_times, smc_time)
    t_mark = time()
    push!(wall_times, t_mark - clock[])
    clock[] = t_mark

    @save fname all_particles iter targetinfos npass_vec smc_times wall_times startup_times prior_path

    return false
end


function build_target_SMC!(
    iter, all_particles, prev_targetinfo, target_ess, prior_path, ldp_builder; 
    verbose=0, Δγ=1e-8,
)
    prev_γ = prev_targetinfo.γ
    particles = all_particles[end]

    @assert iter > 0    

    if prev_γ >= 1.0
        return (nothing, nothing)
    end

    update!(prior_path, prev_γ, iter, all_particles)

    # Tempering
    states = getproperty.(particles, :state)
    loglikes = getproperty.(particles, :loglike)
    prev_logtargets = getproperty.(particles, :logtarget)
    
    ess_final = compute_ess(prior_path.(states, 1.) .+ loglikes .- prev_logtargets)
    if ess_final >= target_ess
        curr_γ = 1.
    else
        curr_γ = IVT_search(
            target_ess, 
            (γ) -> compute_ess(prior_path.(states, γ) .+ γ .* loglikes .- prev_logtargets), 
            max(Δγ, prev_γ), 1.; tol=Δγ,
        )
    end
    curr_ess = compute_ess(prior_path.(states, curr_γ) .+ curr_γ .* loglikes .- prev_logtargets)

    if verbose > 0
        @info "Iter $iter tempering" curr_γ curr_ess
        flush(stdout)
        flush(stderr)
    end

    return (ldp_builder(prior_path, curr_γ), get_targetinfo!(curr_γ, iter, prior_path))
end

function update!(prior_path, prev_γ, iter, all_particles)
    nothing
end

function get_targetinfo!(γ, iter, prior_path)
    return (γ = γ,)
end

function log_backend(backend)
    @info "SMC backend" backend Threads.nthreads() nworkers=(nprocs() > 1 ? nworkers() : 0)
    flush(stdout)
    flush(stderr)
end

# `parallel` selects the backend, :serial, :threads or :distributed (see "Parallel backends"); `max_iter` stops after that many SMC iterations (for tests).
function run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_pop_size=pop_size, init_stepsize::Float64=0.01, 
    max_npass=10, refresh_logtarget=true, compute_pop_info=(particles, target, npass, prev_pop_info)->(;),
    verbose=0, vid_path=nothing, make_fig=nothing,
    parallel=:serial, pbar_lines=pop_size, rng=Random.default_rng(), max_iter=typemax(Int),
)
    myid() == 1 || return nothing  # workers only serve the master (see `start_SMC_workers`)
    parallel = resolve_backend(parallel)
    log_backend(parallel)
    initstates = [init_sampler(rng) for _ in 1:init_pop_size] 

    iter = 0
    logpriors = prior_path.(initstates, 0.)

    # use γ = 1 to extract likelihood
    tmp_target = ldp_builder(prior_path, 1.) 
    tmp_logpriors = prior_path.(initstates, 1.)
    tmp_loglikes = particle_map(
        _logdensity_at, nothing, initstates, tmp_target, (ldp_builder, prior_path, 1.), parallel,
    ) .- tmp_logpriors

    # finite_loglikes = filter(isfinite, tmp_loglikes)
    # finite_tmp_logpriors = filter(isfinite, tmp_logpriors)
    # @info "Initial finite checks" length(finite_loglikes) length(finite_tmp_logpriors)
    flush(stdout)

    all_particles = [SMCParticle.(
        initstates,
        tmp_loglikes,
        logpriors, # actual logtarget has γ = 0
        init_stepsize, 0, Ref(NamedTuple())
    )]
    targetinfos = [get_targetinfo!(0., iter, prior_path)]
    npass_vec, smc_times, wall_times = Int64[], Float64[], Float64[]

    figs = Figure[]
    if !isnothing(vid_path)
        mkpath(vid_path)
    end

    startup_times = [startup_time()]
    clock = Ref(time())
    while true
        iter += 1
        SMC_done = SMC_iteration!(
            iter, pop_size, target_ess,
            all_particles, targetinfos, npass_vec, figs, smc_times, wall_times, startup_times, clock,
            prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rng;
            max_npass, refresh_logtarget, parallel, pbar_lines, vid_path, make_fig, verbose
        )
        (SMC_done || iter >= max_iter) && break
    end
    _SMC_LAST_END[] = time()

    if !isnothing(vid_path)
        VideoIO.save(
            joinpath(vid_path, "iters.mp4"),
            [CairoMakie.Colors.RGB.(colorbuffer(fig)) for fig in figs],
            framerate=3, encoder_options=(crf=23, preset="medium")
        )
    end
end


# Resumes a run_SMC that was interrupted, reading state from `fname``.
function resume_SMC(
    pop_size, target_ess, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    max_npass=10, refresh_logtarget=true, compute_pop_info=(particles, target, npass, prev_pop_info)->(;),
    verbose=0, vid_path=nothing, make_fig=nothing,
    parallel=:serial, pbar_lines=pop_size, rng=Random.default_rng(), max_iter=typemax(Int),
)
    myid() == 1 || return nothing  # workers only serve the master (see `start_SMC_workers`)
    parallel = resolve_backend(parallel)
    log_backend(parallel)
    prior_path_arg = prior_path
    @load fname all_particles iter targetinfos npass_vec smc_times
    # prior_path is updated during SMC, so use the saved one (older files lack it; fall back to the argument)
    prior_path = jldopen(f -> haskey(f, "prior_path") ? f["prior_path"] : prior_path_arg, fname)
    @assert length(all_particles) == (iter + 1)
    # older files lack wall times; their iterations and start-up count as NaN
    wall_times, startup_times = load_SMC_wall(fname)
    wall_times = something(wall_times, fill(NaN, iter))
    startup_times = something(startup_times, [NaN])

    figs = Figure[]
    if !isnothing(vid_path)
        mkpath(vid_path)
        figs = make_fig.(all_particles[2:end], 1:iter)   
    end

    @info "Resuming SMC by loading iter $iter"
    flush(stdout)
    flush(stderr)

    push!(startup_times, startup_time())
    clock = Ref(time())
    while true
        iter += 1
        SMC_done = SMC_iteration!(
            iter, pop_size, target_ess,
            all_particles, targetinfos, npass_vec, figs, smc_times, wall_times, startup_times, clock,
            prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rng;
            max_npass, refresh_logtarget, parallel, pbar_lines, vid_path, make_fig, verbose
        )
        (SMC_done || iter >= max_iter) && break
    end
    _SMC_LAST_END[] = time()

    if !isnothing(vid_path)
        VideoIO.save(
            joinpath(vid_path, "iters.mp4"),
            [CairoMakie.Colors.RGB.(colorbuffer(fig)) for fig in figs],
            framerate=3, encoder_options=(crf=23, preset="medium")
        )
    end
end


# Functions for processing SMC output

function load_SMC(fname)
    @load fname all_particles iter targetinfos npass_vec smc_times
    return all_particles, iter, targetinfos, npass_vec, smc_times
end

# Returns `(wall_times, startup_times)`, each `nothing` for files saved before they were recorded. The total wall time
# of a run (all segments) is `sum(startup_times) + sum(wall_times)`, apart from the last checkpoint write.
function load_SMC_wall(fname)
    return jldopen(fname) do f
        (haskey(f, "wall_times") ? f["wall_times"] : nothing, haskey(f, "startup_times") ? f["startup_times"] : nothing)
    end
end

# Log-odds of the conditional marginal posterior p(z_j = 1 | k_j, ψ) (eq:combi_cond_zpost) for a spike-and-slab
# coordinate x = log10 k_j, with inclusion probability logistic(logit_p). The log10 Jacobian is common to the spike
# and slab densities, so it cancels. Uses the globals `slab_prior` and `spike_prior`.
cond_incl_logodds(x, logit_p) = logpdf(slab_prior, x) - logpdf(spike_prior, x) + logit_p

# Monte Carlo estimate of the model posterior p(z | y) (eq:combi_zpost): the (weighted) average over particles of
# p(z | k_n, ψ_n) = Π_j p(z_j | k_{n,j}, ψ_n). `logit_p(θ)` returns the logit of the inclusion probability
# (0 for p = 1/2). Models are ordered as in `combinations(1:n_ss)`.
function get_mode_probs(states, Ws=ones(length(states)); logit_p=Returns(0.0))
    lookup = zeros(Int, 2^n_ss)
    for (i, elems) in enumerate(combinations(1:n_ss))
        idx = 0
        for elem in elems
            idx |= (1 << (elem - 1))
        end
        lookup[idx + 1] = i
    end

    probs = zeros(2^n_ss)
    q1 = zeros(n_ss) # p(z_j = 1 | k_j, ψ)
    q0 = zeros(n_ss) # p(z_j = 0 | k_j, ψ), computed separately for precision when q1 ≈ 1
    for (state, W) in zip(states, Ws)
        lp = logit_p(state)
        for j in 1:n_ss
            l = cond_incl_logodds(state[ss_idxs[j]], lp)
            q1[j], q0[j] = logistic(l), logistic(-l)
        end
        for idx in 0:(2^n_ss - 1)
            p = W
            for j in 1:n_ss
                p *= isodd(idx >> (j - 1)) ? q1[j] : q0[j]
            end
            probs[lookup[idx + 1]] += p
        end
    end
    return probs ./ sum(Ws)
end

# Implied-model approximation: each particle counts towards the single model with z_j = 1 iff θ[ss_idxs[j]] > thres.
function get_implied_mode_probs(states, thres, Ws=ones(length(states)))
    lookup = zeros(Int, 2^n_ss)
    for (i, elems) in enumerate(combinations(1:n_ss))
        idx = 0
        for elem in elems
            idx |= (1 << (elem - 1))
        end
        lookup[idx + 1] = i
    end

    counts = zeros(2^n_ss)
    for (state, W) in zip(states, Ws)
        idx = 0
        for j in 1:n_ss
            if state[ss_idxs[j]] > thres
                idx |= (1 << (j - 1))
            end
        end
        counts[lookup[idx + 1]] += W
    end
    return counts ./ sum(Ws)
end

function get_pvec(particles, Ws=ones(length(particles)); logit_p=Returns(0.0))
    return get_mode_probs(getproperty.(particles, :state), Ws; logit_p)
end

function make_fig(particles, iter, ss_idxs, ps)
    pop_size = length(particles)
    f = plot_pairs(
        getproperty.(particles, :state);
        title="Iteration $iter",
        figsize=(1000, 1000), skip_upper=true,
        axis_kwargs=(;), hist_axis_kwargs=(; yscale=identity),
        scatter_kwargs=(; markersize=4, alpha=clamp(1000 / pop_size * 0.3, 0.01, 0.5)),
        hist_kwargs=(bins=50,),
    )
    Box(f[ss_idxs, ss_idxs], color=(:green, 0.2), strokevisible=false)

    d = length(ps)
    idx = 0
    for (i1, p1) in enumerate(ps) # which row
        for (i2, p2) in enumerate(ps) # which column
            if i1 < i2
                continue
            end
            idx += 1
            ax = f.content[idx]
            if i1 == i2
                ax.yaxisposition = :right
                ax.yticklabelsize = 14
                ax.yticklabelpad = 0.5
                ax.yticksvisible = true
                ax.yticklabelsvisible = true
            elseif i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = L"\log_{10} %$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.yticklabelsize = 14
                ax.yticksvisible = true
                ax.yticklabelsvisible = true                
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 14
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end
    colgap!(f.layout, -8)
    return f
end