include(joinpath(@__DIR__, "stats_helpers.jl"));
include(joinpath(@__DIR__, "plot_helpers.jl"));

using Distributions, Random, StableRNGs, Statistics, StatsBase
using Distances, LinearAlgebra, LogExpFunctions, NearestNeighbors
using AdvancedHMC, Bijectors, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, PreallocationTools
using Accessors, JLD2, ProgressMeter
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

# Squared jump distance per NUTS iteration; runs saved before `sum_sqjdist` was introduced store it as `info.esjd`.
esjd_per_nuts(p::SMCParticle) = hasproperty(p.info, :esjd) ? p.info.esjd : p.info.sum_sqjdist / p.n_nuts

no_adapt_func(info) = 1.
adapt_using_agg(info) = 1.5^((info.agg_acc_rate-0.8)/0.2)
adapt_using_curr(info) = 1.5^((info.curr_acc_rate-0.8)/0.2)


function perform_moves!(
    particles, target, move_func, n_nuts_func, states_before_move, compute_pop_info,
    iter, targetinfo, rngs;
    max_npass, pop_size, pbar_lines, parallel, verbose
)
    npass = 0
    move_time = 0.0
    last_n_nuts = nothing
    while npass < max_npass
        n_nuts = n_nuts_func(particles, last_n_nuts, npass, states_before_move; verbose, rng = rngs[1])
        n_nuts == 0 && break

        npass += 1
        last_n_nuts = n_nuts
        pop_info = compute_pop_info(particles, target, npass)

        pbar = Progress(pop_size; desc="Iter $iter, pass $(npass)")
        pbar_step = max(1, pop_size ÷ pbar_lines)
        if parallel
            counter = Threads.Atomic{Int}(0)
            pbar_lock = Threads.SpinLock()
            targets = [customcopy(target) for _ in 1:pop_size]
            thread_times = zeros(Threads.nthreads())
            GC.gc(false)

            Threads.@threads :static for i in 1:pop_size
                particle = particles[i]
                tid = Threads.threadid()
                thread_times[tid] += @elapsed begin
                    particles[i] = move_func(rngs[tid], particle, targets[i], n_nuts, pop_info)
                end
                prev_done = Threads.atomic_add!(counter, 1)
                if (prev_done + 1) % pbar_step == 0
                    lock(pbar_lock) do
                        ProgressMeter.update!(pbar, prev_done + 1)
                    end
                end
            end

            if verbose > 1
                display(thread_times)
            end
            move_time += sum(thread_times)
        else
            rng = rngs[1]
            for i in 1:pop_size
                particle = particles[i]
                move_time += @elapsed particles[i] = move_func(rng, particle, target, n_nuts, pop_info)
                if i % pbar_step == 0
                    ProgressMeter.update!(pbar, i)
                end
            end
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
function refresh_logtargets(particles, ldp_builder, prior_path, γ; parallel=false)
    N = length(particles)
    target = ldp_builder(prior_path, γ)
    logtargets = Vector{Float64}(undef, N)
    if parallel
        targets = [customcopy(target) for _ in 1:Threads.nthreads()]
        Threads.@threads :static for i in 1:N
            logtargets[i] = LogDensityProblems.logdensity(targets[Threads.threadid()], particles[i].state)
        end
    else
        for i in 1:N
            logtargets[i] = LogDensityProblems.logdensity(target, particles[i].state)
        end
    end
    return [
        begin
            lt = isfinite(logtargets[i]) ? logtargets[i] : -Inf
            ll = isfinite(lt) ? (lt - prior_path(p.state, γ)) / γ : -Inf
            SMCParticle(p.state, ll, lt, p.stepsize, p.n_nuts, p.info)
        end for (i, p) in enumerate(particles)
    ]
end


# Runs one SMC iteration: reweight → resample → move.
# Modifies all_particles, targetinfos, npass_vec, figs, smc_times.
# Returns whether to terminate SMC.
function SMC_iteration!(
    iter, pop_size, target_ess,
    all_particles, targetinfos, npass_vec, figs, smc_times,
    prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rngs; 
    max_npass=10, refresh_logtarget=true,
    parallel=false, pbar_lines=pop_size, vid_path=nothing, make_fig=nothing, verbose=0
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
    
    rng = rngs[1]   
    γ = targetinfo.γ
    push!(targetinfos, targetinfo)
    
    prev_logtarget_vec = getproperty.(particles, :logtarget)
    logtarget_vec = map(p -> LogDensityProblems.logdensity(target, p.state), particles)
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
        particles, target, move_func, n_nuts_func, states_before_move, compute_pop_info,
        iter, targetinfo, rngs;
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

    @save fname all_particles iter targetinfos npass_vec smc_times prior_path

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

function run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_pop_size=pop_size, init_stepsize::Float64=0.01, 
    max_npass=10, refresh_logtarget=true, compute_pop_info=(particles, target, npass)->(;),
    verbose=0, vid_path=nothing, make_fig=nothing,
    parallel=false, pbar_lines=pop_size, rng=Random.default_rng()
)
    if parallel
        rngs = [StableRNG(rand(rng, UInt64)) for _ in 1:Threads.nthreads()]
    else
        rngs = [rng]
    end
    initstates = [init_sampler(rng) for _ in 1:init_pop_size] 

    iter = 0
    logpriors = prior_path.(initstates, 0.)

    # use γ = 1 to extract likelihood
    tmp_target = ldp_builder(prior_path, 1.) 
    tmp_logpriors = prior_path.(initstates, 1.)
    tmp_loglikes = LogDensityProblems.logdensity.(Ref(tmp_target), initstates) .- tmp_logpriors

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
    npass_vec, smc_times = Int64[], Float64[]

    figs = Figure[]
    if !isnothing(vid_path)        
        mkpath(vid_path)
    end

    while true
        iter += 1
        SMC_done = SMC_iteration!(
            iter, pop_size, target_ess,
            all_particles, targetinfos, npass_vec, figs, smc_times,
            prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rngs;
            max_npass, refresh_logtarget, parallel, pbar_lines, vid_path, make_fig, verbose
        )
        SMC_done && break
    end

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
    max_npass=10, refresh_logtarget=true, compute_pop_info=(particles, target, npass)->(;),
    verbose=0, vid_path=nothing, make_fig=nothing, 
    parallel=false, pbar_lines=pop_size, rng=Random.default_rng()
)
    prior_path_arg = prior_path
    @load fname all_particles iter targetinfos npass_vec smc_times
    # prior_path is updated during SMC, so use the saved one (older files lack it; fall back to the argument)
    prior_path = jldopen(f -> haskey(f, "prior_path") ? f["prior_path"] : prior_path_arg, fname)
    @assert length(all_particles) == (iter + 1)
    if parallel
        rngs = [StableRNG(rand(rng, UInt64)) for _ in 1:Threads.nthreads()]
    else
        rngs = [rng]
    end

    figs = Figure[]
    if !isnothing(vid_path)
        mkpath(vid_path)
        figs = make_fig.(all_particles[2:end], 1:iter)   
    end

    @info "Resuming SMC by loading iter $iter"
    flush(stdout)
    flush(stderr)

    while true
        iter += 1
        SMC_done = SMC_iteration!(
            iter, pop_size, target_ess,
            all_particles, targetinfos, npass_vec, figs, smc_times,
            prior_path, ldp_builder, move_func, n_nuts_func, compute_pop_info, fname, rngs;
            max_npass, refresh_logtarget, parallel, pbar_lines, vid_path, make_fig, verbose
        )
        SMC_done && break
    end

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

function get_mode_probs(states, thres, Ws=ones(length(states)))
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

function get_mode_probs_all(states, prob_func, Ws=ones(length(states)))
    lookup = zeros(Int, 2^n_ss)
    for (i, elems) in enumerate(combinations(1:n_ss))
        idx = 0
        for elem in elems
            idx |= (1 << (elem - 1))
        end
        lookup[idx + 1] = i
    end

    probs = zeros(2^n_ss)
    @showprogress for (state, W) in zip(states, Ws)
        for idx in 1:(2^n_ss)
            p = 1.
            for j in 1:n_ss
                p_factor = prob_func(state[ss_idxs[j]])
                incl_j = isodd((idx-1) >> (j-1))
                p *= incl_j ? p_factor : (1- p_factor)
            end
            probs[lookup[idx]] += p*W
        end        
    end
    return probs ./ sum(Ws)
end 

function get_pvec(particles, thres, Ws=ones(length(particles)))
    return get_mode_probs(getproperty.(particles, :state), thres, Ws)
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