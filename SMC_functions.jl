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


function nuts_move(rng, particle, target, n_nuts; adapt_stepsize_func=adapt_using_curr)
    metric = DiagEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target, ForwardDiff)

    stepsize = particle.stepsize
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, NoAdaptation(), 0; verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
    else
        sum_sqjdist = particle.info.esjd * particle.n_nuts
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
    sum_sqjdist += sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        sum_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        esjd = sum_sqjdist / agg_n_nuts,
        delta = delta
    )

    # adapt stepsize after NUTS using most recent acceptance rate
    stepsize_factor = adapt_stepsize_func(info)
    if isfinite(stepsize_factor)
        stepsize *= stepsize_factor
    end
    logtarget = stats[end].log_density

    return SMCParticle(θs[end], NaN, isfinite(logtarget) ? logtarget : -Inf, stepsize, agg_n_nuts, info)
end

struct PolyakRuppertAveraging{T<:NesterovDualAveraging} <: StepSizeAdaptor
    inner::T
end
PolyakRuppertAveraging(args...) = PolyakRuppertAveraging(NesterovDualAveraging(args...))

AdvancedHMC.getϵ(a::PolyakRuppertAveraging) = exp.(a.inner.state.x_bar)
AdvancedHMC.adapt!(a::PolyakRuppertAveraging, θ, α) = AdvancedHMC.adapt!(a.inner, θ, α)
AdvancedHMC.reset!(a::PolyakRuppertAveraging) = (AdvancedHMC.reset!(a.inner); a)
AdvancedHMC.finalize!(a::PolyakRuppertAveraging) = (AdvancedHMC.finalize!(a.inner); a)

function nuts_PRA_move(rng, particle, target, n_nuts; adapt_stepsize_func=no_adapt_func)
    metric = DiagEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target, ForwardDiff)

    stepsize = particle.stepsize
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    pra = particle.n_nuts > 0 ? PolyakRuppertAveraging(0.05, 10., 0.75, 0.8, particle.info.das) : PolyakRuppertAveraging(0.8, stepsize)

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, pra, n_nuts, verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
    else
        sum_sqjdist = particle.info.esjd * particle.n_nuts
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
    sum_sqjdist += sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        sum_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        esjd = sum_sqjdist / agg_n_nuts,
        delta = delta,
        das = pra.inner.state
    )
    
    logtarget = stats[end].log_density

    return SMCParticle(θs[end], NaN, isfinite(logtarget) ? logtarget : -Inf, AdvancedHMC.getϵ(pra), agg_n_nuts, info)
end

function nuts_NDA_move(rng, particle, target, n_nuts; adapt_stepsize_func=no_adapt_func)
    metric = DiagEuclideanMetric(LogDensityProblems.dimension(target))
    h = Hamiltonian(metric, target, ForwardDiff)

    stepsize = particle.stepsize
    integrator = Leapfrog(stepsize)
    κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    nda = particle.n_nuts > 0 ? NesterovDualAveraging(0.05, 10., 0.75, 0.8, particle.info.das) : NesterovDualAveraging(0.8, stepsize)

    θs, stats = sample(rng, h, κ, particle.state, n_nuts, nda, n_nuts, verbose=false)

    if particle.n_nuts == 0
        sum_sqjdist = 0.
        sum_acc_rate = 0.
    else
        sum_sqjdist = particle.info.esjd * particle.n_nuts
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
    sum_sqjdist += sum(abs2, θs[1] .- particle.state)
    for i in 2:n_nuts
        sum_sqjdist += sum(abs2, θs[i] .- θs[i-1])
    end

    info = (
        curr_acc_rate = acc_rate_incr / n_nuts,
        agg_acc_rate = sum_acc_rate / agg_n_nuts,
        esjd = sum_sqjdist / agg_n_nuts,
        delta = delta,
        das = nda.state
    )
    
    logtarget = stats[end].log_density

    return SMCParticle(θs[end], NaN, isfinite(logtarget) ? logtarget : -Inf, exp(nda.state.x_bar), agg_n_nuts, info)
end

no_adapt_func(info) = 1.
adapt_using_agg(info) = 1.5^((info.agg_acc_rate-0.8)/0.2)
adapt_using_curr(info) = 1.5^((info.curr_acc_rate-0.8)/0.2)


function perform_moves!(
    particles, target, move_func, n_nuts_func, states_before_move,
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
                    particles[i] = move_func(rngs[tid], particle, targets[i], n_nuts)
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
                move_time += @elapsed particles[i] = move_func(rng, particle, target, n_nuts)
                if i % pbar_step == 0
                    ProgressMeter.update!(pbar, i)
                end
            end
        end
        ProgressMeter.finish!(pbar)
    end
    return npass, move_time
end


# Runs one SMC iteration: reweight → resample → move.
# Modifies all_particles, targetinfos, npass_vec, figs, smc_times.
# Returns whether to terminate SMC.
function SMC_iteration!(
    iter, pop_size, target_ess,
    all_particles, targetinfos, npass_vec, figs, smc_times,
    logprior_func, ldp_builder, move_func, n_nuts_func, fname, rngs; 
    max_npass=10,
    parallel=false, pbar_lines=pop_size, vid_path=nothing, make_fig=nothing, verbose=0
)
    t_start = time()

    # Reweight    
    particles = all_particles[end]
    prev_targetinfo = targetinfos[end]
    target, targetinfo = build_target_SMC!(
        iter, all_particles, prev_targetinfo, target_ess, logprior_func, ldp_builder; 
        verbose
    )
    if isnothing(target)
        return true
    end
    
    rng = rngs[1]   
    β, = targetinfo
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
            SMCParticle(p.state, NaN, NaN, p.stepsize, 0, NamedTuple())
        end for idx in idxs
    ]
    states_before_move = getproperty.(particles, :state)

    # Move
    smc_time = time() - t_start
    npass, move_time = perform_moves!(
        particles, target, move_func, n_nuts_func, states_before_move,
        iter, targetinfo, rngs;
        max_npass, pop_size, pbar_lines, parallel, verbose
    )
    smc_time += move_time
    
    particles = [@set p.loglike = (p.logtarget - logprior_func(p.state, β)) / β for p in particles]
    push!(all_particles, particles)

    agg_acc_rate = mean(filter(isfinite, [p.info.agg_acc_rate for p in particles]))
    median_esjd = median([p.info.esjd for p in particles])

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

    @save fname all_particles iter targetinfos npass_vec smc_times

    return false
end


function build_target_SMC!(
    iter, all_particles, prev_targetinfo, target_ess, logprior_func, ldp_builder; 
    verbose=0, Δβ=1e-8,
)
    prev_β, = prev_targetinfo
    particles = all_particles[end]

    @assert iter > 0    

    if prev_β >= 1.0
        return (nothing, nothing)
    end

    # Tempering
    states = getproperty.(particles, :state)
    loglikes = getproperty.(particles, :loglike)
    prev_logtargets = getproperty.(particles, :logtarget)
    
    ess_final = compute_ess(logprior_func.(states, 1.) .+ loglikes .- prev_logtargets)
    if ess_final >= target_ess
        curr_β = 1.
    else
        curr_β = IVT_search(
            target_ess, 
            (β) -> compute_ess(logprior_func.(states, β) .+ β .* loglikes .- prev_logtargets), 
            max(Δβ, prev_β), 1.; tol=Δβ,
        )
    end
    update!(logprior_func, curr_β, iter)
    curr_ess = compute_ess(logprior_func.(states, curr_β) .+ curr_β .* loglikes .- prev_logtargets)

    if verbose > 0
        @info "Iter $iter tempering" curr_β curr_ess
        flush(stdout)
        flush(stderr)
    end

    return (ldp_builder(logprior_func, curr_β), (curr_β,))
end

function run_SMC(
    pop_size, target_ess, init_sampler, logprior_func, ldp_builder, move_func, n_nuts_func, fname;
    init_pop_size=pop_size, init_stepsize::Float64=0.01, 
    max_npass=10,
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
    logpriors = logprior_func.(initstates, 0.)

    # use β = 1 to extract likelihood
    tmp_target = ldp_builder(logprior_func, 1.) 
    tmp_logpriors = logprior_func.(initstates, 1.)
    tmp_loglikes = LogDensityProblems.logdensity.(Ref(tmp_target), initstates) .- tmp_logpriors

    # finite_loglikes = filter(isfinite, tmp_loglikes)
    # finite_tmp_logpriors = filter(isfinite, tmp_logpriors)
    # @info "Initial finite checks" length(finite_loglikes) length(finite_tmp_logpriors)
    flush(stdout)

    all_particles = [SMCParticle.(
        initstates,
        tmp_loglikes,
        logpriors, # actual logtarget has β = 0
        init_stepsize, 0, Ref(NamedTuple())
    )]
    targetinfos = [(0.,)]
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
            logprior_func, ldp_builder, move_func, n_nuts_func, fname, rngs;
            max_npass, parallel, pbar_lines, vid_path, make_fig, verbose
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
    pop_size, target_ess, logprior_func, ldp_builder, move_func, n_nuts_func, fname;
    max_npass=10,
    verbose=0, vid_path=nothing, make_fig=nothing, 
    parallel=false, pbar_lines=pop_size, rng=Random.default_rng()
)
    @load fname all_particles iter targetinfos npass_vec smc_times
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
            logprior_func, ldp_builder, move_func, n_nuts_func, fname, rngs;
            max_npass, parallel, pbar_lines, vid_path, make_fig, verbose
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

function partition_by_thres(states::AbstractVector, thres::Float64)
    classes = Dict{BitVector, Vector{Int}}()
    for (i, θ) in enumerate(states)
        mask = BitVector(θ[j] > thres for j in ss_idxs)
        push!(get!(classes, mask, Int[]), i)
    end
    return classes
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