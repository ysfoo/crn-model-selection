include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using DiffResults, Optim

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

LinearAlgebra.BLAS.set_num_threads(1)

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 9

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[feasible_idxs[dir_idx]];
n_obs = length(data.t);
data_mat = hcat(data.data_E, data.data_L, data.data_A); # N x 3

rx_sys = models[end];
n_θ = length(parameters(rx_sys));
std_idxs = 1:3; # ODE parameters with standard prior
ss_idxs = 4:9; # ODE parameters with spike-and-slab prior
noise_idx = 10;
n_ss = length(ss_idxs);

base_oprob = ODEProblem(rx_sys, u0, (0.0, 10.0), [p => 1. for p in parameters(rx_sys)]; abstol=1e-6, reltol=1e-6);

param_idxs = map((x)->parameter_index(base_oprob, x).idx, parameters(rx_sys));
u0_idxs = map((x)->parameter_index(base_oprob, Initial(x)).idx, unknowns(rx_sys));

ode_params!(buf, θ) = for i in 1:n_θ
    buf[i] = exp10(θ[i]) 
end;

param_labels = [
    L"\lambda_{EL}", L"\lambda_{LA}", L"\rho", 
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A", L"\sigma"
];
sym2label = Dict(zip(Symbol.(parameters(models[end])), param_labels));

# esize = 4;
μ_noise, σ_noise = -1., 1.;
μ_slab, σ_slab = 0., 2.;
μ_spike = -16.; σ_spike = σ_slab;
thres = 0.5 * (μ_spike + μ_slab);
μ0 = -8.;

noise_prior = Normal(μ_noise, σ_noise)
slab_prior = Normal(μ_slab, σ_slab)
spike_prior = Normal(μ_spike, σ_spike)

final_ss_prior = MixtureModel([slab_prior, spike_prior]);
final_dists = [
    fill(slab_prior, length(std_idxs));
    fill(final_ss_prior, length(ss_idxs));
    noise_prior
];
final_logprior_func(θ) = sum(logpdf(dist, val) for (dist, val) in zip(final_dists, θ));

function make_ldp(logprior_func::LPF, γ::Float64, loglike_offset::Function=(θ)->0.) where LPF
    # The closures read the globals through local copies: reading non-const globals makes them type-unstable, and
    # with dual numbers this nearly doubled the allocations per gradient.
    loglike_func = let n_u = n_u, n_obs = n_obs, data_mat = data_mat, noise_idx = noise_idx
        function (sol, θ::AbstractVector{T}) where T
            σ = exp10(θ[noise_idx])
            ll = T(-n_u*n_obs*log(2π)/2)
            for i in 1:n_obs
                for j in 1:n_u
                    y_obs = data_mat[i,j]
                    y_pred = sol.u[i][j]
                    s = 0.01 + σ*max(0., y_pred)
                    ll -= ((y_obs - y_pred) / s)^2 / 2 + log(s)
                end
            end
            return (ll + loglike_offset(θ)) * γ
        end
    end
    ldp_params! = let n_θ = n_θ
        (buf, θ) -> for i in 1:n_θ
            buf[i] = exp10(θ[i])
        end
    end
    return OrdinaryDiffEqLDP(
        base_oprob, param_idxs, ldp_params!, θ -> logprior_func(θ, γ), loglike_func, n_θ;
        solver = AutoTsit5(Rodas5P()),
        solve_kwargs = (saveat=data.t, verbose=false),
    )
end

function run_10(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 1 ? 10 : 0
end

function run_20(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 1 ? 20 : 0
end

function repeat_5(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    if npass < 1
        return 5
    end
    nzeros = mean(p.info.sum_sqjdist == 0. for p in particles)
    return nzeros > 0.01 ? 5 : 0
end

function run_5_3(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 3 ? 5 : 0
end

function partition_by_thres(states::AbstractVector, thres::Float64; ss_idxs=4:9)
    classes = Dict{BitVector, Vector{Int}}()
    for (i, θ) in enumerate(states)
        key = BitVector(θ[j] > thres for j in ss_idxs)
        push!(get!(classes, key, Int[]), i)
    end
    return classes
end

function n_nuts_by_kendall(
    particles, last_n_nuts, npass, prev_states;
    thres=thres, init_n_nuts=10, min_count=20,
    verbose=0, rng=Random.default_rng(),
)
    if isnothing(last_n_nuts)
        return init_n_nuts
    end

    D = length(prev_states[1])
    curr_states = [p.state for p in particles]
    prev_classes = partition_by_thres(prev_states, thres)

    for (key, pre_idxs) in prev_classes
        n_idxs = length(pre_idxs)
        n_idxs < min_count && continue
        ties = collect(n for (_, n) in countmap(prev_states[pre_idxs]) if n > 1)
        pairs = n_idxs*(n_idxs-1)÷2
        tied_pairs = sum(t*(t-1)÷2 for t in ties; init=0)
        # cor_thres = kendall_qt(n_idxs, 1e-4, ties) + 0.2
        cor_thres = 0.1 + 0.9 * sqrt(min_count / n_idxs)
        for d in 1:D
            c = corkendall(getindex.(prev_states[pre_idxs], d), getindex.(curr_states[pre_idxs], d))
            if c > cor_thres
                if verbose > 0
                    @info "Rerun due to $key, dimension $d" (n_idxs, tied_pairs) (c, cor_thres)
                end
                return init_n_nuts
            end
        end
    end
    return 0
end

# Decide whether to extend NUTS based on detailed balance.
function n_nuts_by_detbal(
    particles, last_n_nuts, npass, states_before_move;
    thres=thres, init_n_nuts=10, 
    pseudocount=0.1, n_rand=10_000,
    verbose=0, rng=Random.default_rng(),
)
    if isnothing(last_n_nuts)
        return init_n_nuts
    end

    # if any(p->p.info.esjd == 0., particles)
    #     return 2 * last_n_nuts
    # end

    pop_size = length(particles)

    prev_states = [p.state .- p.info.delta for p in particles]
    curr_states = [p.state for p in particles]

    prev_classes = partition_by_thres(prev_states, thres)
    curr_classes = partition_by_thres(curr_states, thres)

    # Label the components (spike/slab modes).
    keys_union = unique([collect(keys(prev_classes)); collect(keys(curr_classes))])
    comp_id = Dict(k => i for (i, k) in enumerate(keys_union))

    prev_comp = Vector{Int}(undef, pop_size)
    curr_comp = Vector{Int}(undef, pop_size)
    for (key, inds) in prev_classes
        prev_comp[inds] .= comp_id[key]
    end
    for (key, inds) in curr_classes
        curr_comp[inds] .= comp_id[key]
    end

    # -1 if isolated from >thres to <thres, 1 if opposite, 0 if stayed on same side of thres. 
    diff_mat = stack([keys_union[curr_comp[i]] .- keys_union[prev_comp[i]] for i in 1:pop_size])
    # Number of switching particles in each dimension.
    switch_vec = vec(sum(abs, diff_mat; dims=2))
    # Net flow from <thres to >thres.
    net_vec = vec(sum(diff_mat; dims=2))
    # p-value of binomial test on each dimension, null hypothesis is #(positive switches) ~ Bin(#(switches), 1/2).
    binoms = Binomial.(switch_vec, 0.5);
    pvals = [min(1., 2cdf(binom, (switch-abs(net))÷2)) for (switch, net, binom) in zip(switch_vec, net_vec, binoms)]
    # Switches are dependent across dimensions, use randomization to find critical value under null hypothesis.
    bootstrap_pmins = Float64[]
    D, N = size(diff_mat)
    cache = Vector{Int}(undef, D);
    rand_bits = BitVector(undef, N);
    cdf_cache = [Dict{Int,Float64}() for _ in 1:D];
    @elapsed for _ in 1:n_rand
        for j in 1:D
            cache[j] = 0
        end
        rand!(rng, rand_bits)
        for i in 1:N
            sgn = rand_bits[i] ? 1 : -1
            for j in 1:D
                cache[j] += sgn*diff_mat[j,i]
            end
        end
        pmin = 1.0
        @inbounds for j in 1:D
            k = (switch_vec[j] - abs(cache[j])) ÷ 2
            cdf_lookup = cdf_cache[j]
            p = get!(cdf_lookup, k) do
                min(1.0, 2cdf(binoms[j], k))
            end
            pmin = min(pmin, p)
        end
        push!(bootstrap_pmins, pmin)
    end
    overall_pval = mean(minimum(pvals) .>= bootstrap_pmins)
    reject = overall_pval < pseudocount

    if verbose > 0
        # pvals_str = string(round.(pvals; digits=4))
        @info "Pass $npass" reject minimum(pvals) overall_pval string(switch_vec) string(net_vec)
    end

    return reject ? init_n_nuts : 0
end


## Nutpie-style mass matrix
function compute_invmass(particles, target, npass)
    logp_func = Base.Fix1(LogDensityProblems.logdensity, target)

    x1 = particles[1].state
    d, n = length(x1), length(particles)
    T = eltype(x1)

    X = Matrix{T}(undef, d, n)
    G = Matrix{T}(undef, d, n)
    logp = Vector{T}(undef, n)

    result = DiffResults.GradientResult(x1)
    for (i, p) in enumerate(particles)
        X[:, i] .= p.state
        result = ForwardDiff.gradient!(result, logp_func, p.state)  # value + grad in one pass
        logp[i] = DiffResults.value(result)
        G[:, i] .= DiffResults.gradient(result)
    end

    keep = logp .>= quantile(logp, 0.01)   # drop bottom 1% by log target density
    var_x     = vec(var(view(X, :, keep); dims = 2))
    var_alpha = vec(var(view(G, :, keep); dims = 2))
    invmass = sqrt.(var_x ./ var_alpha)
    # display([sqrt.(var_x) sqrt.(var_alpha) invmass])
    return (invmass = invmass,)   # inverse mass diagonal
end

## Population info for `nuts_invmass_move`: nutpie-style mass matrix plus the pool of step sizes
# Use previous pool of step sizes weighted by how close the acceptance rate of the ancestor's last move was to the target.
acc_weight(a; p=4, q=1) = isfinite(a) ? clamp(a, 0., 1.)^p * (1 - clamp(a, 0., 1.))^q : 0.

# Particles without acceptance info (first iteration, no usable weights) give a uniform pool with a wide jitter.
# Particles with `curr_acc_rate < min_acc_rate` can additionally be excluded outright (disabled by default); if 
# that would exclude everyone, the filter is ignored.
function compute_FT_info(
    particles, target, npass; jitter=0.2, init_jitter=0.2, min_acc_rate=0.0,
    weight_func=acc_weight,
)
    stepsizes = [p.stepsize for p in particles]
    accs = [
        hasproperty(p.info, :curr_acc_rate) ? p.info.curr_acc_rate : NaN
        for p in particles
    ]
    valid_step = isfinite.(stepsizes) .& (stepsizes .> 0)
    acc_ok = .!(accs .< min_acc_rate)
    any(valid_step .& acc_ok) && (valid_step = valid_step .& acc_ok)
    ws = [valid_step[i] ? weight_func(accs[i]) : 0. for i in eachindex(particles)]
    if sum(ws) > 0
        used_jitter = jitter
    else
        ws = Float64.(valid_step)
        used_jitter = init_jitter
    end
    cum_probs = cumsum(ws ./ sum(ws))
    cum_probs[end] = 1.0
    return merge(
        compute_invmass(particles, target, npass),
        (stepsizes = stepsizes, cum_probs = cum_probs, jitter = used_jitter),
    )
end


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

function compute_QB_info(
    particles, target, npass; acc_target=0.8, jitter=0.2, max_dlog=Inf, use_ngrad=false, min_obs=20,
)
    stepsizes = [p.stepsize for p in particles]
    valid = filter(s -> isfinite(s) && s > 0, stepsizes)
    stepsize = exp(mean(log.(valid))) # nominal stepsize is previous stepsize

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
    return merge(
        compute_invmass(particles, target, npass),
        (stepsize = stepsize, jitter = jitter),
    )
end


## Fisher-Rao path
abstract type FRStrategy end

mutable struct FRPath{S<:FRStrategy,F0,F1,G}
    init_ss_logprior::F0
    final_ss_logprior::F1
    ϕ::Float64 # acos(BC)
    add_logprior::G
    prev_u::Float64
    prev_γ::Float64
    strategy::S
end

function compute_u(p::FRPath, γ)
    γ <= p.prev_γ && return p.prev_u
    u = raw_u(p.strategy, p.prev_u, p.prev_γ, γ)
    return max(u, p.prev_u)
end

function update!(p::FRPath, prev_γ, iter, all_particles)
    p.prev_γ = prev_γ
    update_strategy!(p.strategy, prev_γ, iter, all_particles)
    return nothing
end

function get_targetinfo!(γ, iter, p::FRPath)
    u = compute_u(p, γ)
    log_schedule(iter, p.strategy, γ, p.prev_u, u)
    p.prev_u = u
    return (γ = γ, u = u)
end

# Prior on spike-and-slab parameters
function FR_logprior(p::FRPath, θ, γ)
    logp_init = p.init_ss_logprior(θ)
    logp_final = p.final_ss_logprior(θ)
    u = compute_u(p, γ)
    try
        return 2 * (logaddexp(log(sin((1-u)*p.ϕ)) + logp_init/2, log(sin(u*p.ϕ)) + logp_final/2) - log(sin(p.ϕ)))
    catch e
        @info "FR error" u p.prev_u γ p.prev_γ
        flush(stdout)
        rethrow()
    end
end

# Prior on non spike-and-slab parameters
function add_logprior(θ)
    lp = 0.0
    @inbounds for i in std_idxs
        lp += logpdf(slab_prior, θ[i])
    end
    lp += logpdf(noise_prior, θ[noise_idx])
    return lp
end

(p::FRPath)(θ, γ) = FR_logprior(p, θ, γ) + p.add_logprior(θ)


## Strategy 1: guess how many SMC iterations are left
mutable struct IterGuessStrat <: FRStrategy
    prev_iter::Int
end

function raw_u(s::IterGuessStrat, prev_u, prev_γ, γ)
    iter_left = (1 - prev_γ) / (γ - prev_γ)
    return (s.prev_iter + 1) / (s.prev_iter + iter_left)
end

function update_strategy!(s::IterGuessStrat, prev_γ, iter, all_particles) 
    (s.prev_iter = iter - 1; nothing)
end

function log_schedule(iter, ::IterGuessStrat, γ, prev_u, u) 
    @info "Iter $iter prior schedule" prev_u u
end

## Strategy 2: match u's increase to the decrease in NUTS mode-switching rate
mutable struct SwitchRateStrat <: FRStrategy
    prev_cor::Float64
    isolated::Bool
end

function raw_u(s::SwitchRateStrat, prev_u, prev_γ, γ; cor_thres=0.9)
    isnan(s.prev_cor) && return 0.
    prev_γ >= 1. && return 1.
    return if s.isolated
        1 - (1-prev_u) * (1-γ) / (1-max(cor_thres, prev_γ))
    else
        1 - (1-prev_u) * (1-γ) / (1-max(0., s.prev_cor))
    end
end

function update_strategy!(s::SwitchRateStrat, prev_γ, iter, all_particles; cor_thres=0.9)
    iter == 1 && return nothing
    s.prev_cor = compute_mincor(all_particles)
    if s.prev_cor >= cor_thres
        s.isolated = true
    end
    return nothing
end

function log_schedule(iter, s::SwitchRateStrat, γ, prev_u, u)
    @info "Iter $iter prior schedule" (s.isolated, s.prev_cor) prev_u u
end

function compute_mincor(all_particles; ss_idxs=ss_idxs, thres=thres, pseudocount=0.5)
    particles = all_particles[end]
    N = length(particles)
    cors = Float64[]
    minprops = Float64[]
    for d in ss_idxs
        pre  = [p.state[d] - p.info.delta[d] > thres for p in particles]
        post = [p.state[d] > thres for p in particles]
        n11 = sum(pre .& post) + pseudocount
        n10 = sum(pre .& .!post) + pseudocount
        n01 = sum(.!pre .& post) + pseudocount
        n00 = sum(.!pre .& .!post) + pseudocount
        row1, row0 = n11+n10, n01+n00
        col1, col0 = n11+n01, n10+n00
        φ = (n11*n00 - n10*n01) / sqrt(row1*row0*col1*col0)
        push!(cors, φ)
        push!(minprops, (min(col1, col0) - 2pseudocount)/N)
    end
    println(round.(cors; digits=3))
    return minimum(cors[findall(minprops .>= 0.01)])
end

## Convenience constructors
FRPath(init_ss_logprior, final_ss_logprior, ϕ, add_logprior, ::Type{IterGuessStrat}) =
    FRPath(init_ss_logprior, final_ss_logprior, ϕ, add_logprior, 0., 0., IterGuessStrat(0))

FRPath(init_ss_logprior, final_ss_logprior, ϕ, add_logprior, ::Type{SwitchRateStrat}) =
    FRPath(init_ss_logprior, final_ss_logprior, ϕ, add_logprior, 0., 0., SwitchRateStrat(NaN, false))