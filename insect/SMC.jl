include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../process_setup.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using DiffResults, Optim

# Workers for `parallel = :distributed`, thread pinning and single-threaded BLAS (see process_setup.jl)
setup_SMC_process()

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

function n_nuts_by_kendall(
    particles, last_n_nuts, npass, prev_states;
    thres=thres, ss_idxs=ss_idxs, init_n_nuts=10, min_count=20,
    verbose=0, rng=Random.default_rng(),
)
    if isnothing(last_n_nuts)
        return init_n_nuts
    end

    D = length(prev_states[1])
    curr_states = [p.state for p in particles]
    prev_classes = partition_by_thres(prev_states, thres, ss_idxs)

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
    thres=thres, init_n_nuts=10, ss_idxs=ss_idxs,
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

    prev_classes = partition_by_thres(prev_states, thres, ss_idxs)
    curr_classes = partition_by_thres(curr_states, thres, ss_idxs)

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