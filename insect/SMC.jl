include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using Distributions, LinearAlgebra, Optim
using JLD2

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

function sq_hellinger(dist1, dist2)
    avg_var = (dist1.σ^2 + dist2.σ^2) / 2
    overlap = sqrt(dist1.σ*dist2.σ/avg_var) * exp(-(dist1.μ - dist2.μ)^2/(8avg_var))
    return 1 - overlap
end

function make_ldp(logprior_func::LPF, β::Float64, loglike_offset::Function=(θ)->0.) where LPF
    function loglike_func(sol, θ::AbstractVector{T}) where T
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
        return (ll + loglike_offset(θ)) * β
    end
    return OrdinaryDiffEqLDP(
        base_oprob, param_idxs, ode_params!, θ -> logprior_func(θ, β), loglike_func, n_θ;
        solver = AutoTsit5(Rodas5P()),
        solve_kwargs = (saveat=data.t, verbose=false),
    )
end

function run_10(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 1 ? 10 : 0
end

function partition_by_thres(states::AbstractVector, thres::Float64; ss_idxs=ss_idxs)
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
    α=0.1, n_rand=10_000,
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

    # -1 if switched from >thres to <thres, 1 if opposite, 0 if stayed on same side of thres. 
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
    reject = overall_pval < α

    if verbose > 0
        # pvals_str = string(round.(pvals; digits=4))
        @info "Pass $npass" reject minimum(pvals) overall_pval string(switch_vec) string(net_vec)
    end

    return reject ? init_n_nuts : 0
end


## Fisher-Rao path
mutable struct FRPriorPath{F0,F1,G}
    init_ss_logprior::F0
    final_ss_logprior::F1
    ϕ::Float64 # acos(BC)
    add_logprior::G
    prev_u::Float64
    prev_β::Float64
    prev_iter::Int
end

function compute_u(p::FRPriorPath, β)
    β <= p.prev_β && return p.prev_u
    iter_left = β == 1. ? 1 : (1 - p.prev_β) / (β - p.prev_β) # avoid division by zero
    u = (p.prev_iter + 1) / (p.prev_iter + iter_left)
    return max(u, p.prev_u)
end

function FR_logprior(p::FRPriorPath, θ, β)
    logp_init = p.init_ss_logprior(θ)
    logp_final = p.final_ss_logprior(θ)
    u = compute_u(p, β)
    try
        logp = 2 * (logaddexp(log(sin((1-u)*p.ϕ)) + logp_init / 2, log(sin(u*p.ϕ)) + logp_final / 2) - log(sin(p.ϕ))) 
    catch e
        @info "FR error" u p.prev_u β p.prev_β p.prev_iter
        flush(stdout)
        rethrow()
    end
end

function update!(p::FRPriorPath, curr_β, iter)
    u = compute_u(p, curr_β)
    @info "Iter $iter prior schedule" p.prev_u u
    p.prev_u = u
    p.prev_β = curr_β
    p.prev_iter = iter
end

(p::FRPriorPath)(θ, β) = FR_logprior(p, θ, β) + p.add_logprior(θ)

# Prior on non spike-and-slab parameters
function add_logprior(θ)
    lp = 0.0
    @inbounds for i in std_idxs
        lp += logpdf(slab_prior, θ[i])
    end
    lp += logpdf(noise_prior, θ[noise_idx])
    return lp
end
