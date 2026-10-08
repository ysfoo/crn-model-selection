include(joinpath(@__DIR__, "SMC.jl"))
using StatsFuns

# Likelihood and prior tempering. Given p, the prior of each spike-and-slab parameter η is always a mixture of
# two Gaussians with fixed weights (p for slab, 1-p for spike), so it is normalised for every p and the
# marginal prior of p stays Logistic along the path. At γ = 0 both components equal the moment-matched
# Gaussian N(m(p), v(p)); at γ = 1 they are the final slab and spike. Component k has
#   variance  s_k²(γ) = (1-γ) v(p) + γ σ_k²,
#   mean      μ_k(γ)  = m(p) + sqrt(γ) (μ_k - m(p)),
# which keeps the mean and variance of the mixture at m(p) and v(p) for all γ.
run_str = "SMC305"

# Mean and variance of p * slab + (1-p) * spike (law of total variance). At p = 1/2 with the current
# hyperparameters, these are -8 and 68.
unimodal_mean(p) = p*μ_slab + (1-p)*μ_spike
unimodal_var(p) = p*σ_slab^2 + (1-p)*σ_spike^2 + p*(1-p)*(μ_slab-μ_spike)^2
unimodal_prior(p) = Normal(unimodal_mean(p), sqrt(unimodal_var(p)))

# Sample from the initial prior: θ_p ~ Logistic, η ~ unimodal prior given p, noise parameters ~ `noise_prior`.
function init_sampler(rng)
    θ = Vector{Float64}(undef, n_θ)
    θ[p_idx] = rand(rng, p_prior)
    uni_prior = unimodal_prior(logistic(θ[p_idx]))
    for i in ss_idxs
        θ[i] = rand(rng, uni_prior)
    end
    for i in noise_idxs
        θ[i] = rand(rng, noise_prior)
    end
    return θ
end

move_func = nuts_invmass_move;  init_stepsize = 0.5;
n_nuts_func = repeat_5;
ldp_builder = make_ldp;
prior_path(θ, γ) = begin
    logp, log1mp = loglogistic(θ[p_idx]), log1mlogistic(θ[p_idx])
    p = logistic(θ[p_idx])
    m, v = unimodal_mean(p), unimodal_var(p)
    μ_slab_γ = m + sqrt(γ)*(μ_slab - m)
    μ_spike_γ = m + sqrt(γ)*(μ_spike - m)
    σ_slab_γ = sqrt((1-γ)*v + γ*σ_slab^2)
    σ_spike_γ = sqrt((1-γ)*v + γ*σ_spike^2)
    sum(
        begin
            x = θ[d]
            logaddexp(
                normlogpdf(μ_spike_γ, σ_spike_γ, x) + log1mp,
                normlogpdf(μ_slab_γ, σ_slab_γ, x) + logp
            )
        end for d in ss_idxs
    ) + add_logprior(θ)
end

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])
# seed = 1

psize_str = "8k"; pop_size = 8000; target_ess = 0.8pop_size; init_pop_size = pop_size;

fname = joinpath(mkpath(joinpath(@__DIR__, "output/seed$seed")), "$(run_str)_$(psize_str).jld2");
display(fname)

rng = StableRNG(hash((run_str, pop_size, seed)))
run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_stepsize=init_stepsize, init_pop_size=init_pop_size, max_npass=10, compute_pop_info=compute_QB_info,
    verbose=1, rng=rng, parallel=true, pbar_lines=20,
);
