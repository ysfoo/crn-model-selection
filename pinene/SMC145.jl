include(joinpath(@__DIR__, "SMC.jl"))
using StatsFuns

# SMC505 (likelihood and prior tempering from the moment-matched unimodal prior) with NUTS on the two-slope
# transformed target and the :distributed backend: maps fitted to the population before the first round of every SMC
# iteration (`compute_TQB_info`), identity mass matrix, step size updated every round (`nuts_transformed_move`).
# Usage: julia --project=.. SMC145.jl <seed> [1k|2k|4k|8k]

# Likelihood and prior tempering. Only the prior of the spike-and-slab parameters η given p is tempered, from a
# unimodal prior to the spike-and-slab mixture. The unimodal prior is the Gaussian that matches the mean and
# variance of the mixture given p (so it depends on p). The priors on p and the noise parameters
# (`add_logprior`) are fixed along the path.
run_str = "SMC145"

# Mean and variance of p * slab + (1-p) * spike (law of total variance). At p = 1/2 with the current
# hyperparameters, these are -8 and 68.
unimodal_mean(p) = p*μ_slab + (1-p)*μ_spike
unimodal_var(p) = p*σ_slab^2 + (1-p)*σ_spike^2 + p*(1-p)*(μ_slab-μ_spike)^2
unimodal_prior(p) = Normal(unimodal_mean(p), sqrt(unimodal_var(p)))

ss_unimodal_logprior(θ) = begin
    p = logistic(θ[p_idx])
    m, s = unimodal_mean(p), sqrt(unimodal_var(p))
    sum(normlogpdf(m, s, θ[d]) for d in ss_idxs)
end

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

move_func = nuts_transformed_move;  init_stepsize = 0.5;
n_nuts_func = repeat_5;
ldp_builder = make_ldp;
prior_path(θ, γ) = (1-γ) * ss_unimodal_logprior(θ) + γ * ss_bimodal_logprior(θ) + add_logprior(θ)

# Command-line arguments: the seed, then optionally the population size (1k, 2k, 4k or 8k; default 8k).
seed = parse(Int64, ARGS[1])
# seed = 1

psize_str = length(ARGS) >= 2 ? ARGS[2] : "8k"
pop_size = Dict("1k" => 1000, "2k" => 2000, "4k" => 4000, "8k" => 8000)[psize_str]; target_ess = 0.8pop_size; init_pop_size = pop_size;

fname = joinpath(mkpath(joinpath(@__DIR__, "output/seed$seed")), "$(run_str)_$(psize_str).jld2");
display(fname)
# run_SMC overwrites `fname` after every iteration; never start over an existing run (use resume_SMC for that)
myid() == 1 && isfile(fname) && error("$(fname) exists; move it away or resume it instead")

rng = StableRNG(hash((run_str, pop_size, seed)))
run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_stepsize=init_stepsize, init_pop_size=init_pop_size, max_npass=10, compute_pop_info=compute_TQB_info,
    verbose=1, rng=rng, parallel=:distributed, pbar_lines=20,  # errors if SMC_NWORKERS is unset
);
