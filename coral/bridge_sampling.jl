# Run `setup.jl` and `../gaussian_mixtures.jl`.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])

INFDIR = joinpath(@__DIR__, "output/seed$(seed)");

# Fetch packages.
using BridgeSampling, MCMCChains, StableRNGs, Turing

# `bridge_idxs` are used in bridge sampling identity as samples drawn from posterior
# `fit_idxs` are used to fit proposal distribution
# `rng` is used for the mixture initialisations and the proposal draws
function bridge_sampling(rng, target, d, chn, bridge_idxs, fit_idxs)
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3])
    d, _, n_chains = size(trace)
    samples = reshape(trace, d, :)    
    logp_samples = extract_logp(chn);
    logp_func(x) = begin
        res = LogDensityProblems.logdensity(target, x)
        isnan(res) ? -Inf : res
    end

    smp1 = samples[:,bridge_idxs]; # for bridge weights
    smp2 = samples[:,fit_idxs]; # for fitting proposal
    n_post = size(smp1, 2)
    n_prop = 10^6

    # Fit mixture        
    K = 50;
    gm_fits = [
        begin
            gm = initialize_gm(smp2, K; method=:random, rng=rng)
            log_liks = fit_gm!(gm, smp2; max_iter=100)
            (gm=gm, em_objs=log_liks)
        end for _ in 1:10
    ];
    best_fit = argmax((x)->last(x.em_objs), gm_fits);
    gm = best_fit.gm

    # Propose from mixture, then evaluate all densities
    mix_smp = rand(rng, gm, n_prop);
    logp_smp1 = logp_samples[bridge_idxs];
    logmix_smp1 = logpdf(gm, smp1);
    logp_mix = logp_func.(eachcol(mix_smp));
    logmix_mix = logpdf(gm, mix_smp);

    logml_mix, i_mix = BridgeSampling.iterative_algorithm(logp_smp1 .- logmix_smp1, logp_mix .- logmix_mix, n_post, n_prop; tol=1e-8, maxiter=1000, use_ess=true, n_chains)
    LML_mix = BridgeSampling.LogMarginalLikelihood(logml_mix, i_mix, logp_smp1, logmix_smp1, logp_mix, logmix_mix)

    return LML_mix
end

# 5 chains, 100k samples total
# Use 90k as bridge_idxs, 10k as fit_idxs
bridge_idxs = filter(x -> mod(x, 10) != 0, 1:10^5)
fit_idxs = filter(x -> mod(x, 10) == 0, 1:10^5)

for model_sym in model_syms
    fname = joinpath(INFDIR, "BS_$(model_sym).jld2")
    target = target_dict[model_sym]
    d = nparam_dict[model_sym]

    mcmc_fname = joinpath(INFDIR, "chains_$(model_sym).jld2");
    @load mcmc_fname chn;

    rng = StableRNG(hash((seed, model_sym, "bridge_sampling")))
    timed_res = @timed bridge_sampling(rng, target, d, chn, bridge_idxs, fit_idxs)
    @save fname timed_res

    @info String(model_sym) timed_res.time timed_res.value.value error_estimate(timed_res.value; n_chains=5).cv
    flush(stderr)
end