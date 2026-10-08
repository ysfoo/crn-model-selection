# Run `setup.jl` and `../gaussian_mixtures.jl`.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));
include(joinpath(@__DIR__, "ldp_setup.jl"));
include(joinpath(@__DIR__, "log_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, Random
using JLD2, ProgressMeter
using BridgeSampling, LogDensityProblems, MCMCChains, PSIS, StableRNGs

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

# `bridge_idxs` are used in bridge sampling identity as samples drawn from posterior
# `fit_idxs` are used to fit proposal distribution
# `rng` is used for the mixture initialisations and the proposal draws
function bridge_sampling(rng, target, d, chn, bridge_idxs, fit_idxs)
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3])
    d, _, n_chains = size(trace)
    samples = reshape(trace, d, :)    
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
    # Re-evaluate rather than use the log density stored in `chn`, so that both sets of samples use the same density.
    logp_smp1 = logp_func.(eachcol(smp1));
    logmix_smp1 = logpdf(gm, smp1);
    logp_mix = logp_func.(eachcol(mix_smp));
    logmix_mix = logpdf(gm, mix_smp);

    logml_mix, i_mix = BridgeSampling.iterative_algorithm(logp_smp1 .- logmix_smp1, logp_mix .- logmix_mix, n_post, n_prop; tol=1e-8, maxiter=1000, use_ess=true, n_chains)
    LML_mix = BridgeSampling.LogMarginalLikelihood(logml_mix, i_mix, logp_smp1, logmix_smp1, logp_mix, logmix_mix)

    return LML_mix
end

LinearAlgebra.BLAS.set_num_threads(1)

model_idxs = 1:n_models
# if dir_idx == 3
#     model_idxs = [10]
# end
# if dir_idx == 5
#     model_idxs = [25]
# end

# Progress (one line per model start and end) goes to the main log.
counter = Threads.Atomic{Int}(0)
BS_summary((timed_res, min_ess)) =
    "logZ $(round(timed_res.value.value; digits=4)), cv $(round(error_estimate(timed_res.value).cv; digits=6)), min ESS $min_ess"
for model_idx in model_idxs
    fname = joinpath(OUTDIR, "BS_8k_model$(model_idx).jld2")
    # isfile(fname) && continue

    with_model_log(model_idx, counter, n_models; summary=BS_summary) do io
    d = nparams[model_idx]
    target = make_insect_ldp(models[model_idx], data)

    mcmc_fname = joinpath(OUTDIR, "chains_8k_model$(model_idx).jld2");
    @nowarn_load mcmc_fname chn ess_df;
    min_ess = round(Int, minimum(ess_df.nt.ess))
    
    # 10000 samples (every 4th of 8000 x 5 chains) fit the mixture, the remaining 30000 enter the bridge identity.
    n_iter, _, n_chain = size(chn)
    n_tot = n_iter * n_chain
    g = gcd(n_iter, 10000 ÷ n_chain)
    b = n_iter ÷ g
    thres = (10000 ÷ n_chain) ÷ g  
    bridge_idxs = filter(x -> mod1(x, b) > thres, 1:n_tot)
    fit_idxs = filter(x -> mod1(x, b) <= thres, 1:n_tot)
    # display(length(bridge_idxs))
    # display(length(fit_idxs))
    # flush(stdout); flush(stderr)

    rng = StableRNG(hash((genmodel_idx, model_idx, "bridge_sampling")))
    timed_res = @timed bridge_sampling(rng, target, d, chn, bridge_idxs, fit_idxs)
    @save fname timed_res
    (timed_res, min_ess)
    end
end
log_failures()