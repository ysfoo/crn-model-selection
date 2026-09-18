# Run `setup.jl` and `../gaussian_mixtures.jl`.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter
using AdvancedHMC, Bijectors, BridgeSampling, LogDensityProblems, LogDensityProblemsAD, MCMCChains, PSIS, Turing

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

# `bridge_idxs` are used in bridge sampling identity as samples drawn from posterior
# `fit_idxs` are used to fit proposal distribution
function bridge_sampling(target, d, chn, bridge_idxs, fit_idxs)
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3])
    d, _, n_chains = size(trace)
    samples = reshape(trace, d, :)    
    logp_samples = extract_logp(chn);
    logp_func(x) = begin
        res = target.logtarget(x)
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
            Random.seed!(seed)
            gm = initialize_gm(smp2, K; method=:random)
            log_liks = fit_gm!(gm, smp2; max_iter=100)
            (gm=gm, em_objs=log_liks)
        end for seed in 1:10
    ];
    best_fit = argmax((x)->last(x.em_objs), gm_fits);
    gm = best_fit.gm

    # Propose from mixture, then evaluate all densities
    mix_smp = rand(gm, n_prop);
    logp_smp1 = logp_samples[bridge_idxs];
    logmix_smp1 = logpdf(gm, smp1);
    logp_mix = logp_func.(eachcol(mix_smp));
    logmix_mix = logpdf(gm, mix_smp);

    logml_mix, i_mix = BridgeSampling.iterative_algorithm(logp_smp1 .- logmix_smp1, logp_mix .- logmix_mix, n_post, n_prop; tol=1e-8, maxiter=1000, use_ess=true, n_chains)
    LML_mix = BridgeSampling.LogMarginalLikelihood(logml_mix, i_mix, logp_smp1, logmix_smp1, logp_mix, logmix_mix)

    return LML_mix
end

model_idxs = 1:n_models
# if dir_idx == 3
#     model_idxs = [10]
# end
# if dir_idx == 5
#     model_idxs = [25]
# end

for model_idx in model_idxs
    fname = joinpath(OUTDIR, "BSnew_model$(model_idx).jld2")
    # isfile(fname) && return false
    println("Model $(model_idx)")
    flush(stdout); flush(stderr);

    d = nparams[model_idx]
    pmodel = create_petab_model(models[model_idx], data, u0)
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false))
    target = PEtabLogDensity(petab_prob)

    mcmc_fname = joinpath(OUTDIR, "chains7000_model$(model_idx).jld2");
    @nowarn_load mcmc_fname chn ess_df;
    min_ess = round(Int, minimum(ess_df.nt.ess))
    
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

    Random.seed!(dir_idx*n_models + model_idx)
    timed_res = @timed bridge_sampling(target, d, chn, bridge_idxs, fit_idxs)
    cv = error_estimate(timed_res.value).cv
    @info "Model $model_idx" round(timed_res.time, digits=2) round(cv, digits=6) min_ess
    @save fname timed_res
end