include(joinpath(@__DIR__, "setup.jl"));

using Turing, MCMCChains, AdvancedHMC
using StableRNGs

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])

mkpath(INFDIR)
INFDIR = joinpath(@__DIR__, "output/seed$(seed)");

@model function turing_model(target)
    θ ~ filldist(Turing.Flat(), LogDensityProblems.dimension(target))
    Turing.@addlogprob! LogDensityProblems.logdensity(target, θ)
    return nothing
end

function run_mcmc(model_sym, nadapts, n_sample, n_chains)
    mcmc_fname = joinpath(INFDIR, "chains_$(model_sym).jld2")
    target = target_dict[model_sym]

    rng = StableRNG(hash((seed, model_sym, "MCMC_chains")))
    chn = sample(
        rng, turing_model(target), Turing.NUTS(0.9, metricT=AdvancedHMC.UnitEuclideanMetric), 
        MCMCThreads(), n_sample, n_chains; nadapts, progress=false
    );
    acc_rates = collect(vec(mean(chn[:acceptance_rate]; dims=1)))
    step_sizes = collect(vec(chn[:step_size][end,:]))
    ess_df = ess(chn)
    duration = round(MCMCChains.compute_duration(chn)/60; digits=2)

    @info String(model_sym) duration
    flush(stderr)
    display(acc_rates)
    display(step_sizes)
    display(ess_df)    
    flush(stdout)    

    @suppress_err @save mcmc_fname chn ess_df;
end

for sym in model_syms
    run_mcmc(sym, 1000, 20000, 5)
end