# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "ldp_setup.jl"));
include(joinpath(@__DIR__, "nuts_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
# dir_idx = 2
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory
mkpath(OUTDIR)

# Fetch packages.
using OrdinaryDiffEq
using JLD2, ProgressMeter, Random, PDMats, StableRNGs, Suppressor
using AdvancedHMC, LinearAlgebra, LogDensityProblems, MCMCChains

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

# Load ground truth, data, MAP results.
@load joinpath(@__DIR__, "data.jld2") all_data;
@load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;

data = all_data[genmodel_idx];

function main(model_idx)  
    nadapts = 1000
    n_sample = 8000
    n_chains = 5

    # nadapts = 50
    # n_sample = 50
    # n_chains = 2

    mcmc_fname = joinpath(OUTDIR, "chains_8k_model$(model_idx).jld2")
    # isfile(mcmc_fname) && return false

    target = make_insect_ldp(models[model_idx], data; tol=1e-6)
    MAP = model_fits[model_idx].xmin
    hess = MAP_hessians[model_idx]
    Σ = inv(PDMat(hermitianpart!(hess)))

    seed = hash((genmodel_idx, model_idx, "MCMC_chains"))
    rng = StableRNG(seed)

    init_params = [
        begin
            p = copy(collect(MAP))
            while true
                p = rand(rng, MvTDist(4, MAP, Σ))
                isfinite(LogDensityProblems.logdensity(target, p)) && break
            end            
            p
        end for _ in 1:n_chains
    ]

    # Run MCMC chains...
    chn = run_nuts_chains(rng, target, init_params, n_sample, nadapts; δ=0.9);
    acc_rates = collect(vec(mean(chn[:acceptance_rate]; dims=1)))
    step_sizes = collect(vec(chn[:step_size][end,:]))
    ess_df = ess(chn)
    dur_mins = round(MCMCChains.compute_duration(chn)/60; digits=2)

    @info "Model $(model_idx)" dur_mins
    flush(stdout)
    flush(stderr)
    display(acc_rates)
    display(ess_df)
    flush(stdout)
    flush(stderr)   

    @suppress_err @save mcmc_fname chn ess_df;
end
return true

model_idxs = 1:n_models
# if dir_idx == 3
#     model_idxs = [10]
# end
# if dir_idx == 5
#     model_idxs = [25]
# end

for model_idx in model_idxs
    ran = main(model_idx);
end

exit()

# Tmp playground

model_idx = 50
D = length(parameters(models[model_idx]))
mcmc_fname = joinpath(OUTDIR, "chains_8k_model$(model_idx).jld2")
@load mcmc_fname chn ess_df;
ess_df

begin
    f = Figure(size=(800, 1000))
    for d in 1:D
        ax_i = cld(d, 2)
        ax_j = mod1(d, 2)
        ax = Axis(f[ax_i,ax_j])
        for c in 1:5
            lines!(chn.value[:,d,c], alpha=0.4)
        end
    end
    display(f)
end