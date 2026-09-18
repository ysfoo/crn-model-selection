# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
# dir_idx = 2
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory
mkpath(OUTDIR)

# Fetch packages.
using PEtab, OrdinaryDiffEq
using JLD2, ProgressMeter, Random, PDMats, StableRNGs, Suppressor
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

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
    n_sample = 7000
    n_chains = 5

    # nadapts = 50
    # n_sample = 50
    # n_chains = 2

    mcmc_fname = joinpath(OUTDIR, "chains$(n_sample)_model$(model_idx).jld2")
    # isfile(mcmc_fname) && return false

    pmodel = create_petab_model(models[model_idx], data, u0)
    petab_prob = PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false))
    target = PEtabLogDensity(petab_prob);
    MAP = model_fits[model_idx].xmin
    hess = MAP_hessians[model_idx]
    Σ = inv(PDMat(hermitianpart!(hess)))

    seed = model_idx
    rng = StableRNG(seed + 2026)

    init_params = [
        begin
            p = copy(collect(MAP))
            while true
                tdist_sim = rand(rng, MvTDist(4, MAP, Σ))
                p = to_prior_scale(tdist_sim, target) |> target.inference_info.bijectors
                isfinite(target.logtarget(p)) && break
            end            
            InitFromParams((θ=p,))
        end for _ in 1:n_chains
    ]
    
    @model function turing_model(target)
        θ ~ filldist(Turing.Flat(), target.dim)
        Turing.@addlogprob! target.logtarget(θ)
        return nothing
    end

    # Eun MCMC chain...        
    chn = sample(
        rng, turing_model(target), Turing.NUTS(0.9, metricT=AdvancedHMC.UnitEuclideanMetric), MCMCThreads(), n_sample, n_chains; 
        initial_params=init_params, 
        nadapts=nadapts, save_state=false, progress=false
    );
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
mcmc_fname = joinpath(OUTDIR, "chains_model$(model_idx).jld2")
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