# Run setup.jl.
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "ldp_setup.jl"));
include(joinpath(@__DIR__, "log_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = mkpath(joinpath(@__DIR__, "output", "data$(dir_idx)")) # output directory

# Load stored MAP if true, otherwise compute afresh
LOAD_MAP = false;

# Fetch packages.
using OrdinaryDiffEq, Optim
using JLD2, Random, StableRNGs, Suppressor

LinearAlgebra.BLAS.set_num_threads(1)

@load joinpath(@__DIR__, "data.jld2") all_data;

# Fits all models to one dataset.
data = all_data[genmodel_idx];
model_order = 1:n_models

if LOAD_MAP
    @load "$OUTDIR/MAP.jld2" model_fits
else
    # Progress (one line per model start and end) goes to the main log.
    fit_times = zeros(n_models)
    model_fits = Vector{Any}(undef, n_models)
    counter = Threads.Atomic{Int}(0)
    fit_summary(fit) = "fmin $(round(fit.fmin; digits=3)), $(count(<(fit.fmin + 1e-3), fit.fmins))/$(length(fit.fmins)) starts within 1e-3 of fmin"
    for i in model_order
        with_model_log(i, counter, n_models; summary=fit_summary) do io
            ldp = make_insect_ldp(models[i], data; tol=1e-6)
            fit_times[i] = @elapsed model_fits[i] = fit_MAP(ldp, 10; rng=StableRNG(hash((genmodel_idx, i, "fit_MAP"))))
            model_fits[i]
        end
    end
    log_failures()
    model_fits = [model_fit for model_fit in model_fits]
    nllhs = [-model_fit.loglik for model_fit in model_fits];
    @suppress_err @save "$OUTDIR/MAP.jld2" model_fits fit_times nllhs
end

hess_times = zeros(n_models)
MAP_hessians = Vector{Matrix{Float64}}(undef, n_models)
for i in model_order
    ldp = make_insect_ldp(models[i], data; tol=1e-6)
    hess_times[i] = @elapsed MAP_hessians[i] = neglogpost_hessian(ldp, model_fits[i].xmin)
end
logmsg("Hessians done, $(round(sum(hess_times); digits=1)) s in total")

@save "$OUTDIR/MAP_hess.jld2" MAP_hessians hess_times
