include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../stats_helpers.jl"));
include(joinpath(@__DIR__, "ldp_setup.jl"));
include(joinpath(@__DIR__, "log_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, Random
using JLD2, ProgressMeter, StableRNGs
using LogDensityProblems
using PSIS

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

OUTDIR = joinpath(@__DIR__, "output", "data$(dir_idx)") # output directory
@nowarn_load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;

function laplace_IS(rng, target, MAP, hess, n_samples; df=4, n_out=10000)
    Σ = inv(PDMat(hermitianpart!(hess)))
    proposal = MvTDist(df, MAP, Σ)
    samples = rand(rng, proposal, n_samples)

    logps = LogDensityProblems.logdensity.(Ref(target), eachcol(samples))
    logps[findall(isnan, logps)] .= -Inf
    logqs = logpdf.(Ref(proposal), eachcol(samples))
    logws = logps .- logqs

    psis_res = psis(logws; normalize=false, warn=false);
    psis_logws = psis_res.log_weights

    return (
        unweighted_samples = [samples[:,idx] for idx in stratified_sampling(exp.(psis_logws .- maximum(psis_logws)), n_out; rng=rng)],
        psis_logws = psis_logws,
        pareto_shape = psis_res.pareto_shape
    )
end

LinearAlgebra.BLAS.set_num_threads(1)

# model_idx = 63
# begin
# Progress (one line per model start and end) goes to the main log.
counter = Threads.Atomic{Int}(0)
IS_summary(res) = "logZ $(round(logsumexp(res.value.psis_logws) - log(length(res.value.psis_logws)); digits=4)), khat $(round(res.value.pareto_shape; digits=3))"
for model_idx in 1:n_models
    fname = joinpath(OUTDIR, "laplace_IS_model$(model_idx).jld2")
    # isfile(fname) && continue

    with_model_log(model_idx, counter, n_models; summary=IS_summary) do io
        d = nparams[model_idx];
        target = make_insect_ldp(models[model_idx], data);
        MAP = collect(model_fits[model_idx].xmin)
        hess = MAP_hessians[model_idx]
        rng = StableRNG(hash((genmodel_idx, model_idx, "laplace_IS")))
        timed_res = @timed laplace_IS(rng, target, MAP, hess, 10^6)
        @save fname timed_res
        timed_res
    end
end
log_failures()

# exit()

# Playground

# model_idx = 20;
# fname = joinpath(OUTDIR, "laplace_IS_model$(model_idx).jld2");
# @load fname timed_res;
# timed_res.time
# N = length(timed_res.value.psis_logws);
# logsumexp(timed_res.value.psis_logws) - log(N)
# compute_ess(timed_res.value.psis_logws)
# timed_res.value.pareto_shape

# for dir_idx in 1:n_feasible
#     OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
#     n_isnan = 0
#     for model_idx in 1:n_models
#         fname = joinpath(OUTDIR, "laplace_IS_model$(model_idx).jld2")
#         @load fname timed_res
#         n_isnan += sum(isnan, timed_res.value.psis_logws)
#     end
#     display((dir_idx, n_isnan))
# end