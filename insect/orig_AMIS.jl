include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../AMIS_helpers.jl"));
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
using Pathfinder, PSIS

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];

OUTDIR = joinpath(@__DIR__, "output", "data$(dir_idx)") # output directory
@nowarn_load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;


function orig_AMIS(rng, target, MAP, hess; Kmax=50, df=4, n_out=10000, io=stdout)
    d = LogDensityProblems.dimension(target)
    Σ = inv(PDMat(hermitianpart!(hess)))
    q_init = MvTDist(df, MAP, Σ)

    n_vec = [0; round.(Int, logrange(1e4, 1e6, 16))]
    incr_vec = diff(n_vec)
    n_iter = length(incr_vec) - 1
    gm_init = GaussianMixture(
        1, d, [1.],
        [copy(MAP)],
        [cholesky(hess)], 
    );
    gm_vec = [gm_init]; # I
    all_samples = Matrix{Float64}(undef, d, 0); # D x N
    all_logps = Float64[]; # N
    all_logqs_mat = Matrix{Float64}(undef, 1, 0); # I x N
    all_logws = Float64[]; # N
    for iter in 1:n_iter
        n_tot = sum(incr_vec[1:iter])
        n_incr = incr_vec[iter]
        # n_next = incr_vec[iter + 1]
        prop_ws = incr_vec[1:iter] ./ n_tot

        # Draw and evaluate new samples
        gm = gm_vec[end]
        if iter == 1
            new_samples = rand(rng, q_init, n_incr)
        else
            new_samples = rand(rng, gm, n_incr)
        end
        all_samples = hcat(all_samples, new_samples)

        t = @elapsed new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples))
        println(io, "  log target of $n_incr samples: $(round(t; digits=2)) s")
        new_logps[findall(isnan, new_logps)] .= -Inf
        append!(all_logps, new_logps)

        new_logqs_mat = stack([
            i == 1 ? logpdf(q_init, new_samples) : logpdf(gm_i, new_samples) 
        for (i, gm_i) in enumerate(gm_vec)]) # N_incr x I
        all_logqs_mat = hcat(all_logqs_mat, new_logqs_mat')

        all_logqs = vec(logsumexp(all_logqs_mat .+ log.(prop_ws); dims=1))
        all_logws = all_logps .- all_logqs 

        @assert size(all_samples) == (d, n_tot)
        @assert size(all_logps) == (n_tot,)
        @assert size(all_logqs_mat) == (iter, n_tot)
        @assert size(all_logws) == (n_tot,)

        psis_res = psis(all_logws; normalize=false, warn=false)
        psis_logws = psis_res.log_weights

        Zhat = round(logsumexp(psis_res.log_weights)-log(n_tot); digits=4)
        wESS = round(compute_ess(all_logws); digits=4)

        n_em = min(n_tot, 20000)
        em_idxs = sortperm(psis_logws, rev=true)[1:n_em]        
        em_ws = exp.(psis_logws[em_idxs] .- maximum(psis_logws))
        em_ws .*= n_em / sum(em_ws)

        gm = deepcopy(gm)
        X = all_samples[:,em_idxs];
        
        # Re-init mixture
        K_add = Kmax - gm.K
        sample_idxs = sample(rng, 1:n_em, weights(em_ws), K_add; replace=false)
        overall_var = var(X; dims=2) |> vec
        new_prec_chol = cholesky(diagm(1 ./ overall_var))
        gm = K_add == 0 ? deepcopy(gm) : GaussianMixture(
            Kmax, gm.d, [gm.weights .* (gm.K/Kmax); fill(1/Kmax, K_add)], 
            [gm.means; [copy(X[:, idx]) for idx in sample_idxs]], 
            [gm.chols; [deepcopy(new_prec_chol) for _ in 1:K_add]]
        )
        @assert sum(gm.weights) ≈ 1.

        # Fit Gaussian mixture using subset of accumulated samples
        t = @elapsed log_liks = fit_gm!(gm, X; xweights=em_ws, max_iter=100)
        println(io, "  EM fit: $(round(t; digits=2)) s")

        push!(gm_vec, trim_gm(gm, 1e-4))
        all_logqs_mat = vcat(all_logqs_mat, logpdf(gm_vec[end], all_samples)')

        @info "Iter $iter:" n_tot Zhat wESS psis_res.pareto_shape
        flush(io)
    end

    n_tot = sum(incr_vec);
    n_incr = incr_vec[end]
    prop_ws = incr_vec ./ n_tot;
    gm = gm_vec[end];
    new_samples = rand(rng, gm, n_incr);
    all_samples = hcat(all_samples, new_samples);

    new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples));
    new_logps[findall(isnan, new_logps)] .= -Inf
    append!(all_logps, new_logps);

    new_logqs_mat = stack([
        i == 1 ? logpdf(q_init, new_samples) : logpdf(gm_i, new_samples) 
    for (i, gm_i) in enumerate(gm_vec)]) # N_incr x I
    all_logqs_mat = hcat(all_logqs_mat, new_logqs_mat');

    all_logqs = vec(logsumexp(all_logqs_mat .+ log.(prop_ws); dims=1));
    all_logws = all_logps .- all_logqs;

    psis_res = psis(all_logws; normalize=false, warn=false);
    psis_logws = psis_res.log_weights

    return (
        incr_vec = incr_vec,
        gm_vec = gm_vec,
        unweighted_samples = [all_samples[:,idx] for idx in stratified_sampling(exp.(psis_logws .- maximum(psis_logws)), n_out; rng=rng)],
        psis_logws = psis_logws,
        pareto_shape = psis_res.pareto_shape
    )
end

LinearAlgebra.BLAS.set_num_threads(1)

# model_idx = 63
# begin
# Progress goes to the main log, details of each model to logs/data[d]/orig_AMIS/model[m].log.
LOGDIR = mkpath(joinpath(@__DIR__, "logs", "data$(dir_idx)", "orig_AMIS"))
counter = Threads.Atomic{Int}(0)
amis_summary(res) = "logZ $(round(logsumexp(res.value.psis_logws) - log(length(res.value.psis_logws)); digits=4)), khat $(round(res.value.pareto_shape; digits=3))"
for model_idx in 1:n_models
    fname = joinpath(OUTDIR, "orig_AMIS_model$(model_idx).jld2")
    # isfile(fname) && continue

    with_model_log(model_idx, counter, n_models; logdir=LOGDIR, summary=amis_summary) do io
        d = nparams[model_idx]
        target = make_insect_ldp(models[model_idx], data);
        MAP = collect(model_fits[model_idx].xmin)
        hess = MAP_hessians[model_idx]

        rng = StableRNG(hash((genmodel_idx, model_idx, "orig_AMIS")))
        timed_res = @timed orig_AMIS(
            rng, target, MAP, hess; io=io
        ); 
        @save fname timed_res
        timed_res
    end
end
log_failures()

# exit()

## Playgroud

# dir_idx = 2
# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
# model_idx = 64
# fname = joinpath(OUTDIR, "orig_AMIS_model$(model_idx).jld2")

# @load fname timed_res;
# timed_res.time
# res = timed_res.value;

# compute_ess(res.psis_logws)
# logsumexp(res.psis_logws) .- log(length(res.psis_logws))

# keep_dists = reduce(
#     vcat, [
#         [
#             MvNormal(gm.means[k], Matrix(inv(gm.chols[k])))
#             for k in 1:gm.K if gm.weights[k] > 0.02
#         ] for gm in res.gm_vec    
#     ]
# );
# length(keep_dists)

# Random.seed!(dir_idx*n_models + model_idx);
# @time keep_dists, viable_dists = init_dists(target, prior_sampler, prior_means, prior_vars, 50, 100, 2d; progress=true);
# length(viable_dists)
# length(keep_dists)
# round.(stack(getproperty.(keep_dists, :μ))', digits=2)

# zs = stack([(dist.μ .- prior_means) ./ sqrt.(prior_vars) for dist in viable_dists]);
# summarystats(zs)

# include(joinpath(@__DIR__, "../plot_helpers.jl"));
# using Turing, MCMCChains

# model_idx = 64;
# d = nparams[model_idx]
# dir_idx = 2
# genmodel_idx = feasible_idxs[dir_idx]

# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
# mcmc_fname = joinpath(OUTDIR, "MCMC_model$(model_idx).jld2");

# @nowarn_load mcmc_fname chn ess_df;
# trace = chn.value[:,1:d,1].data;
# # ess_df
# # f = plot_pairs(eachrow(trace), title="Dataset $dir_idx, model $model_idx", scatter_kwargs=(markersize=5, alpha=0.05))

# begin
#     i1 = 4
#     # i2 = 3
#     i2 = 7
#     f, ax, sc = scatter(trace[:,i1], trace[:,i2], color=:grey, alpha=0.05)
#     autolimits!(ax)
#     ax_limits = ax.finallimits[]
#     scatter!(
#         getindex.(getproperty.(keep_dists[1:50], :μ), i1), 
#         getindex.(getproperty.(keep_dists[1:50], :μ), i2), 
#         color=1:length(keep_dists[1:50]), colormap=Reverse(:viridis), alpha=0.8, markersize=8,
#     )
#     for (i, dist) in enumerate(keep_dists[1:50])
#         add_ellipse!(
#             ax, dist.μ, dist.Σ, i1, i2, 
#             color=i, colormap=Reverse(:viridis), colorrange=(1, length(keep_dists)), alpha=0.4
#         )
#     end
#     # scatter!(
#     #     getindex.(getproperty.(keep_dists, :μ), i1), 
#     #     getindex.(getproperty.(keep_dists, :μ), i2), 
#     #     alpha=0.4, markersize=6,
#     # )
#     limits!(ax, ax_limits)
#     display(current_figure())
# end

# hist(reduce(vcat, keep_dists .|> Base.Fix2(getproperty, :Σ) .|> diag))

# model_idx = 8
# for dir_idx in 1:3
#     genmodel_idx = feasible_idxs[dir_idx]
#     OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
#     mcmc_fname = joinpath(OUTDIR, "MCMC_model$(model_idx).jld2");
#     @nowarn_load mcmc_fname chn ess_df;
#     trace = chn.value[:,1:d,1].data;
#     # ess_df
#     f = plot_pairs(eachrow(trace), title="Dataset $dir_idx, model $model_idx", scatter_kwargs=(markersize=5, alpha=0.05))
#     display(f)
# end