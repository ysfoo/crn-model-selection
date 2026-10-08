include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../AMIS_helpers.jl"));
include(joinpath(@__DIR__, "ldp_setup.jl"));
include(joinpath(@__DIR__, "log_helpers.jl"));

# This script takes one command-line argument, which is the index of `feasible_idxs`.
dir_idx = parse(Int64, ARGS[1])
# dir_idx = 2
genmodel_idx = feasible_idxs[dir_idx]

OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)") # inference result directory
mkpath(OUTDIR)

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, Random
using JLD2, ProgressMeter
using LogDensityProblems
using Pathfinder, PSIS, StableRNGs

@load joinpath(@__DIR__, "data.jld2") all_data;
data = all_data[genmodel_idx];


# `grad_target` (tolerances suitable for gradients) is used by Pathfinder, `target` by the importance sampling.
function robust_AMIS(rng, grad_target, target, prior_sampler, prior_means, prior_vars; 
                    nruns=50, Kmax=50, n_out=10000, io=stdout)

    d = LogDensityProblems.dimension(target)
    # The initial proposal uses all viable Pathfinder fits (can be thousands of components); the EM initialisation uses
    # at most Kmax spread-out ones. Using only the latter as q1 underestimated logZ for d = 9, 10 (twice the error).
    t = @elapsed q1_dists, em_init_dists = init_dists(grad_target, prior_sampler, prior_means, prior_vars, nruns, Kmax, 2d; rng=rng, io=io)
    println(io, "Pathfinder initialisation ($nruns runs): $(round(t; digits=2)) s, $(length(q1_dists)) components")
    K1 = length(q1_dists)

    n_vec = [0; round.(Int, logrange(1e4, 1e6, 16))]
    incr_vec = diff(n_vec)
    n_iter = length(incr_vec) - 1
    gm = GaussianMixture(
        K1, d, fill(1/K1, K1),
        [copy(dist.μ) for dist in q1_dists],
        [cholesky(inv(hermitianpart(dist.Σ))) for dist in q1_dists], 
    );
    gm_vec = [gm]; # I
    all_samples = Matrix{Float64}(undef, d, 0); # D x N
    all_logps = Float64[]; # N
    all_logqs_mat = Matrix{Float64}(undef, 1, 0); # I x N
    all_logws = Float64[]; # N
    for iter in 1:n_iter
        n_tot = sum(incr_vec[1:iter])
        n_incr = incr_vec[iter]
        n_next = incr_vec[iter + 1]
        prop_ws = incr_vec[1:iter] ./ n_tot

        # Draw and evaluate new samples
        gm = gm_vec[end]
        new_samples = rand(rng, gm, n_incr)
        all_samples = hcat(all_samples, new_samples)

        t = @elapsed new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples))
        println(io, "  log target of $n_incr samples: $(round(t; digits=2)) s")
        new_logps[findall(isnan, new_logps)] .= -Inf
        append!(all_logps, new_logps)

        new_logqs_mat = stack([logpdf(gm_i, new_samples) for gm_i in gm_vec]) # N_incr x I
        all_logqs_mat = hcat(all_logqs_mat, new_logqs_mat')

        all_logqs = vec(logsumexp(all_logqs_mat .+ log.(prop_ws); dims=1))
        all_logws = all_logps .- all_logqs 

        @assert size(all_samples) == (d, n_tot)
        @assert size(all_logps) == (n_tot,)
        @assert size(all_logqs_mat) == (iter, n_tot)
        @assert size(all_logws) == (n_tot,)

        psis_res = psis(all_logws; normalize=false, warn=false)
        psis_logws = psis_res.log_weights
        pareto_shape = psis_res.pareto_shape

        Zhat = round(logsumexp(psis_res.log_weights)-log(n_tot); digits=4)
        wESS = round(compute_ess(all_logws); digits=4)

        psis_res = psis(all_logws; normalize=false, warn=false) # TODO: divide by -q log q?
        # lognegqlogq = map((logq) -> logq < -1 ? logq + log(-logq) : -1 , all_logqs)
        # psis_res = psis(all_logws .- lognegqlogq; normalize=false, warn=false)
        psis_logws = psis_res.log_weights
        n_em = min(n_tot, 20000)
        em_idxs = sortperm(psis_logws, rev=true)[1:n_em]        
        em_ws = exp.(psis_logws[em_idxs] .- maximum(psis_logws))
        em_ws .*= n_em / sum(em_ws)

        gm = deepcopy(gm)
        X = all_samples[:,em_idxs];

        # Re-init mixture
        K_add = max(Kmax - gm.K, 0)
        sample_idxs = sample(rng, 1:n_em, weights(em_ws), K_add; replace=false)
        overall_var = var(X; dims=2) |> vec
        new_prec_chol = cholesky(diagm(1 ./ overall_var))
        if iter == 1
            gm = GaussianMixture(
                Kmax, d, fill(1/Kmax, Kmax),
                [[copy(dist.μ) for dist in em_init_dists]; [copy(X[:, idx]) for idx in sample_idxs]],
                [[cholesky(inv(hermitianpart(dist.Σ))) for dist in em_init_dists]; [deepcopy(new_prec_chol) for _ in 1:K_add]],
            );
        else
            gm = K_add == 0 ? gm : GaussianMixture(
                Kmax, gm.d, [gm.weights .* (gm.K/Kmax); fill(1/Kmax, K_add)], 
                [gm.means; [copy(X[:, idx]) for idx in sample_idxs]], 
                [gm.chols; [deepcopy(new_prec_chol) for _ in 1:K_add]]
            )
        end
        @assert sum(gm.weights) ≈ 1.

        # Fit Gaussian mixture using subset of accumulated samples
        t = @elapsed log_liks = fit_gm!(gm, X; xweights=em_ws, max_iter=100)
        println(io, "  EM fit: $(round(t; digits=2)) s")
        
        log_coefs = 2 .* all_logps .- all_logqs;
        n_cs = min(n_tot, 100000)
        cs_idxs = sortperm(all_logws, rev=true)[1:n_cs] # TODO: choose n_cs by ranking which quantity?

        log_comp_probs = zeros(gm.K, n_cs);
        diff_tmp = zeros(d);
        for k in 1:gm.K
            log_mvn_pdf!(view(log_comp_probs, k, :), all_samples[:,cs_idxs], gm.means[k], gm.chols[k], diff_tmp)
        end

        log_dot_vec = vec(logsumexp(log_comp_probs .+ log.(gm.weights); dims=1));
        intercepts = all_logqs[cs_idxs] .+ log(n_tot/(n_vec[end] - n_tot)) # TODO: should denom be n_next or n_vec[end] - n_tot?
        offset_terms = log_coefs[cs_idxs] .- logaddexp.(log_dot_vec, intercepts);
        offset = logsumexp(offset_terms)

        cs_args = (
            N=n_cs, K=gm.K, log_coefs=log_coefs[cs_idxs], log_probs=log_comp_probs, intercepts=intercepts,
            offset=offset, cache=zeros(gm.K)
        );
        t = @elapsed opt_w, history, converged = cauchy_simplex(cs_obj_func, cs_grad_func!, gm.weights, cs_args; max_iter=100);
        println(io, "  mixture weights (Cauchy simplex): $(round(t; digits=2)) s")
        
        gm.weights .= opt_w

        push!(gm_vec, trim_gm(gm, 1e-4))
        all_logqs_mat = vcat(all_logqs_mat, logpdf(gm_vec[end], all_samples)')

        @info "Iter $iter:" n_tot Zhat wESS pareto_shape
        flush(io)
        if false
            i1 = 4
            # i2 = 3
            i2 = 7
            f, ax, sc = scatter(trace[:,i1], trace[:,i2], color=:grey, alpha=0.05, axis=(title="Iter $iter",))
            autolimits!(ax)
            ax_limits = ax.finallimits[]
            scatter!(
                getindex.(gm.means, i1), 
                getindex.(gm.means, i2), 
                color=1:gm.K, colormap=Reverse(:viridis), alpha=0.8, markersize=8,
            )
            for i in 1:gm.K
                add_ellipse!(
                    ax, gm.means[i], Matrix(inv(gm.chols[i])), i1, i2, 
                    color=gm.weights[i], colormap=Reverse(:viridis), colorrange=(0, 1), alpha=0.6
                )
            end
            # scatter!(
            #     getindex.(getproperty.(keep_dists, :μ), i1), 
            #     getindex.(getproperty.(keep_dists, :μ), i2), 
            #     alpha=0.4, markersize=6,
            # )
            limits!(ax, ax_limits)
            display(f)
        end
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

    new_logqs_mat = stack([logpdf(gm_i, new_samples) for gm_i in gm_vec]); # N_incr x I
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
# Progress goes to the main log, details of each model to logs/data[d]/robust_AMIS/model[m].log.
LOGDIR = mkpath(joinpath(@__DIR__, "logs", "data$(dir_idx)", "robust_AMIS"))
counter = Threads.Atomic{Int}(0)
amis_summary(res) = "logZ $(round(logsumexp(res.value.psis_logws) - log(length(res.value.psis_logws)); digits=4)), khat $(round(res.value.pareto_shape; digits=3))"
for model_idx in 1:n_models
    fname = joinpath(OUTDIR, "robust_AMIS_model$(model_idx).jld2")
    # isfile(fname) && continue # TODO: uncomment

    with_model_log(model_idx, counter, n_models; logdir=LOGDIR, summary=amis_summary) do io
        d = nparams[model_idx]
        grad_target = make_insect_ldp(models[model_idx], data; tol=1e-6);
        target = make_insect_ldp(models[model_idx], data);
        prior_sampler = create_prior_sampler(prior_dists(d));

        prior_means = [fill(0.0, d-1); -1.0];
        prior_vars = [fill(2.0^2, d-1); 1.0^2];

        rng = StableRNG(hash((genmodel_idx, model_idx, "robust_AMIS")))
        timed_res = @timed robust_AMIS(
        rng, grad_target, target, prior_sampler, 
        prior_means, prior_vars; nruns=20, io=io
        ); 
        @save fname timed_res
        timed_res
    end
end
log_failures()

# exit()

## Playground

# fname = joinpath(OUTDIR, "robust_AMIS_model$(model_idx).jld2");
# @load fname timed_res;
# res = timed_res.value;
# N = 10^6;
# logsumexp(res.psis_logws) - log(N)
# compute_ess(res.psis_logws)

# include(joinpath(@__DIR__, "../plot_helpers.jl"));
# using Turing, MCMCChains

# model_idx = 63;
# d = nparams[model_idx]
# dir_idx = 2
# genmodel_idx = feasible_idxs[dir_idx]

# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
# mcmc_fname = joinpath(OUTDIR, "MCMC_model$(model_idx).jld2");

# @nowarn_load mcmc_fname chn ess_df;
# trace = chn.value[:,1:d,1].data;