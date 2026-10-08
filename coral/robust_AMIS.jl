include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../AMIS_helpers.jl"));

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])

using PDMats, LogExpFunctions, PSIS, StableRNGs

@load joinpath(@__DIR__, "output/MAPs.jld2") model_fits;

mkpath(INFDIR)
INFDIR = joinpath(@__DIR__, "output/seed$(seed)");

function robust_AMIS(rng, target, prior_sampler, prior_means, prior_vars; 
                    nruns=20, Kmax=50)

    d = LogDensityProblems.dimension(target)
    @time q1_dists, em_init_dists = init_dists(target, prior_sampler, prior_means, prior_vars, nruns, Kmax, 2d; rng=rng)
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

        new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples))
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
                [copy(dist.μ) for dist in em_init_dists],
                [cholesky(inv(hermitianpart(dist.Σ))) for dist in em_init_dists], 
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
        @time log_liks = fit_gm!(gm, X; xweights=em_ws, max_iter=100)   
        
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
        @time opt_w, history, converged = cauchy_simplex(cs_obj_func, cs_grad_func!, gm.weights, cs_args; max_iter=100);
        
        gm.weights .= opt_w

        push!(gm_vec, trim_gm(gm, 1e-4))
        all_logqs_mat = vcat(all_logqs_mat, logpdf(gm_vec[end], all_samples)')

        @info "Iter $iter:" n_tot Zhat wESS pareto_shape
        flush(stderr)
        # if false
        #     i1 = 4
        #     # i2 = 3
        #     i2 = 7
        #     f, ax, sc = scatter(trace[:,i1], trace[:,i2], color=:grey, alpha=0.05, axis=(title="Iter $iter",))
        #     autolimits!(ax)
        #     ax_limits = ax.finallimits[]
        #     scatter!(
        #         getindex.(gm.means, i1), 
        #         getindex.(gm.means, i2), 
        #         color=1:gm.K, colormap=Reverse(:viridis), alpha=0.8, markersize=8,
        #     )
        #     for i in 1:gm.K
        #         add_ellipse!(
        #             ax, gm.means[i], Matrix(inv(gm.chols[i])), i1, i2, 
        #             color=gm.weights[i], colormap=Reverse(:viridis), colorrange=(0, 1), alpha=0.6
        #         )
        #     end
        #     # scatter!(
        #     #     getindex.(getproperty.(q1_dists, :μ), i1), 
        #     #     getindex.(getproperty.(q1_dists, :μ), i2), 
        #     #     alpha=0.4, markersize=6,
        #     # )
        #     limits!(ax, ax_limits)
        #     display(f)
        # end
    end

    n_tot = sum(incr_vec);
    n_incr = incr_vec[end]
    prop_ws = incr_vec ./ n_tot;
    gm = gm_vec[end];
    new_samples = rand(rng, gm, n_incr);
    all_samples = hcat(all_samples, new_samples);

    new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples))
    new_logps[findall(isnan, new_logps)] .= -Inf
    append!(all_logps, new_logps);

    new_logqs_mat = stack([logpdf(gm_i, new_samples) for gm_i in gm_vec]); # N_incr x I
    all_logqs_mat = hcat(all_logqs_mat, new_logqs_mat');

    all_logqs = vec(logsumexp(all_logqs_mat .+ log.(prop_ws); dims=1));
    all_logws = all_logps .- all_logqs;

    psis_res = psis(all_logws; normalize=false, warn=false);

    return (
        incr_vec = incr_vec,
        gm_vec = gm_vec,
        all_samples = all_samples,
        psis_logws = psis_res.log_weights,
        pareto_shape = psis_res.pareto_shape
    )
end

# for seed in 1:100
for model_sym in model_syms
    fname = joinpath(INFDIR, "robust_AMIS_$(model_sym).jld2")
    target = target_dict[model_sym]

    prior_dists = priors_dict[model_sym]
    prior_sampler = create_prior_sampler(prior_dists)
    prior_means = getproperty.(prior_dists, :μ)
    prior_vars = getproperty.(prior_dists, :σ) .|> abs2

    model_fit = model_fits[model_sym].value
    rng = StableRNG(hash((seed, model_sym, "robust_AMIS")))
    timed_res = @timed robust_AMIS(rng, target, prior_sampler, prior_means, prior_vars; nruns=30)
    @save fname timed_res

    psis_logws = timed_res.value.psis_logws
    Zhat = logsumexp(psis_logws) - log(length(psis_logws))
    @info String(model_sym) Zhat compute_ess(psis_logws) timed_res.value.pareto_shape
    flush(stderr)
end
# end