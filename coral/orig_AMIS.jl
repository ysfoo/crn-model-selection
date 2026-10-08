include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])

using PDMats, LogExpFunctions, PSIS, StableRNGs

@load joinpath(@__DIR__, "output/MAPs.jld2") model_fits;

INFDIR = joinpath(@__DIR__, "output/seed$(seed)");
mkpath(INFDIR)

function orig_AMIS(rng, target, MAP, hess; Kmax=50, df=4)
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

        new_logps = LogDensityProblems.logdensity.(Ref(target), eachcol(new_samples))
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
        @time log_liks = fit_gm!(gm, X; xweights=em_ws, max_iter=100)        

        push!(gm_vec, trim_gm(gm, 1e-4))
        all_logqs_mat = vcat(all_logqs_mat, logpdf(gm_vec[end], all_samples)')

        @info "Iter $iter:" n_tot Zhat wESS psis_res.pareto_shape
        

        # begin
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

    new_logqs_mat = stack([
        i == 1 ? logpdf(q_init, new_samples) : logpdf(gm_i, new_samples) 
    for (i, gm_i) in enumerate(gm_vec)]) # N_incr x I
    all_logqs_mat = hcat(all_logqs_mat, new_logqs_mat');

    all_logqs = vec(logsumexp(all_logqs_mat .+ log.(prop_ws); dims=1));
    all_logws = all_logps .- all_logqs;

    psis_res = psis(all_logws; normalize=false, warn=false);

    return (
        incr_vec = incr_vec,
        gm_vec = gm_vec,
        # all_logps = all_logps,
        # all_logqs = all_logqs,
        all_samples = all_samples,
        psis_logws = psis_res.log_weights,
        pareto_shape = psis_res.pareto_shape
    )
end

for model_sym in model_syms
    fname = joinpath(INFDIR, "orig_AMIS_$(model_sym).jld2")
    target = target_dict[model_sym]

    model_fit = model_fits[model_sym].value
    rng = StableRNG(hash((seed, model_sym, "orig_AMIS")))
    timed_res = @timed orig_AMIS(rng, target, model_fit.MAP, model_fit.hess)
    @save fname timed_res

    psis_logws = timed_res.value.psis_logws
    Zhat = logsumexp(psis_logws) - log(length(psis_logws))
    @info String(model_sym) Zhat compute_ess(psis_logws) timed_res.value.pareto_shape
    flush(stderr)
end