include(joinpath(@__DIR__, "gaussian_mixtures.jl"));

using Distributions, LinearAlgebra, Optim, Pathfinder, ProgressMeter, Random
using Logging, LoggingExtras

nowarn_logger = EarlyFilteredLogger(global_logger()) do log
    log.level != Logging.Warn
end


function cs_obj_func(w, p)
    s = 0.
    for i in 1:p.N
        for j in 1:p.K
            p.cache[j] = log(w[j]) + p.log_probs[j, i]
        end
        log_dot = logsumexp(p.cache)
        s += exp(p.log_coefs[i] - logaddexp(log_dot, p.intercepts[i]) - p.offset)
    end    
    return s
end


function cs_grad_func!(grad_cache, w, p)
    fill!(grad_cache, 0.)
    for i in 1:p.N
        for j in 1:p.K
            p.cache[j] = log(w[j]) + p.log_probs[j, i]
        end
        log_dot = logsumexp(p.cache)
        log_fct = p.log_coefs[i] -2logaddexp(log_dot, p.intercepts[i])
        for j in 1:p.K
            grad_cache[j] -= exp(log_fct + p.log_probs[j, i] - p.offset)
        end
    end
end


function create_prior_sampler(dists::Vector{<:Distribution})
    return (rng) -> [rand(rng, dist) for dist in dists]
end


function run_pathfinder(
    target, prior_sampler, prior_means, prior_vars; 
    rng=Random.default_rng(), warn=false, io=stdout)
    tries = 0
    while true
        try
            res = with_logger(nowarn_logger) do
                pf_init = prior_sampler(rng);
                pf_res = pathfinder(
                    target; rng=rng, init=pf_init, ndraws=0, ndraws_elbo=1,
                    history_length=30, optimizer=Optim.LBFGS(; m=30, linesearch=Optim.BackTracking())
                );
                if all(abs.(pf_res.optim_solution.u .- prior_means) .< 4 .* sqrt.(prior_vars))
                    return pf_res # return for do scope
                end
                nothing
            end
            if isnothing(res)
                tries += 1
                mod(tries, 10) == 0 && println(io, "Pathfinder failed $tries times")
                warn && println(io, "Pathfinder returned infeasible solution")
                flush(io)
            else
                return res
            end
        catch e
            e isa InterruptException && rethrow(e)
            println(io, "Pathfinder failed due to $e")
        end
    end
end


function sqhdist_func(dist1::MvNormal, dist2::MvNormal)
    μ1 = dist1.μ
    μ2 = dist2.μ
    Σ1 = dist1.Σ
    Σ2 = dist2.Σ
    Σ_avg = (Σ1 .+ Σ2) ./ 2
    logdet_1 = logdet(Σ1)
    logdet_2 = logdet(Σ2)
    logdet_avg = logdet(Σ_avg)
    Δμ = μ1 - μ2    
    logdet_term = 0.25logdet_1 + 0.25logdet_2 - 0.5logdet_avg
    qf = dot(Δμ, Σ_avg \ Δμ)
    sqhdist = 1 - exp(logdet_term - qf/8)
    return clamp(sqhdist, 0., 1.)
end

# Same as above, from dense covariances with precomputed logdets (computing these for every pair on Pathfinder's
# WoodburyPDMat is about 100 times slower). `Σ_avg` and `Δμ` are workspaces.
function sqhdist_func(μ1, Σ1, logdet_1, μ2, Σ2, logdet_2, Σ_avg, Δμ)
    @. Σ_avg = (Σ1 + Σ2) / 2
    chol_avg = cholesky!(Hermitian(Σ_avg))
    @. Δμ = μ1 - μ2
    logdet_term = 0.25logdet_1 + 0.25logdet_2 - 0.5logdet(chol_avg)
    qf = sum(abs2, chol_avg.U' \ Δμ)
    sqhdist = 1 - exp(logdet_term - qf/8)
    return clamp(sqhdist, 0., 1.)
end


# function revKL_func(dist1::MvNormal, dist2::MvNormal)
#     μ1 = dist1.μ
#     μ2 = dist2.μ
#     Σ1 = dist1.Σ
#     Σ2 = dist2.Σ
#     d = length(μ1)
#     Δμ = μ1 - μ2   
    
#     return 0.5 * (tr(Σ2 \ Σ1) - d + dot(Δμ, Σ2 \ Δμ) + logdet(Σ2) - logdet(Σ1))
# end


function init_dists(target, prior_sampler, prior_means, prior_vars, nruns, Kmax, logp_thres; 
                    progress=false, rng=Random.default_rng(), io=stdout)
    if progress p = Progress(nruns; dt=1) end
    pf_res_vec = [
        begin
            pf_res = run_pathfinder(target, prior_sampler, prior_means, prior_vars; rng=rng, io=io)
            progress && next!(p)
            pf_res
        end for _ in 1:nruns
    ];

    all_dists = reduce(vcat, getproperty.(pf_res_vec, :fit_distributions))
    all_logps = reduce(vcat, map(res->res.optim_trace.log_densities, pf_res_vec))
    has_finite_logp = findall(isfinite, all_logps)
    sort_order = sortperm(all_logps[has_finite_logp], rev=true)
    all_dists = all_dists[has_finite_logp][sort_order]
    all_logps = all_logps[has_finite_logp][sort_order]
    n_all = length(all_dists)
    max_logp = maximum(all_logps)

    cand_idxs = findall(
        i->max_logp - all_logps[i] <= logp_thres && all(diag(all_dists[i].Σ) .< prior_vars) && all(abs.(all_dists[i].μ .- prior_means) .< 4 .* sqrt.(prior_vars)), 
        1:n_all
    )
    # Dense covariances and their logdets, computed once per candidate (indexed by position in cand_idxs)
    cand_μs = [Vector(all_dists[c].μ) for c in cand_idxs]
    cand_Σs = [Matrix(all_dists[c].Σ) for c in cand_idxs]
    cand_logdets = logdet.(cand_Σs)
    Σ_avg = zeros(length(prior_means), length(prior_means))
    Δμ = zeros(length(prior_means))
    viable_pos = Int64[]
    for c in eachindex(cand_idxs)
        is_viable = true
        for v in viable_pos
            if sqhdist_func(cand_μs[c], cand_Σs[c], cand_logdets[c], cand_μs[v], cand_Σs[v], cand_logdets[v], Σ_avg, Δμ) <= 0.1
                is_viable = false
                break
            end
        end
        is_viable && push!(viable_pos, c)
    end
    viable_idxs = cand_idxs[viable_pos]
    viable_dists = all_dists[viable_idxs]    
    viable_logps = all_logps[viable_idxs]
    n_viable = length(viable_idxs)

    if length(viable_dists) <= Kmax
        return viable_dists, viable_dists
    end

    max_idx = argmax(viable_logps)
    keep_dists = MvNormal[viable_dists[max_idx]];
    dist_vec = fill(Inf, n_viable)
    dist_vec[max_idx] = 0.
    # can_skip = (1:n_viable) .== max_idx    
    for k in 2:Kmax
        recent_dist = keep_dists[k - 1]
        for (v, dist) in enumerate(viable_dists)
            # can_skip[v] && continue
            dist_vec[v] = min(norm(recent_dist.μ .- dist.μ), dist_vec[v])
            # if dist_vec[v] < sqhdist_thres
            #     can_skip[v] = true
            # end
        end
        max_idx = argmax(dist_vec)
        push!(keep_dists, viable_dists[max_idx])
        dist_vec[max_idx] = 0.
        # can_skip[max_idx] = true
    end

    # for i in sortperm(viable_logps, rev=true)
    #     dist = viable_dists[i]
    #     Σ = Matrix(dist.Σ)
    #     all(diag(dist.Σ) .< prior_vars) || continue

    #     keep = true
    #     for kept_dist in keep_dists
    #         if sqhdist_func(dist, kept_dist) < sqhdist_thres
    #             keep = false
    #             break
    #         end
    #     end
    #     keep || continue

    #     push!(keep_dists, MvNormal(dist.μ, Σ))
    #     length(keep_dists) == Kmax && break
    # end;

    return viable_dists, keep_dists
end


# function init_dists(target, prior_sampler, prior_means, prior_vars, nruns, Kmax; sqhdist_thres=0.9, progress=false)
#     if progress p = Progress(nruns; dt=1) end
#     pf_res_vec = [
#         begin
#             pf_res = run_pathfinder(target, prior_sampler, prior_means, prior_vars)
#             progress && next!(p)
#             pf_res
#         end for _ in 1:nruns
#     ];

#     all_dists = reduce(vcat, getproperty.(pf_res_vec, :fit_distributions))
#     all_logps = reduce(vcat, map(res->res.optim_trace.log_densities, pf_res_vec))
#     keep_dists = MvNormal[];  
#     for i in sortperm(all_logps, rev=true)
#         dist = all_dists[i]
#         Σ = Matrix(dist.Σ)
#         all(diag(dist.Σ) .< prior_vars) || continue

#         keep = true
#         for kept_dist in keep_dists
#             if sqhdist_func(dist, kept_dist) < sqhdist_thres
#                 keep = false
#                 break
#             end
#         end
#         keep || continue

#         push!(keep_dists, MvNormal(dist.μ, Σ))
#         length(keep_dists) == Kmax && break
#     end;

#     return keep_dists, all_dists
# end