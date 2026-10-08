include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../stats_helpers.jl"));

# This script takes one command-line argument, which is the seed.
seed = parse(Int64, ARGS[1])

using PDMats, LogExpFunctions, PSIS, ProgressMeter, StableRNGs

@load joinpath(@__DIR__, "output/MAPs.jld2") model_fits;

function laplace_IS(rng, target, MAP, hess, n_samples; df=4)
    Σ = inv(PDMat(hermitianpart!(hess)))
    proposal = MvTDist(df, MAP, Σ)
    samples = rand(rng, proposal, n_samples)

    logps = LogDensityProblems.logdensity.(Ref(target), eachcol(samples))
    logps[findall(isnan, logps)] .= -Inf
    logqs = logpdf.(Ref(proposal), eachcol(samples))
    logws = logps .- logqs

    psis_res = psis(logws; normalize=false, warn=false)

    return (
        # logps = logps,
        # logqs = logqs,
        all_samples = samples,
        psis_logws = psis_res.log_weights,
        pareto_shape = psis_res.pareto_shape
    )
end

n_samples = 10^6
# @showprogress for seed in 1:100
begin
    INFDIR = joinpath(@__DIR__, "output/seed$(seed)");
    mkpath(INFDIR)
    for model_sym in model_syms
        fname = joinpath(INFDIR, "laplace_IS_$(model_sym).jld2")
        target = target_dict[model_sym]

        model_fit = model_fits[model_sym].value
        rng = StableRNG(hash((seed, model_sym, "laplace_IS")))
        timed_res = @timed laplace_IS(rng, target, model_fit.MAP, model_fit.hess, n_samples)
        @save fname timed_res

        Zhat = logsumexp(timed_res.value.psis_logws) - log(n_samples)
        # @info String(model_sym) Zhat compute_ess(timed_res.value.psis_logws) timed_res.value.pareto_shape
        # flush(stderr)
    end
end