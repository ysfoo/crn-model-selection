include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

using Turing, MCMCChains, ProgressMeter
using PDMats, LogExpFunctions
using BridgeSampling, PSIS, StatsBase
using KernelDensity

OUTDIR = joinpath(@__DIR__, "output")
@load "$OUTDIR/MAPs.jld2" model_fits;

n_seed = 100;

Zhat_BIC_dict = Dict(
    sym => begin
        MAP = model_fits[sym].value.MAP
        # hess = model_fits[sym].value.hess
        # Σ = inv(PDMat(hermitianpart!(hess)))
        target = target_dict[sym]
        prior_dists = priors_dict[sym]
        ML = LogDensityProblems.logdensity(target, MAP) - compute_logprior(prior_dists, MAP)
        ML - 0.5*log(n_times)*nparam_dict[sym]
    end for sym in model_syms
)

Zhat_BS_dict = Dict{Symbol,Vector{Float64}}();
re2_BS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "BS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_BS_dict[model_sym] = getproperty.(resvec, :value)
    re2_BS_dict[model_sym] = getproperty.(error_estimate.(resvec), :rmse)
end

Zhat_LIS_dict = Dict{Symbol,Vector{Float64}}();
ess_LIS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "laplace_IS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_LIS_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_LIS_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_orig_dict = Dict{Symbol,Vector{Float64}}();
ess_orig_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "orig_AMIS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_orig_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_orig_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_rAMIS_dict = Dict{Symbol,Vector{Float64}}();
ess_rAMIS_dict = Dict{Symbol,Vector{Float64}}();
for model_sym in model_syms
    resvec = [
        begin
            INFDIR = joinpath(@__DIR__, "output/seed$s");
            fname = joinpath(INFDIR, "robust_AMIS_$(model_sym).jld2");
            @load fname timed_res
            timed_res.value
        end for s in 1:n_seed
    ]
    Zhat_rAMIS_dict[model_sym] = logsumexp.(getproperty.(resvec, :psis_logws)) .- log(10^6)
    ess_rAMIS_dict[model_sym] = getproperty.(resvec, :psis_logws) .|> compute_ess
end

Zhat_gold_dict = Dict(
    sym => logsumexp(Zhats)-log(n_seed) for (sym, Zhats) in Zhat_BS_dict
)

# Combine posteriors

get_K(θ, model_sym) = exp10(θ[2])

get_duration(θ, model_sym) = begin
    invfunc = invfunc_dict[model_sym]
    θpos = exp10.(θ)
    θpos[3] = 5
    invfunc(θpos, 50)
end

# Plot settings

method_colors = Makie.wong_colors()[[1, 3, 4, 2]]

# Combine posteriors
INFDIR = joinpath(@__DIR__, "output/seed1");
func = get_duration
samplesize = 10^5

laplace_BMA_preds, oAMIS_BMA_preds, rAMIS_BMA_preds = [
    begin
        all_preds = Dict(
            model_sym => begin
                fname = joinpath(INFDIR, "$(method_name)_$(model_sym).jld2");
                @load fname timed_res;
                psis_logws = timed_res.value.psis_logws
                idxs = sample(
                    1:10^6, weights(exp.(psis_logws .- maximum(psis_logws))), 
                    samplesize, replace=true
                )
                [func(timed_res.value.all_samples[:,i], model_sym) for i in idxs]
            end for model_sym in model_syms
        );
        logZvec_seed1 = [Zhat_dict[model_sym][1] for model_sym in model_syms]
        BMA_ws = repeat(exp.(logZvec_seed1 .- maximum(logZvec_seed1)), inner=samplesize);
        cat_preds = reduce(vcat, [all_preds[model_sym] for model_sym in model_syms]);
        BMA_preds = sample(cat_preds, weights(BMA_ws), samplesize, replace=true);
    end for (method_name, Zhat_dict) in zip(
        ["laplace_IS", "orig_AMIS", "robust_AMIS"], 
        [Zhat_LIS_dict, Zhat_orig_dict, Zhat_rAMIS_dict]
    )
];

BS_BMA_preds = begin
    all_preds = Dict(
        model_sym => begin
            mcmc_fname = joinpath(INFDIR, "chains_$(model_sym).jld2");
            d = nparam_dict[model_sym]
            @load mcmc_fname chn;
            trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
            samples = reshape(trace, d, :);
            idxs = sample(1:size(samples, 2), samplesize, replace=true)
            [func(samples[:,i], model_sym) for i in idxs]
        end for model_sym in model_syms
    );
    logZvec_seed1 = [Zhat_BS_dict[model_sym][1] for model_sym in model_syms]
    BMA_ws = repeat(exp.(logZvec_seed1 .- maximum(logZvec_seed1)), inner=samplesize);
    cat_preds = reduce(vcat, [all_preds[model_sym] for model_sym in model_syms]);
    BMA_preds = sample(cat_preds, weights(BMA_ws), samplesize, replace=true);
end;

begin
    f = Figure(size=(500, 400))
    # ax = Axis(
    #     f[1, 1], title="BMA for carrying capacity", titlesize=18,
    #     ylabel="Posterior density", ylabelsize=17,
    #     limits=((70, 95), (0, nothing)), yticks=0:0.1:1,
    #     xlabel=L"$K$ (% coral cover)", xlabelsize=17,
    #     xticklabelsize=15
    # )
    ax = Axis(
        f[1, 1], title="BMA for carrying capacity", titlesize=18,
        ylabel="Posterior density", ylabelsize=17,
        limits=((800, 1800), (0, nothing)),
        xlabel=L"$K$ (% coral cover)", xlabelsize=17,
        xticklabelsize=15
    )
    BMA_collection = [
        laplace_BMA_preds, oAMIS_BMA_preds, rAMIS_BMA_preds, BS_BMA_preds,
    ]
    for i in 1:4
        color = method_colors[i]
        k = kde(BMA_collection[i])
        lines!(k.x, k.density, linewidth=2, color=color)
    end

    display(f)
    # save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    # save("$(save_dir)/comparison_coral.png", f, px_per_unit=4);
end
