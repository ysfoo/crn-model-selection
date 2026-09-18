### Set all variables named `LOAD_XXX` to false to compute outputs from evidence estimation (for first run), instead of loading saved results.

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter, Suppressor
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using Pathfinder, PSIS, StableRNGs, BridgeSampling
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

@load joinpath(@__DIR__, "data.jld2") all_data;

# Posterior variance
vrats_fname = joinpath(@__DIR__, "output/var_ratios.jld2");
LOAD_VAR_RATIOS = true;
if LOAD_VAR_RATIOS
    @load vrats_fname var_ratios
else
    all_postvars = Matrix{Vector{Float64}}(undef, n_feasible, n_models);
    all_priorvars = Matrix{Vector{Float64}}(undef, n_feasible, n_models);
    @showprogress for dir_idx in 1:n_feasible
        OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

        for model_idx in 1:n_models
            d = nparams[model_idx]
            chains_fname = "$OUTDIR/chains7000_model$model_idx.jld2"
            @load chains_fname chn
            trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
            X = reshape(trace, d, :);
            all_postvars[dir_idx, model_idx] = vec(var(X; dims=2))
            all_priorvars[dir_idx, model_idx] = [fill(4., d-1); 1.]
        end
    end

    var_ratios = broadcast((x,y)->maximum(x./y), all_postvars, all_priorvars)
    @save vrats_fname var_ratios
end

size(var_ratios)
# heatmap(log.(var_ratios))
findall(>(1), var_ratios)
sort(vec(var_ratios), rev=true)[1:10]


hist(sqrt.(vec(var_ratios)), bins=0:0.1:1)
sum(abs.(sqrt.(vec(var_ratios)) .- 0.55) .< 0.05)
# 1401/2816
1458/2816

begin
    dir_idx = 3
    OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

    model_idx = 41
    d = nparams[model_idx]
    chains_fname = "$OUTDIR/chains7000_model$model_idx.jld2"
    @load chains_fname chn    
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
    X = reshape(trace, d, :);

    postvars = vec(var(X; dims=2))
    priorvars = [fill(4., d-1); 1.]
    # display([priorvars postvars])

    params = names(chn, :parameters)

    n_chains = length(chains(chn))
    n_samples = length(chn)

    fig = Figure(size=(600, 900))

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel=string(param))
        for chain in 1:n_chains
            values = chn[:, param, chain]
            lines!(ax, 1:n_samples, values; label=string(chain), alpha=0.5)
        end
        if i < length(params)
            hidexdecorations!(ax; grid=false)
        else
            ax.xlabel = "Iteration"
        end
    end

    fig
end

# Importance sampling effective sample sizes
ess_fname = joinpath(@__DIR__, "output/ess.jld2");
LOAD_ESS = true
if LOAD_ESS
    @load ess_fname LIS_essmat orig_essmat rAMIS_essmat rAMIS_khatmat
else
    LIS_essmat = Matrix{Float64}(undef, n_feasible, n_models);
    orig_essmat = Matrix{Float64}(undef, n_feasible, n_models);
    rAMIS_essmat = Matrix{Float64}(undef, n_feasible, n_models);
    rAMIS_khatmat = Matrix{Float64}(undef, n_feasible, n_models);
    @showprogress for dir_idx in 1:n_feasible
        OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
        for model_idx in 1:n_models
            d = nparams[model_idx]
            fname = "$OUTDIR/laplace_IS_model$model_idx.jld2"
            @load fname timed_res
            LIS_essmat[dir_idx, model_idx] = compute_ess(timed_res.value.psis_logws)
            # continue
            fname = "$OUTDIR/orig_AMIS_model$model_idx.jld2"
            @load fname timed_res
            orig_essmat[dir_idx, model_idx] = compute_ess(timed_res.value.psis_logws)
            fname = "$OUTDIR/robust_AMIS_model$model_idx.jld2"
            @load fname timed_res
            rAMIS_essmat[dir_idx, model_idx] = compute_ess(timed_res.value.psis_logws)
            rAMIS_khatmat[dir_idx, model_idx] = timed_res.value.pareto_shape
        end
    end
    @save ess_fname LIS_essmat orig_essmat rAMIS_essmat rAMIS_khatmat
end

MCMCstats_fname = joinpath(@__DIR__, "output/MCMCstats.jld2");
LOAD_MCMCSTATS = true;
if LOAD_MCMCSTATS
    @load MCMCstats_fname MCMC_miness MCMC_maxrhat MCMC_times
else
    MCMC_miness = [Float64[] for _ in 1:n_feasible];
    MCMC_maxrhat = [Float64[] for _ in 1:n_feasible];
    MCMC_times = [Float64[] for _ in 1:n_feasible];
    @showprogress for dir_idx in 1:n_feasible
        OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)")
        for model_idx in 1:n_models
            fname = joinpath(OUTDIR, "chains7000_model$(model_idx).jld2");
            @load fname chn ess_df
            push!(MCMC_miness[dir_idx], minimum(ess_df.nt.ess))
            push!(MCMC_maxrhat[dir_idx], maximum(rhat(chn).nt.rhat))
            push!(MCMC_times[dir_idx], MCMCChains.compute_duration(chn) / 60)
        end
    end
    @save MCMCstats_fname MCMC_miness MCMC_maxrhat MCMC_times
end

MCMC_essmat = stack(MCMC_miness)';
summarystats((MCMC_times ./ 60) .|> sum)

hist(vec(MCMC_essmat), bins=0:100:2000)
summarystats(vec(MCMC_essmat))

tmp_idxs = partialsortperm(vec(MCMC_essmat), 1:10);
top_idxs = CartesianIndices(MCMC_essmat)[tmp_idxs]
MCMC_essmat[top_idxs]

sum(reduce(vcat, MCMC_maxrhat) .> 1.01)
maximum(MCMC_maxrhat .|> maximum)
for dir_idx in 1:n_feasible
    for model_idx in 1:n_models
        if MCMC_maxrhat[dir_idx][model_idx] > 1.01
            display((dir_idx, model_idx))
        end
    end
end


# OUTDIR = joinpath(@__DIR__, "output/data25")
# for model_idx in 1:n_models
#     fname = joinpath(OUTDIR, "chains7000_model$(model_idx).jld2");
#     @load fname chn
#     display(chn.info.stop_time .- chn.info.start_time)
#     display(diff(sort(chn.info.start_time)))
#     break
# end

# sum(MCMC_times[25]) / 60
# 2*24+17 # job wall clock time
# 7*24+17 # cpu utilized

logZs_fname = joinpath(@__DIR__, "output/logZs.jld2");

LOAD_LOGZS = true
if LOAD_LOGZS
    @load logZs_fname all_times BIC_logZvecs LIS_logZvecs orig_logZvecs rAMIS_logZvecs BS_logZvecs
else
    all_times = Vector{Float64}[];
    BIC_logZvecs = [Float64[] for _ in 1:n_feasible];
    LIS_logZvecs = [Float64[] for _ in 1:n_feasible];
    orig_logZvecs = [Float64[] for _ in 1:n_feasible];
    rAMIS_logZvecs = [Float64[] for _ in 1:n_feasible];
    BS_logZvecs = [Float64[] for _ in 1:n_feasible];

    @showprogress for dir_idx in 1:n_feasible
        MAP_time = 0.
        hess_time = 0.
        LIS_time = 0.
        orig_time = 0.
        rAMIS_time = 0.
        chains_time = 0.
        BS_time = 0.

        OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
        @nowarn_load "$OUTDIR/MAP.jld2" model_fits fit_times;
        @load "$OUTDIR/MAP_hess.jld2" hess_times;
        genmodel_idx = feasible_idxs[dir_idx]
        data = all_data[genmodel_idx];
        petab_models = [create_petab_model(model, data, u0) for model in models];
        petab_probs = [PEtabODEProblem(pmodel; odesolver=ODESolver(Rodas5P(), verbose=false)) for pmodel in petab_models];

        MAP_time += sum(fit_times) / 60
        hess_time += sum(hess_times) / 60

        for model_idx in 1:n_models
            xmin = model_fits[model_idx].xmin;
            fmin = model_fits[model_idx].fmin;
            n_t = length(data.t);
            logp_BIC = -fmin -petab_probs[model_idx].prior(collect(xmin)) - 0.5nparams[model_idx]*log(n_t)
            push!(BIC_logZvecs[dir_idx], logp_BIC)

            fname = "$OUTDIR/laplace_IS_model$model_idx.jld2"
            @load fname timed_res
            LIS_time += timed_res.time / 60
            logws = timed_res.value.psis_logws;
            N = length(logws);
            logp_LIS = logsumexp(logws) - log(N)
            push!(LIS_logZvecs[dir_idx], logp_LIS)

            fname = "$OUTDIR/orig_AMIS_model$model_idx.jld2"
            @load fname timed_res
            orig_time += timed_res.time / 60
            logws = timed_res.value.psis_logws;
            N = length(logws);
            logp_orig = logsumexp(logws) - log(N)
            push!(orig_logZvecs[dir_idx], logp_orig)

            fname = "$OUTDIR/robust_AMIS_model$model_idx.jld2"
            @load fname timed_res
            rAMIS_time += timed_res.time / 60
            logws = timed_res.value.psis_logws;
            N = length(logws);
            logp_rAMIS = logsumexp(logws) - log(N)
            push!(rAMIS_logZvecs[dir_idx], logp_rAMIS)

            fname = "$OUTDIR/chains7000_model$model_idx.jld2"
            @load fname chn
            chains_time += MCMCChains.compute_duration(chn) / 60

            fname = "$OUTDIR/BSnew_model$model_idx.jld2"
            @load fname timed_res
            BS_time += timed_res.time / 60
            push!(BS_logZvecs[dir_idx], timed_res.value.value)
        end

        # @info "Data $dir_idx time (min)" MAP_time MAP_time+hess_time+LIS_time MAP_time+hess_time+orig_time rAMIS_time chains_time BS_time
        times = [MAP_time, MAP_time+hess_time+LIS_time, MAP_time+hess_time+orig_time, rAMIS_time, chains_time+BS_time]
        
        push!(all_times, times)
        flush(stderr)
    end
    @save logZs_fname all_times BIC_logZvecs LIS_logZvecs orig_logZvecs rAMIS_logZvecs BS_logZvecs
end

hrs_mat = stack(all_times) ./ 60; # method x datasets

hrs_mat[5, 25] / hrs_mat[4, 25]

function calc_tvd(logZvec1, logZvec2)
    logp1 = exp.(logZvec1 .- logsumexp(logZvec1))
    logp2 = exp.(logZvec2 .- logsumexp(logZvec2))
    return 0.5sum(abs, logp1.-logp2)
end

LIS_errors = reduce(vcat, LIS_logZvecs .- BS_logZvecs);
orig_errors = reduce(vcat, orig_logZvecs .- BS_logZvecs);
rAMIS_errors = reduce(vcat, rAMIS_logZvecs .- BS_logZvecs);

mean(LIS_errors .|> abs .< 0.1)
mean(orig_errors .|> abs .< 0.1)
mean(rAMIS_errors .|> abs .< 0.1)

BIC_tvds = calc_tvd.(BIC_logZvecs, BS_logZvecs);
LIS_tvds = calc_tvd.(LIS_logZvecs, BS_logZvecs);
orig_tvds = calc_tvd.(orig_logZvecs, BS_logZvecs);
rAMIS_tvds = calc_tvd.(rAMIS_logZvecs, BS_logZvecs);
all_tvds = [BIC_tvds, LIS_tvds, orig_tvds, rAMIS_tvds];

pvecs_BIC = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in BIC_logZvecs];
pvecs_LIS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in LIS_logZvecs];
pvecs_orig = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in orig_logZvecs];
pvecs_rAMIS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in rAMIS_logZvecs];
pvecs_BS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in BS_logZvecs];


summarystats(hrs_mat[1,:]) # BIC
summarystats(hrs_mat[2,:]) # LIS
summarystats(hrs_mat[3,:]) # orig AMIS
summarystats(hrs_mat[4,:]) # new AMIS
summarystats(hrs_mat[5,:]) # BS

((MCMC_times .|> sum) |> mean) / 60 / mean(hrs_mat[5,:])

# summarystats(BIC_tvds)
# summarystats(LIS_tvds)
summarystats(orig_tvds)
summarystats(rAMIS_tvds)

sortperm(rAMIS_tvds)

dir_idx = 20
pdiffs = pvecs_rAMIS[dir_idx] .- pvecs_BS[dir_idx];
scatter(MCMC_essmat[dir_idx,:], pdiffs)


COLORS = [:grey60; Makie.wong_colors()[[1, 3, 4, 2]]];
method_names = ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"];
MARKERS = [:rect, :circle, :diamond, :xcross]

all_tvds .|> maximum
tvd_bins = 0:0.01:0.34
tvd_hists = fit.(Histogram, all_tvds, Ref(tvd_bins));
tvd_histmaxs = getproperty.(tvd_hists, :weights) .|> maximum
tvd_heights = tvd_histmaxs ./ maximum(tvd_histmaxs)

all_essvecs = [vec(LIS_essmat), vec(orig_essmat), vec(rAMIS_essmat)];
extrema.(all_essvecs)
ess_bins = logrange(1, 1e6, 31)[3:end]
ess_hists = fit.(Histogram, all_essvecs, Ref(ess_bins));
ess_histmaxs = getproperty.(ess_hists, :weights) .|> maximum
ess_heights = ess_histmaxs ./ maximum(ess_histmaxs)

begin
    f = Figure(size=(1200, 1200))

    cats = repeat(1:4, inner=n_feasible)

    ax11 = Axis(
        f[1,1],
        yticklabelsize=16,
        limits=(nothing, (-0.01, nothing)),
        ylabel="Total variation distance\nfrom bridge sampling", ylabelsize=18,
        title="Discrepancy of posterior distributions\nfrom gold standard", titlesize=18,
        xlabel="Per-dataset runtime (hours)", xlabelsize=18,
        xticklabelsize=16, 
    )
    
    for i in 1:4
        scatter!(
            hrs_mat[i,:], all_tvds[i], alpha=0.8,
            color=COLORS[i], marker=MARKERS[i], label=method_names[i],
        )
    end

    axislegend(ax11, position=:rt, labelsize=18)

    var_ratios_vec = reduce(vcat, eachrow(var_ratios))

    ax12 = Axis(
        f[1,2], 
        title="Estimate error vs\nparameter identifiability", titlesize=18,
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Log-evidence error relative\nto bridge sampling",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        # yscale=symsqrt, 
        # yticks=(tick_vals, tick_labels),
        limits=((0, 1.1), nothing)
    )
    lines!([0, 1.1], [0, 0], alpha=0.5, color=:black, linestyle=:dash)
    func = identity
    for (i, errs) in enumerate([LIS_errors, orig_errors, rAMIS_errors])
        scatter!(
            sqrt.(var_ratios_vec), errs,
            markersize=6, alpha=0.4, color=COLORS[i+1], marker=MARKERS[i+1],
        )
    end

    cats = repeat(1:3, inner=(n_feasible*n_models))

    ax21 = Axis(
        f[2,1],
        yticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6]),
        yticklabelsize=16,
        yscale=log10,
        ylabel="Effective sample size", ylabelsize=18,
        title="Effective sample size for\nimportance sampling methods", titlesize=18,
        # xticks=(1:3, ["Laplace IS", "Standard AMIS", "Robust AMIS"]),
        # xticklabelrotation=π/6,
        xticks=(1:3, ["Laplace\nIS", "Standard\nAMIS", "Robust\nAMIS"]),
        xticklabelsize=18, xgridvisible=true
    )
    
    for i in 1:3
        hist!(
            all_essvecs[i], color=COLORS[i+1], 
            scale_to=-0.9*ess_heights[i], offset=i, 
            direction=:x, bins=ess_bins
        )
    end

    ax22 = Axis(
        f[2,2], 
        title="Effective sample size vs\nparameter identifiability", titlesize=18,
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Effective sample size",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        yscale=log10, 
        yticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6]),
        limits=((0, 1.1), nothing)
    )
    for (i, essmat) in enumerate([LIS_essmat, orig_essmat, rAMIS_essmat])
        scatter!(
            sqrt.(var_ratios_vec),
            vec(essmat'), markersize=6, alpha=0.4, color=COLORS[i+1], marker=MARKERS[i+1],
        )
    end 

    linkyaxes!(ax21, ax22)

    colgap!(f.layout, 1, 25)

    Legend(
        f[:, 3],
        [MarkerElement(color=COLORS[c], marker=MARKERS[c], markersize=8) for c in 2:4],
        method_names[2:4],
        labelsize=18, tellheight=false, 
    )

    for i in 1:4
        label = ["A", "B", "C", "D"][i]
        loc = [f[1, 1, TopLeft()], f[1, 2, TopLeft()], f[2, 1, TopLeft()], f[2, 2, TopLeft()]][i]
        rpad = [32, 32, 32, 32][i]
        Label(loc, label,
            fontsize = 24, font = :bold,
            padding = (0, rpad, 0, -30), # left, right, bottom, top
            halign = :right, valign = :center,
        )
    end

    g = GridLayout(f[3, :])
    for (i, pvecs_other) in enumerate([pvecs_BIC, pvecs_LIS, pvecs_orig, pvecs_rAMIS])
        ax = Axis(
            g[2,i], aspect=DataAspect(), xticks=0:0.2:1,
            xlabel=method_names[i], xlabelsize=18, 
            xticklabelsize=16, yticklabelsize=16,
            ylabel=i == 1 ? "Bridge sampling" : "", ylabelsize=18
        )
        vcat_pvecs_other = reduce(vcat, pvecs_other)
        pmax = pvecs_BS .|> maximum |> maximum
        lines!([0, pmax], [0, pmax], color=(:grey10, 0.8), linestyle=:dash)
        sc = scatter!(
            vcat_pvecs_other, reduce(vcat, pvecs_BS),
            color=sqrt.(vec(var_ratios')), colorrange=(0, 0.6), #colormap=Reverse(:viridis),
            highclip=:yellow,
            alpha=0.6, markersize=7
        )
        colsize!(g, i, Auto(maximum(vcat_pvecs_other)-minimum(vcat_pvecs_other)))
        if i == 4
            Colorbar(
                g[2,end+1], sc, ticklabelsize=16, alignmode=Mixed(right=0),
                label="Max posterior-to-prior SD ratio", labelsize=18
            )
        end
    end
    
    Label(g[1,:], "Comparison of model posterior probabilities", fontsize=20, font=:bold)
    Label(g[1,1,Left()], "E",
        fontsize = 24, font = :bold,
        padding = (0,32, 0, 0), # left, right, bottom, top
        halign = :right, valign = :center,
    )
    rowgap!(g, 1, -10)
    rowsize!(f.layout, 3, Relative(0.28))

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/comparison_insect.png", f, px_per_unit=4);
end



begin
    f = Figure(size=(1200, 320))
    for (i, pvecs_other) in enumerate([pvecs_BIC, pvecs_LIS, pvecs_orig, pvecs_rAMIS])
        ax = Axis(
            f[1,i], aspect=DataAspect(), xticks=0:0.2:1,
            xlabel=method_names[i], xlabelsize=18, 
            xticklabelsize=16, yticklabelsize=16,
            ylabel=i == 1 ? "Bridge sampling" : "", ylabelsize=18
        )
        vcat_pvecs_other = reduce(vcat, pvecs_other)
        sc = scatter!(
            vcat_pvecs_other, reduce(vcat, pvecs_BS),
            color=sqrt.(vec(var_ratios')), colorrange=(0, 0.6), #colormap=Reverse(:viridis),
            highclip=:yellow,
            alpha=0.6, markersize=8
        )
        colsize!(f.layout, i, Auto(maximum(vcat_pvecs_other)-minimum(vcat_pvecs_other)))
        if i == 4
            Colorbar(f[1,end+1], sc, ticklabelsize=16, label="Max posterior-to-prior SD ratio", labelsize=18)
        end
    end
    # colgap!(f.layout, 0)
    Label(f[0,:], "Comparison of model posterior probabilities", fontsize=20, font=:bold)
    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/modelposts_identifiability.png", f, px_per_unit=4);
end

begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        title="Estimate error vs\neffective sample size", titlesize=18,
        xlabel="Effective sample size",
        ylabel="Log-evidence error relative\nto bridge sampling",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        xscale=log10,
        # yscale=symsqrt, 
        # yticks=(tick_vals, tick_labels),
        yticks=-1:0.1:1,
        # limits=((0, 1.1), nothing)
    )
    lines!([0, 1.1], [0, 0], alpha=0.5, color=:black, linestyle=:dash)
    func = identity
    for (i, errs) in enumerate([LIS_errors, orig_errors, rAMIS_errors])
        i == 3 || continue
        essmat = [LIS_essmat, orig_essmat, rAMIS_essmat][i]
        scatter!(
            vec(essmat'), errs,
            markersize=6, alpha=0.4, color=COLORS[i+1], marker=MARKERS[i+1],
        )
    end
    display(f)
end

exit()

## Playground

# length(rAMIS_logZvecs)
# length(MCMC_maxrhat)
# plot_order = sortperm(abs.(rAMIS_errors));

# scatter(
#     vec(rAMIS_essmat')[plot_order], reduce(vcat, MCMC_maxrhat)[plot_order],
#     color=rAMIS_errors[plot_order], colorrange=(-1.1, 1.1), colormap=:curl,
#     axis=(xscale=log10,)
# )

# scatter(vec(rAMIS_essmat')[plot_order], rAMIS_errors[plot_order])

# scatter(vec(rAMIS_khatmat')[plot_order], rAMIS_errors[plot_order])

# vec(rAMIS_essmat')[abs.(rAMIS_errors) .> 0.5]

# summarystats(rAMIS_essmat)
# sum(rAMIS_essmat .< 1e5)
# sum(rAMIS_essmat .< 1e4)

# sum(rAMIS_khatmat .> 0.9)
# sum(rAMIS_khatmat .> 1)


# quantile.(Ref(rAMIS_errors .|> abs), 0.05:0.05:0.95)


# mean(LIS_essmat .> 1e4)
# mean(orig_essmat .> 1e4)
# mean(rAMIS_essmat .> 1e4)

# scatter(reduce(vcat, eachrow(rAMIS_essmat)), rAMIS_errors)

# size(rAMIS_essmat)

# rAMIS_avgess = [
#     logsumexp(log.(rAMIS_essvec) .+ rAMIS_logZvec) - logsumexp(rAMIS_logZvec)
#     for (rAMIS_essvec, rAMIS_logZvec) in zip(eachrow(rAMIS_essmat), rAMIS_logZvecs)
# ] .|> exp;

# rAMIS_hmess = [
#     logsumexp(rAMIS_logZvec) - logsumexp(rAMIS_logZvec .- log.(rAMIS_essvec))
#     for (rAMIS_essvec, rAMIS_logZvec) in zip(eachrow(rAMIS_essmat), rAMIS_logZvecs)
# ] .|> exp;

# scatter(vec(minimum(rAMIS_essmat; dims=2)), rAMIS_tvds, axis=(xscale=log10,))
# scatter(vec(maximum(rAMIS_khatmat; dims=2)), rAMIS_tvds)

# scatter(rAMIS_avgess, rAMIS_tvds, axis=(xscale=log10,))
# scatter(rAMIS_hmess, rAMIS_tvds, axis=(xscale=log10,))

# function guess_tvd(logZvec, essvec)
#     M = length(logZvec)
#     return maximum(
#         begin
#             alt = copy(logZvec)
#             alt[m] += 10/sqrt(essvec[m])
#             calc_tvd(logZvec, alt)
#         end for m in 1:M
#     )
# end

# guesses = [guess_tvd(rAMIS_logZvecs[dir_idx], rAMIS_essmat[dir_idx,:]) for dir_idx in 1:44];
# scatter(guesses, rAMIS_tvds)

# dir_idx = 21;
# scatter(
#     exp.(BS_logZvecs[dir_idx] .- logsumexp(BS_logZvecs[dir_idx])),
#     # sqrt.(1 ./ rAMIS_essmat[dir_idx,:]),
#     rAMIS_logZvecs[dir_idx] .- BS_logZvecs[dir_idx]
# )

# scatter(
#     guesses, rAMIS_tvds
# )

# hist(log10.(vec(rAMIS_essmat)))

# begin
#     f = Figure(size=(1200, 675))

#     ax = Axis(
#         f[1,1],
#         yscale=log10, yticklabelsize=15,
#         ylabel="Computation time (hr)", ylabelsize=17,
#         title="Computation time per dataset", titlesize=18
#     )
#     for i in 1:5
#         boxplot!(fill(i,n_feasible), hrs_mat[i,:], color=COLORS[i])
#     end

#     ax = Axis(f[1,2])

#     ax = Axis(f[1,3])

#     Legend(
#         f[2,1:2],
#         [PolyElement(color=COLORS[c]) for c in 1:5],
#         ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"],
#         labelsize=17, tellwidth=false, orientation=:horizontal
#     )

#     display(f)
# end


# begin
#     f = Figure(size=(1000, 900))

#     ax = Axis(
#         f[1,1],
#         xticklabelsize=16, yticklabelsize=16,
#         yscale=sqrt, yticks=[0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
#         xlabel="Computation time per dataset (hr)", xlabelsize=18,
#         ylabel="Total variation distance", ylabelsize=18,
#         title="Performance of model selection methods", titlesize=18
#     )
#     for i in 1:4
#        scatter!(
#         hrs_mat[i,:], all_tvds[i], 
#         color=COLORS[i], marker=MARKERS[i], label=method_names[i]
#     )
#     end
#     axislegend(ax, position=:lb, labelsize=18)

#     n = 2
#     tick_vals = [.-(0.1 .^ (0:n)); 0; 0.1 .^ (0:n)]
#     tick_labels = [
#         v == 0 ? L"0" :
#         begin
#             exp = round(Int, log10(abs(v)))
#             sign = v < 0 ? "-\\!" : ""
#             L"%$(sign)10^{%$exp}"
#         end
#         for v in tick_vals
#     ]

#     ax = Axis(
#         f[1,2],
#         title="Log-evidence errors relative\nto bridge sampling",
#         xscale=Makie.Symlog10(1/10^n),
#         yscale=Makie.Symlog10(1/10^n),
#         limits=((-6, 6), (-6, 6)),
#         xticks=(tick_vals, tick_labels), xticklabelsize=16,
#         yticks=(tick_vals, tick_labels), yticklabelsize=16,
#         titlesize=18,
#         xlabel="Standard AMIS", xlabelsize=18,
#         ylabel="Robust AMIS", ylabelsize=18
#     )
    
#     band!([-20,-1/10^n], [-20,-20], [20,20], color=:grey, alpha=0.15)
#     band!([1/10^n,20], [-20,-20], [20,20], color=:grey, alpha=0.15)
#     band!([-20,20], [1/10^n,1/10^n], [20,20], color=:grey, alpha=0.15)
#     band!([-20,20], [-20,-20], [-1/10^n,-1/10^n], color=:grey, alpha=0.15)

#     rng = StableRNG(2)
#     plot_order = shuffle(rng, 1:n_feasible*n_models)

#     var_ratios_vec = reduce(vcat, eachrow(var_ratios))
#     sc = scatter!(
#         reduce(vcat, orig_logZvecs .- BS_logZvecs)[plot_order],
#         reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[plot_order],
#         color=var_ratios_vec[plot_order], colorscale=log10, alpha=0.6, markersize=8,
#     )
#     Colorbar(f[1:2,3], sc, label="Posterior-to-prior variance ratio", labelsize=18, ticklabelsize=16, tellheight=false, height=Auto(0.8))
    
#     ax = Axis(
#         f[2,1], title="Performance of standard AMIS",
#         titlesize=18,
#         xlabel="Effective sample size", xlabelsize=18,
#         ylabel="Log-evidence errors relative\nto bridge sampling", ylabelsize=18,
#         xticklabelsize=16, yticklabelsize=16,
#         limits=((2, 1.2e6), (-6, 6)),
#         xscale=log10,
#         yscale=Makie.Symlog10(1/10^n), 
#         yticks=(tick_vals, tick_labels),
#         xticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6])
#     )

#     band!([1,1e7], [-20,-20], [-1/10^n,-1/10^n], color=:grey, alpha=0.15)
#     band!([1,1e7], [1/10^n,1/10^n], [20,20], color=:grey, alpha=0.15)

#     scatter!(
#         reduce(vcat, eachrow(orig_essmat))[plot_order],
#         reduce(vcat, orig_logZvecs .- BS_logZvecs)[plot_order],
#         color=var_ratios_vec[plot_order], colorscale=log10, alpha=0.6, markersize=8,
#     )

#     ax = Axis(
#         f[2,2], title="Performance of Robust AMIS",
#         titlesize=18,
#         xlabel="Effective sample size", xlabelsize=18,
#         ylabel="Log-evidence errors relative\nto bridge sampling", ylabelsize=18,
#         xticklabelsize=16, yticklabelsize=16,
#         limits=((2, 1.2e6), (-6, 6)),
#         xscale=log10,
#         yscale=Makie.Symlog10(1/10^n), 
#         yticks=(tick_vals, tick_labels),
#         xticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6])
#     )

#     band!([1,1e7], [-20,-20], [-1/10^n,-1/10^n], color=:grey, alpha=0.15)
#     band!([1,1e7], [1/10^n,1/10^n], [20,20], color=:grey, alpha=0.15)

#     scatter!(
#         reduce(vcat, eachrow(rAMIS_essmat))[plot_order],
#         reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[plot_order],
#         color=var_ratios_vec[plot_order], colorscale=log10, alpha=0.6, markersize=8,
#     )

#     display(f)
#     save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
#     save("$(save_dir)/comparison_insect.png", f, px_per_unit=4);
# end

for dir_idx in 1:44
    logZdiffs = rAMIS_logZvecs[dir_idx] .- BS_logZvecs[dir_idx];
    for model_idx in 1:64
        essval = rAMIS_essmat[dir_idx, model_idx]
        if essval > 1e5 && logZdiffs[model_idx] < -0.2
            @info "Outlier" dir_idx model_idx MCMC_essmat[dir_idx, model_idx]
        end
    end
end


hist(log10.(var_ratios_vec))
hist(sqrt.(var_ratios_vec))
mean(sqrt.(var_ratios_vec) .< 0.2)

scatter(var_ratios_vec, reduce(vcat, rAMIS_logZvecs .- BS_logZvecs))
sub_idxs = findall(reduce(vcat, rAMIS_logZvecs .- BS_logZvecs) .|> abs .> log(2))
sqrt(minimum(var_ratios_vec[sub_idxs]))

scatter(var_ratios_vec, reduce(vcat, orig_logZvecs .- BS_logZvecs))
sub_idxs = findall(reduce(vcat, orig_logZvecs .- BS_logZvecs) .|> abs .> log(2))
sqrt(minimum(var_ratios_vec[sub_idxs]))

scatter(var_ratios_vec, reduce(vcat, LIS_logZvecs .- BS_logZvecs))
sub_idxs = findall(reduce(vcat, LIS_logZvecs .- BS_logZvecs) .|> abs .> log(2))
sqrt(minimum(var_ratios_vec[sub_idxs]))

sub_idxs = findall(var_ratios_vec .< 0.05)
reduce(vcat, LIS_logZvecs .- BS_logZvecs)[sub_idxs] |> extrema .|> exp

stairs(1:2, 1:2, step=:post)

tick_vals = [-4., -2., -1., -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0, 4.0]
tick_labels = [
    begin
        val = abs(round(v) ≈ v ? Int(v) : v)
        sign = v < 0 ? "-\\!" : ""
        L"%$sign%$val"
    end for v in tick_vals
]

vratio_order = sortperm(var_ratios_vec);
LIS_errors = reduce(vcat, LIS_logZvecs .- BS_logZvecs)[vratio_order];
orig_errors = reduce(vcat, orig_logZvecs .- BS_logZvecs)[vratio_order];
rAMIS_errors = reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[vratio_order];
begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        xlabel="Threshold for max posterior-to-prior SD ratio",
        ylabel="Worst log-evidence error magnitude",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        yscale=symsqrt, 
        yticks=(tick_vals, tick_labels),
        limits=(nothing,(0, nothing))
    )
    stairs!(
        sqrt.(var_ratios_vec[vratio_order]),
        accumulate(max, LIS_errors .|> abs), step=:post, color=COLORS[2]
    )
    stairs!(
        sqrt.(var_ratios_vec[vratio_order]),
        accumulate(max, orig_errors .|> abs), step=:post, color=COLORS[3]
    )
    stairs!(
        sqrt.(var_ratios_vec[vratio_order]),
        accumulate(max, rAMIS_errors .|> abs), step=:post, color=COLORS[4]
    )
    display(f)
end

begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Log-evidence error",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        yscale=symsqrt, 
        yticks=(tick_vals, tick_labels),
        # limits=(nothing,(-0.001, nothing))
    )
    func = identity
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        LIS_errors .|> func, markersize=4, alpha=0.5, color=COLORS[2]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        orig_errors .|> func, markersize=4, alpha=0.5, color=COLORS[3]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        rAMIS_errors .|> func, markersize=4, alpha=0.5, color=COLORS[4]
    )
    display(f)
end

begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Log-evidence error",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        # yscale=log10, 
        # yticks=(tick_vals, tick_labels),
        limits=((0., nothing), nothing)
    )
    func = identity
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        LIS_errors .|> func, markersize=6, alpha=0.4, color=COLORS[2]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        orig_errors .|> func, markersize=6, alpha=0.4, color=COLORS[3]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        rAMIS_errors .|> func, markersize=6, alpha=0.4, color=COLORS[4]
    )
    display(f)
end

begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Log-evidence error relative\nto bridge sampling",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        # yscale=log10, 
        # yticks=(tick_vals, tick_labels),
        limits=((0., nothing), nothing)
    )
    func = identity
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        LIS_errors .|> func, markersize=6, alpha=0.4, color=COLORS[2]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        orig_errors .|> func, markersize=6, alpha=0.4, color=COLORS[3]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        rAMIS_errors .|> func, markersize=6, alpha=0.4, color=COLORS[4]
    )
    display(f)
end


begin
    f = Figure()
    ax = Axis(
        f[1,1], 
        xlabel="Max posterior-to-prior SD ratio",
        ylabel="Effective sample size",
        xticklabelsize=16, yticklabelsize=16,
        xlabelsize=18, ylabelsize=18,
        yscale=log10, 
        # yticks=(tick_vals, tick_labels),
        limits=((0., nothing), nothing)
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        LIS_essmat |> vec, markersize=6, alpha=0.4, color=COLORS[2]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        orig_essmat |> vec, markersize=6, alpha=0.4, color=COLORS[3]
    )
    scatter!(
        sqrt.(var_ratios_vec[vratio_order]),
        rAMIS_essmat |> vec, markersize=6, alpha=0.4, color=COLORS[4]
    )
    display(f)
end

# Log-evidence errors vs ESS
begin
    f = Figure()

    tick_vals = [-4., -3., -2., -1., -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0]
    tick_labels = [
        begin
            val = abs(round(v) ≈ v ? Int(v) : v)
            sign = v < 0 ? "-\\!" : ""
            L"%$sign%$val"
        end for v in tick_vals
    ]

    ax = Axis(
        f[1,1], title="Performance of Robust AMIS",
        titlesize=18,
        xlabel="Effective sample size", xlabelsize=18,
        ylabel="Log-evidence errors relative\nto bridge sampling", ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16,
        # limits=((2, 1.2e6), (-6, 6)),
        xscale=log10,
        yscale=symsqrt,
        yticks=(tick_vals, tick_labels),
        xticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6])
    )

    scatter!(
        reduce(vcat, eachrow(rAMIS_essmat))[plot_order],
        reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[plot_order],
        color=var_ratios_vec[plot_order], colorscale=log10, alpha=0.5, markersize=7,
    )

    display(f)
end


# Compare ESS
begin
    f = Figure(size=(400, 500))

    cats = repeat(1:3, inner=(n_feasible*n_models))

    ax = Axis(
        f[1,1],
        yticklabelsize=16,
        yscale=log10,
        ylabel="Effective sample size", ylabelsize=18,
        title="Effective sample size for\nimportance sampling methods", titlesize=18,
        # xticks=(1:3, ["Laplace IS", "Standard AMIS", "Robust AMIS"]),
        # xticklabelrotation=π/6,
        xticks=(1:3, ["Laplace\nIS", "Standard\nAMIS", "Robust\nAMIS"]),
        xticklabelsize=18
    )
    
    # rainclouds!(
    #     cats, vcat(vec(LIS_essmat), vec(orig_essmat), vec(rAMIS_essmat)),
    #     color=COLORS[2:4][cats], markersize=2,
    #     plot_boxplots=false, clouds=nothing, 
    #     gap=-0.1, dodge_gap=0.1,
    #     jitter_width=0.67
    # )

    for (i, essmat) in enumerate([LIS_essmat, orig_essmat, rAMIS_essmat])
        hist!(
            ax, vec(essmat), bins=logrange(1, 1e6, 25), 
            scale_to=-0.7, offset=i, direction=:x, color=COLORS[i+1]
        )
    end

    display(f)
end


# Old comparison plots

begin
    f = Figure(size=(1000, 920))

    cats = repeat(1:4, inner=n_feasible)

    ax = Axis(
        f[1,1],
        yticklabelsize=16,
        limits=(nothing, (-0.01, nothing)),
        ylabel="Total variation distance\nfrom bridge sampling", ylabelsize=18,
        title="Accuracy of model selection methods", titlesize=18,
        # xticks=(1:4, ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS"]),
        # xticklabelrotation=π/6, 
        xticks=(1:4, ["BIC", "Laplace\nIS", "Standard\nAMIS", "Robust\nAMIS"]),
        xticklabelsize=18
    )
    
    for i in 1:4
        rainclouds!(
            fill(i, n_feasible), all_tvds[i],
            color=COLORS[i],
            plot_boxplots=false, 
            jitter_width=0.2, markersize=6,
            show_median=false, 
            clouds=hist, cloud_width=tvd_heights[i], gap=0.0, dodge_gap=0.1, side_nudge=0.125, hist_bins=tvd_bins,
            # clouds=violin, cloud_width=1, violin_limits=(0, Inf),
            
        )
    end
    # for i in 1:4
    #    scatter!(
    #     hrs_mat[i,:], all_tvds[i], 
    #     color=COLORS[i], marker=MARKERS[i], label=method_names[i]
    # )
    # end
    # axislegend(ax, position=:lb, labelsize=18)

    tick_vals = [-4., -2., -1., -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0, 4.0]
    tick_labels = [
        begin
            val = abs(round(v) ≈ v ? Int(v) : v)
            sign = v < 0 ? "-\\!" : ""
            L"%$sign%$val"
        end for v in tick_vals
    ]

    ax = Axis(
        f[1,2],
        title="Log-evidence errors relative\nto bridge sampling",
        xscale=symsqrt,
        yscale=symsqrt,
        # limits=((-6, 2), (-6, 2)),
        xticks=(tick_vals, tick_labels), yticks=(tick_vals, tick_labels),
        xticklabelsize=16, yticklabelsize=16,
        titlesize=18,
        xlabel="Standard AMIS", xlabelsize=18,
        ylabel="Robust AMIS", ylabelsize=18
    )

    var_ratios_vec = reduce(vcat, eachrow(var_ratios))

    plot_order = sortperm(var_ratios_vec, rev=true)
    # rng = StableRNG(1)
    # plot_order = shuffle(rng, 1:n_feasible*n_models)
    
    sc = scatter!(
        reduce(vcat, orig_logZvecs .- BS_logZvecs)[plot_order],
        reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[plot_order],
        color=sqrt.(var_ratios_vec[plot_order]), colormap=Reverse(:viridis), alpha=0.6, markersize=7,
        # color=var_ratios_vec[plot_order], colorscale=log10, alpha=0.5, markersize=7,
    )
    autolimits!(ax)
    ax_limits = ax.finallimits[]
    lines!([-6, 2], [-6, 2], color=:black, alpha=0.4, linestyle=:dash)
    lines!([-6, 2], [6, -2], color=:black, alpha=0.4, linestyle=:dash)
    limits!(ax, ax_limits)
    
    Colorbar(f[1,3], sc, label="Max posterior-to-prior SD ratio", labelsize=18, ticklabelsize=16, tellheight=false, height=Auto(0.8))

    cats = repeat(1:3, inner=(n_feasible*n_models))

    ax = Axis(
        f[2,1],
        yticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6]),
        yticklabelsize=16,
        yscale=log10,
        ylabel="Effective sample size", ylabelsize=18,
        title="Effective sample size for\nimportance sampling methods", titlesize=18,
        # xticks=(1:3, ["Laplace IS", "Standard AMIS", "Robust AMIS"]),
        # xticklabelrotation=π/6,
        xticks=(1:3, ["Laplace\nIS", "Standard\nAMIS", "Robust\nAMIS"]),
        xticklabelsize=18,
    )
    
    for i in 1:3
        rainclouds!(
            fill(i, n_feasible*n_models), all_essvecs[i],
            color=COLORS[i+1], markersize=2,
            plot_boxplots=false, 
            jitter_width=0.2,
            show_median=false, 
            clouds=hist, cloud_width=ess_heights[i], gap=0.0, dodge_gap=0.1, side_nudge=0.125, hist_bins=logrange(1, 1e6, 31)[4:end]
        )
    end

    ax = Axis(
        f[2,2], title="Performance of robust AMIS",
        titlesize=18,
        xlabel="Effective sample size", xlabelsize=18,
        ylabel="Log-evidence errors relative\nto bridge sampling", ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16,
        # limits=((2, 1.2e6), (-6, 6)),
        xscale=log10,
        yscale=symsqrt,
        yticks=(tick_vals, tick_labels),
        xticks=(10 .^ (1:6), [L"10^{%$p}" for p in 1:6])
    )

    sc = scatter!(
        reduce(vcat, eachrow(rAMIS_essmat))[plot_order],
        reduce(vcat, rAMIS_logZvecs .- BS_logZvecs)[plot_order],
        color=sqrt.(var_ratios_vec[plot_order]), colormap=Reverse(:viridis), alpha=0.6, markersize=7,
    )

    Colorbar(f[2,3], sc, label="Max posterior-to-prior SD ratio", labelsize=18, ticklabelsize=16, tellheight=false, height=Auto(0.8))    

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/comparison_insect.png", f, px_per_unit=4);
end





sum(reduce(vcat, MCMC_maxrhat) .< 1.01)
sum(reduce(vcat, MCMC_maxrhat) .< 1.05)
2505 / 2816
maximum(maximum.(MCMC_maxrhat))

hist(map(minimum, MCMC_miness)) # >200
hist(map(maximum, MCMC_maxrhat)) # < 1.05
hist(map(sum, all_times ./ 60))


OUTDIR = joinpath(@__DIR__, "output/data1");
fname = joinpath(OUTDIR, "chains_model1.jld2");
@load fname chn;
rhat(chn)
rhat(chn).nt.rhat