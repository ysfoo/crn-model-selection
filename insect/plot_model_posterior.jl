### Set `LOAD_VAR_RATIOS` to false if `plot_comparison.jl` has not been run (or run that script first).

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../gaussian_mixtures.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random, SymbolicIndexingInterface
using JLD2, ProgressMeter, Suppressor
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using Pathfinder, PSIS, StableRNGs, BridgeSampling
using AdvancedHMC, Bijectors, LinearAlgebra, LogDensityProblems, LogDensityProblemsAD, MCMCChains, Turing

@load joinpath(@__DIR__, "data.jld2") all_data;
@load joinpath(@__DIR__, "params.jld2") tuned_params;

# Posterior variance
vrats_fname = joinpath(@__DIR__, "output/var_ratios.jld2")
LOAD_VAR_RATIOS = true
if LOAD_VAR_RATIOS
    @load vrats_fname var_ratios
else
    all_postvars = Matrix{Vector{Float64}}(undef, n_feasible, n_models);
    all_priorvars = Matrix{Vector{Float64}}(undef, n_feasible, n_models);
    @showprogress for dir_idx in 1:n_feasible
        OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

        for model_idx in 1:n_models
            d = nparams[model_idx]
            chains_fname = "$OUTDIR/chains_model$model_idx.jld2"
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

            fname = "$OUTDIR/chains_model$model_idx.jld2"
            @load fname chn
            chains_time += MCMCChains.compute_duration(chn) / 60

            fname = "$OUTDIR/BS_model$model_idx.jld2"
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

function calc_tvd(logZvec1, logZvec2)
    logp1 = exp.(logZvec1 .- logsumexp(logZvec1))
    logp2 = exp.(logZvec2 .- logsumexp(logZvec2))
    return 0.5sum(abs, logp1.-logp2)
end

ents = [
    begin
        logps = logZvec .- logsumexp(logZvec)
        (logps .+ log.(.-logps)) |> logsumexp |> exp
    end for logZvec in BS_logZvecs
];
tvds = calc_tvd.(rAMIS_logZvecs, BS_logZvecs);

[sortperm(ents) sort(ents) tvds[sortperm(ents)]]

# dir_idx = 12;
# f = Figure();
# ax = Axis(f[1,1]);
# for logZvecs in [LIS_logZvecs, BS_logZvecs, orig_logZvecs, rAMIS_logZvecs]
#     logZvec = logZvecs[dir_idx];
#     scatter!(exp.(logZvec .- logsumexp(logZvec)), alpha=0.5)
# end
# display(f)
# models[feasible_idxs[dir_idx]]
# feasible_idxs[dir_idx]

n_plot = 10;
ps = parameters(models[end]);

# for dir_idx in 1:44
# println("Data $(dir_idx)")
dir_idx = 25;
OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
genmodel_idx = feasible_idxs[dir_idx];
parameters(models[genmodel_idx])
data = all_data[genmodel_idx];
tuned_params[genmodel_idx]
ents[dir_idx], tvds[dir_idx]

logZvec = BS_logZvecs[dir_idx];
pvec = exp.(logZvec .- logsumexp(logZvec));
to_plot = sortperm(pvec, rev=true)[1:n_plot];
marg_ps = [sum(pval for (i, pval) in enumerate(pvec) if any(parameters(models[i]) .=== p)) for p in ps[4:9]]
sqrt.(var_ratios[dir_idx, to_plot])

param_labels = [
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A"
];

COLORS = [:grey60; Makie.wong_colors()[[1, 3, 4, 2]]];
method_names = ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"];

begin
    f = Figure(size=(1080, 540))

    ax = Axis(
        f[1, 2], 
        title="Model posterior summary", titlesize=20,
        xlabel="Death mechanism", xlabelsize=18,
        ylabel="Inclusion posterior probability", ylabelsize=18,
        limits = ((0.5, 6.5), (0, 1.04)),
        xticks = (1:6, param_labels), xticklabelsize=20,
        yticks = 0:0.2:1, yticklabelsize=16
    )

    for (i, logZvecs) in enumerate([BIC_logZvecs, LIS_logZvecs, orig_logZvecs, rAMIS_logZvecs, BS_logZvecs])
        logZvec = logZvecs[dir_idx];
        pvec = exp.(logZvec .- logsumexp(logZvec));
        marg_ps = [sum(pval for (i, pval) in enumerate(pvec) if any(parameters(models[i]) .=== p)) for p in ps[4:9]]
        barplot!(
            (1:6) .+ 0.16*(i-3), marg_ps, 
            color=COLORS[i], width=0.16, gap=0
        )
    end

    ax1 = Axis(
        f[1, 1], title="Top $n_plot models ranked by bridge sampling", titlesize=20,
        limits = ((0.5, n_plot + 0.5), (0, nothing)),
        xlabel="Models ranked by bridge sampling", xlabelsize=18, xlabelpadding=10,
        ylabel="Model posterior probability", ylabelsize=18,
        xticks=1:n_plot, xticklabelsvisible=false, yticklabelsize=16,
    )

    for (i, logZvecs) in enumerate([BIC_logZvecs, LIS_logZvecs, orig_logZvecs, rAMIS_logZvecs, BS_logZvecs])
        logZvec = logZvecs[dir_idx];
        pvec = exp.(logZvec .- logsumexp(logZvec));
        barplot!(
            (1:n_plot) .+ 0.16*(i-3), pvec[to_plot], 
            color=COLORS[i], width=0.16, gap=0
        )
    end

    ax2 = Axis(
        f[2, 1],
        yreversed=true, alignmode=Mixed(top = -5),
        xticks=1:n_plot, xticklabelsvisible=false, xaxisposition=:top,
        yticks=(1:6, param_labels), yticklabelsize=20,
        ylabel="Includes mechanism", ylabelsize=18,
        limits = ((0.5, n_plot + 0.5), (0.5, 6.5)),
    )
    linkxaxes!(ax1, ax2)
    # heatmap!(
    #     [any(parameters(models[i]) .=== p) for i in to_plot, p in ps[4:9]],
    #     colormap=Reverse(:grays)
    # )
    sc_points = Tuple{Int,Int}[]
    for y in 1:6
        lines!([-0.5, n_plot + 0.5], [y - 0.5, y - 0.5], color=:black, linewidth=1)
        for x in 1:n_plot
            any(parameters(models[to_plot[x]]) .=== ps[3+y]) && push!(sc_points, (x, y))
        end
    end
    scatter!(first.(sc_points), last.(sc_points), marker = '✓', markersize = 24, color=:black)

    Legend(
        f[2, 2],
        [PolyElement(color=COLORS[c]) for c in 1:5],
        method_names,
        halign=:left,
        labelsize=18, tellheight=false, tellwidth=false,
    )

    for i in 1:2
        label = ["A", "B"][i]
        loc = [f[1, 1, TopLeft()], f[1, 2, TopLeft()]][i]
        rpad = [60, 60][i]
        Label(loc, label,
            fontsize = 24, font = :bold,
            padding = (0, rpad, 0, 0), # left, right, bottom, top
            halign = :right, valign = :center,
        )
    end

    colsize!(f.layout, 2, Auto(0.6))
    rowsize!(f.layout, 2, Auto(0.55))
    rowgap!(f.layout, 1, 0) 

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/model_posts/"));
    save("$(save_dir)/data$(dir_idx).png", f, px_per_unit=4);
end

# end


## Single-model predictions

# dir_idx = 25;
# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
# genmodel_idx = feasible_idxs[dir_idx];
# parameters(models[genmodel_idx])
# data = all_data[genmodel_idx];
# tuned_params[genmodel_idx]
# ents[dir_idx], tvds[dir_idx]

# n_plot = 10;
# ps = parameters(models[end]);
# pred_dim = 3; # adult population

# logZvec = BS_logZvecs[dir_idx];
# pvec = exp.(logZvec .- logsumexp(logZvec));
# to_plot = sortperm(pvec, rev=true)[1:n_plot];
# marg_ps = [sum(pval for (i, pval) in enumerate(pvec) if any(parameters(models[i]) .=== p)) for p in ps[4:9]]

# petab_probs = PEtabODEProblem.(create_petab_model.(models, Ref(data), Ref(u0)); odesolver=ODESolver(Rodas5P(), verbose=false));
# t_span = (0, 20);
# t_pred = range(0, 20, 81);
# n_preds = 10^4;

# oprob_vec = [get_odeproblem(get_x(petab_prob), petab_prob)[1] for petab_prob in petab_probs];
# param_idxs_vec = [map((x)->parameter_index(oprob, x).idx, parameters(model)) for (oprob, model) in zip(oprob_vec, models)];

# param_labels = [
#     L"\delta_E", L"\delta_L", L"\delta_A", 
#     L"\kappa_E", L"\kappa_L", L"\kappa_A"
# ];

# COLORS = Makie.wong_colors()[[5, 1, 3, 4, 2]];
# method_names = ["BIC", "Laplace IS", "Standard AMIS", "Robust AMIS", "Bridge sampling"];

# line_alpha = 0.9; band_alpha = 0.4;

# begin
#     f = Figure(size=(1080, 1200))

#     for ax_i1 in 1:5
#         for ax_i2 in 1:2
#             ax_i = (ax_i1-1)*2 + ax_i2
#             model_idx = to_plot[ax_i]

#             BS_all_params = Vector{Float64}[];
#             mcmc_fname = joinpath(OUTDIR, "chains_model$(model_idx).jld2");
#             d = nparams[model_idx]
#             @load mcmc_fname chn;
#             trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
#             BS_samples = reshape(trace, d, :);
#             append!(BS_all_params, eachcol(BS_samples))

#             BS_single_preds = [
#                 begin
#                     θ = BS_all_params[i]
#                     oprob = oprob_vec[model_idx]
#                     param_idxs = param_idxs_vec[model_idx]
#                     oprob.p[param_idxs] .= exp10.(θ)
#                     sol = solve(oprob, Rodas5P(); tspan=t_span, saveat=t_pred);
#                     getindex.(sol.u, pred_dim)
#                 end for i in 1:15000
#             ];
#             BS_single_predmat = stack(BS_single_preds);
#             BS_lo_preds = quantile.(eachrow(BS_single_predmat), 0.05);
#             BS_med_preds = quantile.(eachrow(BS_single_predmat), 0.5);
#             BS_hi_preds = quantile.(eachrow(BS_single_predmat), 0.95);

#             loc = f[ax_i1,ax_i2]
#             ax = Axis(
#                 loc,
#                 xlabel = ax_i1==5 ? "Time (a.u.)" : "", xlabelsize=18, 
#                 xticklabelsvisible = ax_i1==5,
#                 yticklabelsvisible = ax_i2==1,
#                 limits = ((-0.5, 20.5), (2.1, 5.1)),
#                 xticklabelsize=16,  yticklabelsize=16,
#             )
#             scatter!(data.t, getproperty(data, propertynames(data)[pred_dim]), color=:grey10, markersize=8)

#             band!(t_pred, BS_lo_preds, BS_hi_preds, color=COLORS[end], alpha=band_alpha)
#             lines!(
#                 t_pred, BS_lo_preds,
#                 color=(COLORS[end], line_alpha), linestyle=Linestyle([0, 0, 3, 6]), linewidth=3
#             )
#             lines!(
#                 t_pred, BS_hi_preds,
#                 color=(COLORS[end], line_alpha), linestyle=Linestyle([0, 0, 3, 6]), linewidth=3
#             )
#             lines!(t_pred, BS_med_preds, color=COLORS[end], alpha=line_alpha, linewidth=3)
#         end
#     end

#     Label(f[0,1:2], L"$\textbf{Single‐model predictions for data generated with death mechanisms }\delta_E,\, \delta_L,\, \delta_A,\, \kappa_L$\\$\textbf{under top 10 models ranked by bridge sampling}$", fontsize=20)
#     pop_name = ["Egg", "Larva", "Adult"][pred_dim]
#     Label(f[1:5,0], "$(pop_name) population size (a.u.)", fontsize=18, rotation = pi/2)

#     Legend(
#         f[1:5,3],
#         [
#             MarkerElement(color=:grey10, marker=:circle, markersize=10),
#             LineElement(color=(COLORS[end], line_alpha), linewidth=2), 
#             PolyElement(color=(COLORS[end], band_alpha), strokewidth=2, strokecolor=(COLORS[end], line_alpha), linestyle=:dash)
#         ],
#         ["Data"; "Posterior median"; "90% credible interval"],
#         halign=:center, valign=:center,
#         labelsize=18, tellheight=false, tellwidth=false,
#     )

#     colsize!(f.layout, 3, Auto(0.5))

#     display(f)
#     save_dir = mkpath(joinpath(@__DIR__, "imgs/BS_preds/"));
#     save("$(save_dir)/data$(dir_idx).png", f, px_per_unit=4);
# end

## Playground

# begin
#     f = Figure()
#     Label(f[1, 1, Top()], L"\textbf{Reconstructed }\mathbf{X_1}")
#     display(f)
# end

# dir_idx = 43;
# model_idx = genmodel_idx = feasible_idxs[dir_idx];
# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

# fname = joinpath(OUTDIR, "chains_model$(genmodel_idx).jld2");
# @load fname chn;
# describe(chn)
# [exp10.(mean(chn).nt.mean) tuned_params[genmodel_idx]]

# d = length(parameters(models[genmodel_idx]))
# trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
# X = reshape(trace, d, :);

# @nowarn_load "$OUTDIR/MAP.jld2" model_fits;
# @load "$OUTDIR/MAP_hess.jld2" MAP_hessians;
# MAP_est = model_fits[model_idx].xmin;
# hess = MAP_hessians[model_idx];
# Σ = inv(PDMat(hermitianpart!(hess)));

# f = plot_pairs(
#     # eachcol(exp10.(X)),
#     eachcol(X),
#     [MAP_est], [Σ],
#     # title="Posterior\nsamples under model $model_idx for data generated from model $genmodel_idx",
#     figsize=(120*d+180, 120*d+40),
#     scatter_kwargs=(color=(:grey, 0.01), markersize=4),
#     ellipse_kwargs=(color=Makie.wong_colors()[2],),
#     hist_kwargs=(color=:grey,)
# ); 
# display(current_figure())

# dir_idx = 12
# OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

# length(BS_logZvecs)
# BS_logZvecs[dir_idx][63]

# model_idx = 63

# fname = "$OUTDIR/robust_AMIS_model$model_idx.jld2"
# @load fname timed_res;
# timed_res.time
# timed_res.value.unweighted_samples;

# fname = "$OUTDIR/orig_AMIS_model$model_idx.jld2"
# @load fname timed_res;
# timed_res.time
# timed_res.value.unweighted_samples;

# fname = "$OUTDIR/laplace_IS_model$model_idx.jld2"
# @load fname timed_res;
# timed_res.time
# timed_res.value.unweighted_samples;

# sAMIS_times = Float64[];
# @showprogress for dir_idx in 1:n_feasible
#     sAMIS_time = 0.
#     OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
#     for model_idx in 1:n_models
#         fname = "$OUTDIR/orig_AMIS_model$model_idx.jld2"
#         @load fname timed_res
#         sAMIS_time += timed_res.time / 60
#     end
#     push!(sAMIS_times, sAMIS_time)
# end
# summarystats(rAMIS_times)

# rAMIS_times = Float64[];
# @showprogress for dir_idx in 1:n_feasible
#     rAMIS_time = 0.
#     OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
#     for model_idx in 1:n_models
#         fname = "$OUTDIR/robust_AMIS_model$model_idx.jld2"
#         @load fname timed_res
#         rAMIS_time += timed_res.time / 60
#     end
#     push!(rAMIS_times, rAMIS_time)
# end
# summarystats(rAMIS_times)