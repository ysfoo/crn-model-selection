include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../plot_helpers.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, PEtab, Random
using JLD2, ProgressMeter, Suppressor
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using AdvancedHMC, Bijectors, BridgeSampling, LogDensityProblems, LogDensityProblemsAD, MCMCChains, PSIS, Turing

# Posterior plots

# dir_idx = 3
dir_idx = 25;
genmodel_idx = feasible_idxs[dir_idx];
OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");

param_labels = [
    L"\lambda_{EL}", L"\lambda_{LA}", L"\rho", 
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A", L"\sigma"
];
sym2label = Dict(zip(Symbol.(parameters(models[end])), param_labels))

@nowarn_load "$OUTDIR/MAP.jld2" model_fits;
@load "$OUTDIR/MAP_hess.jld2" MAP_hessians;

for model_idx in [64]
    d = nparams[model_idx]
    ps = parameters(models[model_idx])
    
    mcmc_fname = joinpath(OUTDIR, "chains7000_model$(model_idx).jld2")
    @nowarn_load mcmc_fname chn
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
    samples = reshape(trace, d, :);
    X = samples[:,1:1:end];

    MAP = model_fits[model_idx].xmin
    hess = MAP_hessians[model_idx]
    Σ = inv(PDMat(hermitianpart!(hess)))

    extremas = extrema.(eachrow(samples[:,1:1:end]))
    ax_limits = extremas

    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        [MAP], [Σ],
        # title="Posterior samples under model $model_idx\nfor data generated from model $genmodel_idx",
        title=L"\textbf{Posterior samples under saturated model for data generated with death mechanisms }\delta_E,\, \delta_L,\, \delta_A,\, \kappa_L", 
        titlesize=20,
        figsize=(120*d+40, 120*d+40), skip_upper=true,
        scatter_kwargs=(color=(:grey30, 0.01), markersize=4),
        ellipse_kwargs=(color=Makie.wong_colors()[3],),
        hist_kwargs=(color=:grey,),
        axis_kwargs=(aspect=1,), 
        hist_axis_kwargs=(aspect=1, yscale=log10,),
        bins_vec=[range(a, b, 41) for (a, b) in extremas]
    ); 

    idx = 0
    for (i1, p1) in enumerate(ps) # which row
        for (i2, p2) in enumerate(ps) # which column
            if i1 < i2
                continue
            end
            idx += 1
            ax = f.content[idx]
            if i1 == i2
                autolimits!(ax)
                fl = ax.finallimits[]
                i = i1
                dist = i1 == d ? Normal(-1, 1) : Normal(0, 2)
                xs = dist.μ-5dist.σ:0.05:dist.μ+5dist.σ
                lines!(ax, xs, pdf.(Ref(dist), xs), color=Makie.wong_colors()[1])
                ylow, yheight = fl.origin[2], fl.widths[2]
                new_ylow = ylow*((ylow+yheight)/ylow)^(1/21)
                limits!(ax, ax_limits[i2], (new_ylow, fl.origin[2]+fl.widths[2]))
            else
                limits!(ax, ax_limits[i2], ax_limits[i1])
            end
            if i1 == i2
                ax.yaxisposition = :right
                ax.yticklabelsize = 14
                ax.yticklabelpad = 0.5
                ax.yticksvisible = true
                ax.yticklabelsvisible = true
            elseif i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = L"\log_{10} %$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.yticklabelsize = 14
                ax.yticksvisible = true
                ax.yticklabelsvisible = true                
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticks = WilkinsonTicks(6; k_min = 3, k_max=6)
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 14
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    colgap!(f.layout, -8)

    Legend(
        f[:,d], 
        [
            LineElement(color = Makie.wong_colors()[1], ),
            PolyElement(color = :grey, strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]),
            MarkerElement(color = :grey30, marker=:circle, markersize=8), 
            LineElement(color = Makie.wong_colors()[3], ),
        ],
        ["Prior\ndensity", "Posterior\ndensity", "Posterior\nsamples", "Laplace\napprox."],
        labelsize=18, rowgap=10, tellheight=false, tellwidth=false, halign=:center, valign=:center
    )

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/post_logparams/"))
    mkpath(save_dir)
    save("$(save_dir)/data$(dir_idx)_model$(model_idx).png", f, px_per_unit=4)
end

# Old code

@showprogress for model_idx in 1:2
    d = nparams[model_idx]
    ps = parameters(models[model_idx])
    
    mcmc_fname = joinpath(OUTDIR, "chains_model$(model_idx).jld2")
    @nowarn_load mcmc_fname chn
    trace = permutedims(chn.value[:,1:d,:].data, [2, 1, 3]);
    X = reshape(trace, d, :);

    MAP = model_fits[model_idx].xmin
    hess = MAP_hessians[model_idx]
    Σ = inv(PDMat(hermitianpart!(hess)))

    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        [MAP], [Σ],
        # title="Posterior samples under model $model_idx for data generated from model $genmodel_idx",
        figsize=(120*d+180, 120*d+40),
        scatter_kwargs=(color=(:grey, 0.01), markersize=4),
        ellipse_kwargs=(color=Makie.wong_colors()[2],),
        hist_kwargs=(color=:grey,)
    ); 

    for (i1, p1) in enumerate(ps)
        for (i2, p2) in enumerate(ps)
            idx = (i2-1)*d + i1
            ax = f.content[idx]
            if i1 == i2
                autolimits!(ax)
                ax_limits = ax.finallimits[]
                xs = -10:0.01:10
                dist = i1 == d ? Normal(-1, 1) : Normal(0, 2)
                lines!(ax, xs, pdf.(Ref(dist), xs), color=Makie.wong_colors()[1])
                limits!(ax, ax_limits)
                xlims!(ax, (ax_limits.origin[1], ax_limits.origin[1]+ax_limits.widths[1]))
            end
            if i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = (i1 == i2) ? "" : L"\log_{10} %$(sym2label[Symbol(p1)])"
                # ax.ylabel = (i1 == i2) ? "" : L"%$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticklabelsize = 12
                ax.yticksvisible = i1 != i2
                ax.yticklabelsvisible = i1 != i2
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [1, d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                # ax.xlabel = L"%$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    Legend(
        f[:,d+1], 
        [
            LineElement(color = Makie.wong_colors()[1], ),
            PolyElement(color = :grey, strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]),
            MarkerElement(color = :grey, marker=:circle, markersize=8), 
            LineElement(color = Makie.wong_colors()[2], ),
        ],
        ["Log prior\ndensity", "Log posterior\ndensity", "Posterior\nsamples", "Laplace\napprox."],
        labelsize=15, rowgap=10,        
    )
    
    save_dir = mkpath(joinpath(@__DIR__, "imgs/data$(dir_idx)_logparams/"))
    mkpath(save_dir)
    save("$(save_dir)/model$(model_idx).png", f)

    ## Now with prior and LA
    f = plot_pairs(
        # eachcol(exp10.(X)),
        eachcol(X),
        [MAP], [Σ],
        [[i == d ? -1 : 0 for i in 1:d]],
        [diagm([i == d ? 1 : 4 for i in 1:d])],
        # title="Posterior\nsamples under model $model_idx for data generated from model $genmodel_idx",
        figsize=(120*d+180, 120*d+40),
        scatter_kwargs=(color=(:grey, 0.2), markersize=5),
        ellipse_kwargs=(color=:black,),
        hist_kwargs=(fillto=1e-4,),
    ); 

    for (i1, p1) in enumerate(ps)
        for (i2, p2) in enumerate(ps)
            idx = (i2-1)*d + i1
            ax = f.content[idx]
            if i1 == i2
                i = i1
                xs = (i == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                limits!(ax, extrema(xs), (2e-4, 1))
                lines!(ax, xs, pdf.(Ref(i == d ? Normal(-1, 1) : Normal(0, 2)), xs), color=:black)
            else
                xs = (i2 == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                ys = (i1 == d) ? (-4.5:0.05:2.5) : (-7:0.05:7)
                limits!(ax, extrema(xs), extrema(ys))
            end
            if i2 ∈ [1, d]
                ax.yaxisposition = i2 == 1 ? :left : :right
                ax.ylabel = (i1 == i2) ? "" : L"\log_{10} %$(sym2label[Symbol(p1)])"
                # ax.ylabel = (i1 == i2) ? "" : L"%$(sym2label[Symbol(p1)])"
                ax.ylabelsize = 18
                ax.yticklabelsize = 12
                ax.yticksvisible = i1 != i2
                ax.yticklabelsvisible = i1 != i2
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i1 ∈ [1, d]
                ax.xaxisposition = i1 == 1 ? :top : :bottom
                ax.xlabel = L"\log_{10} %$(sym2label[Symbol(p2)])"
                # ax.xlabel = L"%$(sym2label[Symbol(p2)])"
                ax.xlabelsize = 18
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end

    Legend(
        f[:,d+1], 
        [
            MarkerElement(color = :grey, marker=:circle, markersize=8), 
            PolyElement(color=Makie.wong_colors()[1], strokewidth = 0, points = Point2f[(0, 0.25), (1, 0.25), (1, 0.75), (0, 0.75)]),
            LineElement(color=:black, )
        ],
        ["Posterior\nsamples", "Log posterior\ndensity", "Log prior\ndensity"],
        labelsize=18, rowgap=10,
    )
    
    save_dir = mkpath(joinpath(@__DIR__, "imgs/data$(dir_idx)_logparams_prior/"))
    mkpath(save_dir)
    save("$(save_dir)/model$(model_idx).png", f, px_per_unit=4)
end