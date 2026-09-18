### Plotting helpers

using CairoMakie, LaTeXStrings
using Accessors, DataPipes
using LinearAlgebra
set_theme!(theme_latexfonts());
update_theme!(
	Axis=(;
		xgridvisible=false, ygridvisible=false,
	),
)

function add_ellipse!(ax, meanvec, covmat, i1, i2; kwargs...)
    thres = quantile(Chisq(length(meanvec)), 0.95)
    eigenvals, eigenvecs = eigen(covmat[[i1,i2],[i1,i2]])
    semi_axes = sqrt.(eigenvals * thres)
    θ = atan(eigenvecs[2, 1], eigenvecs[1, 1])
    tgrid = range(0, 2π, length=200)
    ellipse_unit = hcat(cos.(tgrid), sin.(tgrid))
    ellipse_scaled = ellipse_unit * diagm(semi_axes)
    rotation_matrix = [cos(θ) -sin(θ);
                        sin(θ)  cos(θ)]
    ellipse_rotated = ellipse_scaled * rotation_matrix'
    ellipse = ellipse_rotated .+ [meanvec[i1] meanvec[i2]]
    lines!(ax, ellipse; kwargs...)
end

function plot_pairs(
    states, μs=[], Σs=[]; 
    title=nothing, titlesize=16,
    figsize=nothing, layout=nothing, skip_upper=false,
    axis_kwargs=(;), hist_axis_kwargs=(;yscale=log10),
    scatter_kwargs=(;markersize=5, alpha=0.5), 
    hexbin_kwargs=nothing,
    ellipse_kwargs=(;),
    hist_kwargs=(;), bins_vec=nothing
)
    n_dim = length(states[1])
    δs = std.(eachrow(stack(states))) / 3
    if isnothing(figsize)
        figsize = (120*n_dim, 120*n_dim)
    end
    if isnothing(layout)
        f = Figure(size=figsize)
    else
        f = layout
    end
    for i2 in 1:n_dim
        for i1 in 1:n_dim
            if skip_upper && i1 > i2
                continue
            end
            if i1 == i2
                ax = Axis(f[i2,i1]; hist_axis_kwargs...)
                δ = δs[i1]
                v = getindex.(states, i1)
                bins = bins_vec === nothing ? (fld(minimum(v), δ):cld(maximum(v), δ)) .* δ : bins_vec[i1]
                hist!(v; bins=bins, normalization=:pdf, hist_kwargs...)
            else
                ax = Axis(f[i2,i1]; axis_kwargs...)
                if hexbin_kwargs === nothing
                    scatter!(
                        getindex.(states, i1), getindex.(states, i2); scatter_kwargs...
                    )
                else
                    hexbin!(
                        getindex.(states, i1), getindex.(states, i2), cellsize=(0.2*sqrt(3)*δs[i1], 0.4*δs[i2]); hexbin_kwargs...
                    )
                end
                autolimits!(ax)
                ax_limits = ax.finallimits[]
                for (μ, Σ) in zip(μs, Σs)
                    add_ellipse!(ax, μ, Σ, i1, i2; ellipse_kwargs...)
                end
                limits!(ax, ax_limits)
            end

            if i1 ∈ [1, n_dim]
                ax.yaxisposition = i1 == 1 ? :left : :right
                ax.yticklabelsize = 12
                ax.yticksvisible = ax.yticklabelsvisible = i1 == 1 || !skip_upper
            else
                ax.yticksvisible = false
                ax.yticklabelsvisible = false
            end
            
            if i2 ∈ [1, n_dim]
                ax.xaxisposition = i2 == 1 ? :top : :bottom
                ax.xticklabelrotation = π/4
                ax.xticklabelsize = 12
                ax.xticksvisible = ax.xticklabelsvisible = i2 == n_dim || !skip_upper
            else
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
            end
        end
    end
    if !isnothing(title) && length(title) > 0
        Label(f[0,:], title, font=:bold, fontsize=titlesize)
    end
    return f
end


function add_boxplots!(xs, vecs, tf=identity; kwargs...)
    for (x, vals) in zip(xs, vecs)
        boxplot!(fill(x, length(vals)), vals .|> tf; markersize=6, kwargs...)
    end
end


# Make square root scale work for negative values
function symsqrt(x)
	sign(x)*sqrt(abs(x))
end

function Makie.inverse_transform(::typeof(symsqrt))
    x -> sign(x)*x*x
end

Makie.defaultlimits(::typeof(symsqrt)) = (0.0, 1.0)

Makie.defined_interval(::typeof(symsqrt)) = Makie.OpenInterval(-Inf, Inf)

function get_pos_sqrt_ticks(maxval)
	all_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.];
	upper_filter = all_ticks .<= maxval
	lower_filter = all_ticks .>= 0.02*maxval
	return [0.0; all_ticks[upper_filter .& lower_filter]]
end


## Source: https://github.com/JuliaAPlavin/MakieExtra.jl/blob/000000000f9a024e4769eefea8653e1b48879ba1/src/ticks.jl

@kwdef struct BaseMulTicks
    subs = nothing
    base = 10.
    k_min::Int = 7
end
BaseMulTicks(subs; kwargs...) = BaseMulTicks(; subs, kwargs...)

function Makie.get_tickvalues(t::BaseMulTicks, vmin, vmax)
    vmin < vmax || return []
    vmin < 0 && vmax ≤ 0 && return .-Makie.get_tickvalues(t, -vmax, -vmin)
    @assert vmin ≥ 0 && vmax ≥ 0
    if !isnothing(t.subs)
        @p [
            mul * t.base^pow
            for pow in floor(Int, log(t.base, vmin) - 0.1):ceil(Int, log(t.base, vmax) + 0.1)
            for mul in t.subs
        ] filter!(∈(vmin..vmax)) map(round(_, sigdigits=4)) map(isinteger(_) ? Int(_) : _)
    else
        for subs in [[1], [1,3], [1,2,5], [1,2,3,5], 1:9]
            ticks = Makie.get_tickvalues((@set t.subs = subs), vmin, vmax)
            length(ticks) ≥ t.k_min && return ticks
        end
        return Makie.get_tickvalues(WilkinsonTicks(5), vmin, vmax)
    end
end

Makie.get_minor_tickvalues(t::BaseMulTicks, scale, tickvals, vmin, vmax) = Makie.get_tickvalues(t, scale, vmin, vmax)