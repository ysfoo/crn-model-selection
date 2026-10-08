### Compare the three SMC distribution paths on the 44 insect data sets: number of SMC iterations,
### TVD of model posterior probabilities from bridge sampling against runtime, and per-iteration diagnostics.
### Set `LOAD_SUMMARIES` to false to recompute summaries from the per-data-set SMC outputs (slow: loads every run).
### Recomputing also overwrites the summary files `output/<run>.jld2` read by `SMC_results.jl`.

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../SMC_functions.jl"));

using Distributions, LogExpFunctions, Statistics, StatsBase
using JLD2, ProgressMeter

std_idxs = 1:3 # ODE parameters with standard prior
ss_idxs = 4:9 # ODE parameters with spike-and-slab prior
n_ss = length(ss_idxs)
μ_slab, μ_spike = 0., -16.;
thres = 0.5 * (μ_spike + μ_slab);

extract_γ(targetinfo) = begin
    hasproperty(targetinfo, :γ) && return targetinfo.γ
    hasproperty(targetinfo, :β) && return targetinfo.β
    return last(targetinfo)
end

psize_str = "4k"
pop_size = 4000
run_strs = ["SMC205", "SMC505", "SMC305"];
path_names = ["Power posteriors", "Alt. geometric path", "Non-geometric path"];
path_ticks = ["Power\nposteriors", "Alt.\ngeometric", "Non-\ngeometric"];
n_paths = length(run_strs)

# Bridge sampling reference
@load joinpath(@__DIR__, "output/logZs_backup.jld2") BS_logZvecs;
pvecs_BS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in BS_logZvecs];

calc_tvd(p1, p2) = 0.5sum(abs, p1 .- p2)

## Summaries of each run

ss_pattern(state) = [state[j] > thres for j in ss_idxs]

# Fraction of particles whose spike/slab pattern changes during the final round of moves of an iteration
# (`info.delta` is the displacement over the final round only).
switch_frac(particles) = mean(
    ss_pattern(p.state) != ss_pattern(p.state .- p.info.delta) for p in particles
)

function summarise_run(fname)
    all_particles, iter, targetinfos, npass_vec, smc_times = load_SMC(fname)
    moved = all_particles[2:end]
    return (;
        pvec = get_pvec(all_particles[end], thres),
        γs = extract_γ.(targetinfos), # includes γ = 0
        smc_times, npass_vec,
        n_nuts = [ps[1].n_nuts for ps in moved],
        n_grads = [sum(p.info.n_grad for p in ps) for ps in moved],
        switch_fracs = switch_frac.(moved),
    )
end

summ_fname = joinpath(@__DIR__, "output/SMC_paths_$(psize_str).jld2");
LOAD_SUMMARIES = true;
if LOAD_SUMMARIES
    @load summ_fname summaries
else
    summaries = Dict{String, Vector{Any}}()
    for run_str in run_strs
        runs = Vector{Any}(undef, n_feasible)
        @showprogress desc=run_str for dir_idx in 1:n_feasible
            fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str)_$(psize_str).jld2")
            runs[dir_idx] = summarise_run(fname)
            @assert runs[dir_idx].γs[end] == 1. "$(run_str) on data $(dir_idx) is incomplete"
        end
        summaries[run_str] = runs

        # Same format as `SMC_results.jl`
        pvecs_SMC = [r.pvec for r in runs]
        hours_SMC = [sum(r.smc_times) / 3600 for r in runs]
        ns_nuts = [sum(r.n_nuts) for r in runs]
        targetinfos_vec = [[(; γ) for γ in r.γs] for r in runs]
        @save joinpath(@__DIR__, "output/$(run_str)_$(psize_str).jld2") pvecs_SMC hours_SMC ns_nuts targetinfos_vec
    end
    @save summ_fname summaries
end

@assert length(summaries[run_strs[1]][1].pvec) == length(pvecs_BS[1])

n_iters = [[length(r.γs) - 1 for r in summaries[s]] for s in run_strs];
hours = [[sum(r.smc_times) / 3600 for r in summaries[s]] for s in run_strs];
# NUTS transitions per particle over the whole run (every particle performs the same number in an iteration)
ns_nuts = [[sum(r.n_nuts) for r in summaries[s]] for s in run_strs];
# Leapfrog steps (gradient evaluations) per particle over the whole run, averaged over the population
n_leapfrogs = [[sum(r.n_grads) / pop_size for r in summaries[s]] for s in run_strs];
tvds = [[calc_tvd(r.pvec, p) for (r, p) in zip(summaries[s], pvecs_BS)] for s in run_strs];
# Bridge sampling probability of models that hold no particles at the end
missing_mass = [[sum(p[r.pvec .== 0]) for (r, p) in zip(summaries[s], pvecs_BS)] for s in run_strs];

for (i, s) in enumerate(run_strs)
    @info path_names[i] s median(n_iters[i]) median(hours[i]) median(tvds[i]) maximum(tvds[i]) median(missing_mass[i]) maximum(missing_mass[i])
end

## Figure

PATH_COLORS = Makie.wong_colors()[[1, 2, 3]];
MARKERS = [:rect, :circle, :diamond];

# Concatenate per-data-set curves, separated by NaN, so that each path is a single `lines!` call
nan_join(vecs) = reduce(vcat, [[v; NaN] for v in vecs])

# Thin baseline of a histogram drawn with `direction=:x` at `offset`, spanning the occupied bins
function hist_baseline!(ax, v, bins, offset; color)
    w = fit(Histogram, v, bins).weights
    lo, hi = findfirst(>(0), w), findlast(>(0), w)
    linesegments!(ax, [Point2f(offset, bins[lo]), Point2f(offset, bins[hi+1])]; color, linewidth=1)
end

begin
    f = Figure(size=(1000, 800))

    # A: number of SMC iterations
    iter_bins = (minimum(minimum.(n_iters)) - 0.5):1:(maximum(maximum.(n_iters)) + 0.5)
    iter_histmaxs = [maximum(fit(Histogram, v, iter_bins).weights) for v in n_iters]
    ax = Axis(
        f[1,1],
        limits=((0.2, 3.15), nothing),
        ylabel="Number of SMC iterations", ylabelsize=18,
        title="Length of SMC runs", titlesize=18,
        xticks=(1:n_paths, path_ticks), xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        hist!(
            ax, n_iters[i], bins=iter_bins,
            scale_to=-0.75 * iter_histmaxs[i] / maximum(iter_histmaxs), offset=i,
            direction=:x, color=PATH_COLORS[i],
        )
        hist_baseline!(ax, n_iters[i], iter_bins, i; color=PATH_COLORS[i])
    end

    # B: TVD from bridge sampling
    tvd_bins = 0:0.02:(ceil(maximum(maximum.(tvds)) / 0.02) * 0.02)
    tvd_histmaxs = [maximum(fit(Histogram, v, tvd_bins).weights) for v in tvds]
    axB = ax = Axis(
        f[1,2],
        limits=((0.6, 3.12), (-0.005, nothing)),
        ylabel="Total variation distance\nfrom bridge sampling", ylabelsize=18,
        title="Accuracy of model posterior", titlesize=18,
        xticks=(1:n_paths, path_ticks), xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        hist!(
            ax, tvds[i], bins=tvd_bins,
            scale_to=-0.80 * tvd_histmaxs[i] / maximum(tvd_histmaxs), offset=i,
            direction=:x, color=PATH_COLORS[i],
        )
        hist_baseline!(ax, tvds[i], tvd_bins, i; color=PATH_COLORS[i])
    end

    # C: TVD against runtime
    hour_ticks = [10, 20, 50, 100, 200]
    ax = Axis(
        f[2,1],
        limits=(nothing, (-0.005, nothing)),
        xscale=log10, xticks=hour_ticks,
        xminorticks=[30, 40, 60, 70, 80, 90], xminorticksvisible=true, xminorticksize=5,
        xlabel="Runtime (hours)", xlabelsize=18,
        ylabel="Total variation distance\nfrom bridge sampling", ylabelsize=18,
        title="Accuracy against runtime", titlesize=18,
        xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        scatter!(
            ax, hours[i], tvds[i],
            color=(PATH_COLORS[i], 0.7), marker=MARKERS[i], markersize=10,
        )
    end

    linkyaxes!(axB, ax)

    # D: mixing between spike/slab patterns
    ax = Axis(
        f[2,2],
        limits=(nothing, (-0.01, nothing)),
        xlabel=L"Path parameter $\gamma$", xlabelsize=18,
        ylabel="Fraction of particles with implied\nmodel changed in final NUTS round", ylabelsize=18,
        title="Frequency of between-model jumps", titlesize=18,
        xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        runs = summaries[run_strs[i]]
        lines!(
            ax, nan_join([r.γs[2:end] for r in runs]), nan_join([r.switch_fracs for r in runs]),
            color=(PATH_COLORS[i], 0.3), linewidth=1,
        )
    end

    Legend(
        f[:, 3],
        [MarkerElement(color=PATH_COLORS[i], marker=MARKERS[i], markersize=12) for i in 1:n_paths],
        ["Power\nposteriors", "Alt. geometric\npath", "Non-geometric\npath"],
        labelsize=18, rowgap=10, tellheight=false,
    )
    colgap!(f.layout, 2, 25)

    for (i, loc) in enumerate([f[1, 1, TopLeft()], f[1, 2, TopLeft()], f[2, 1, TopLeft()], f[2, 2, TopLeft()]])
        Label(loc, string('A' + i - 1),
            fontsize = 24, font = :bold,
            padding = (0, 0, 0, -30), # left, right, bottom, top
            halign = :left, valign = :center,
        )
    end

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/SMC_paths_insect.png", f, px_per_unit=4);
end

# Runtime against number of leapfrog steps, and leapfrog steps against NUTS transitions (appendix)
begin
    f = Figure(size=(800, 385), figure_padding=(16, 28, 16, 2)) # left, right, bottom, top

    leapfrog_tick(x) = x == 0 ? L"0" : x == 100000 ? L"10^5" : L"%$(x ÷ 10000)\times 10^4"
    leapfrog_ticks(xs) = (xs, leapfrog_tick.(xs))
    leapfrog_label = "Number of leapfrog steps\nper particle"

    # A: runtime against leapfrog steps
    ax = Axis(
        f[1,1],
        limits=((0, nothing), (0, nothing)),
        xticks=leapfrog_ticks(0:40000:80000),
        xlabel="Number of leapfrog steps per particle", xlabelsize=18,
        ylabel="Runtime (hours)", ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        scatter!(
            ax, n_leapfrogs[i], hours[i],
            color=(PATH_COLORS[i], 0.7), marker=MARKERS[i], markersize=10,
        )
    end

    # B: leapfrog steps against NUTS transitions
    ax = Axis(
        f[1,2],
        limits=((0, nothing), (0, nothing)),
        yticks=leapfrog_ticks(0:20000:100000),
        xlabel="Number of NUTS transitions per particle", xlabelsize=18,
        ylabel=leapfrog_label, ylabelsize=18,
        xticklabelsize=16, yticklabelsize=16,
    )
    for i in 1:n_paths
        scatter!(
            ax, ns_nuts[i], n_leapfrogs[i],
            color=(PATH_COLORS[i], 0.7), marker=MARKERS[i], markersize=10,
        )
    end

    Legend(
        f[2, :],
        [MarkerElement(color=PATH_COLORS[i], marker=MARKERS[i], markersize=12) for i in 1:n_paths],
        path_names,
        orientation=:horizontal, labelsize=18,
    )

    for (i, loc) in enumerate([f[1, 1, TopLeft()], f[1, 2, TopLeft()]])
        Label(loc, string('A' + i - 1),
            fontsize = 20, font = :bold,
            padding = (0, 0, 4, 0), # left, right, bottom, top
            halign = :left, valign = :bottom,
        )
    end

    display(f)
    save_dir = mkpath(joinpath(@__DIR__, "imgs/"));
    save("$(save_dir)/SMC_paths_nuts_insect.png", f, px_per_unit=4);
end

## Playground

n_iters
median.(n_iters)

# begin
#     f = Figure()
#     ax = Axis(f[1,1])
#     for i in 1:44
#         xs = summaries["SMC305"][i].γs
#         ys = summaries["SMC505"][i].γs
#         if length(xs) < length(ys)
#             xs = [xs; ones(length(ys) - length(xs))]
#         elseif length(xs) > length(ys)
#             ys = [ys; ones(length(xs) - length(ys))]
#         end
#         scatterlines!(xs, ys, alpha=0.2, color=:black)
#     end
#     display(f)
# end

sum(tvds[1] .< 0.1)
sum(tvds[2] .< 0.1)
sum(tvds[3] .< 0.05)

median(tvds[2])
median(tvds[3])

mean(tvds[2])
mean(tvds[3])

std(tvds[2])
std(tvds[3])