# Diagnostic: relationship between step size and ESJD per gradient (and acceptance rate) in a nuts_invmass_move run.
# Uses the last move stored on each particle at each SMC iteration. Also colours by spike/slab pattern to see
# whether the relationship depends on the region (mode) a particle is in.
include(joinpath(@__DIR__, "../SMC_functions.jl"));
using CairoMakie, JLD2, Statistics

dir_idx = 1
run_str = "SMC503_1k"
ss_idxs = 4:9
thres = -8.
OUTDIR = joinpath(get(ENV, "SMC_OUTROOT", joinpath(@__DIR__, "output")), "data$(dir_idx)")
@load joinpath(OUTDIR, "$(run_str).jld2") all_particles targetinfos
imgdir = mkpath(joinpath(@__DIR__, "imgs"))

iters = 2:length(all_particles)   # all_particles[1] is the initial population
γs = [targetinfos[k-1].γ for k in iters]
nslab(p) = count(>(thres), p.state[ss_idxs])

function binned_median(x, y; nbins=12)
    edges = quantile(x, range(0, 1, length=nbins+1))
    xs, ys = Float64[], Float64[]
    for i in 1:nbins
        idx = findall(j -> edges[i] <= x[j] <= edges[i+1], eachindex(x))
        length(idx) < 5 && continue
        push!(xs, median(x[idx])); push!(ys, median(y[idx]))
    end
    return xs, ys
end

ncol = 3; nrow = cld(length(iters), ncol)
figs = Dict()
for (name, ylab, getter) in [
    ("esjd", "log10 ESJD per grad", p -> log10(max(p.info.curr_esjd_per_grad, 1e-12))),
    ("acc", "acceptance rate", p -> p.info.curr_acc_rate),
]
    for (cname, cfun, cmap) in [("acc", p -> p.info.curr_acc_rate, :viridis), ("nslab", nslab, :plasma)]
        name == "acc" && cname == "acc" && continue
        fig = Figure(size=(400ncol, 330nrow))
        for (n, k) in enumerate(iters)
            ps = all_particles[k]
            x = [log10(p.stepsize) for p in ps]; y = getter.(ps)
            ax = Axis(fig[fld(n-1, ncol)+1, mod(n-1, ncol)+1];
                title="iter $(k-1), γ=$(round(γs[n], sigdigits=2)), npass=$(ps[1].n_nuts÷5)",
                xlabel="log10 step size", ylabel=ylab)
            scatter!(ax, x, y; color=cfun.(ps), colormap=cmap, markersize=5)
            xb, yb = binned_median(x, y)
            lines!(ax, xb, yb; color=:red, linewidth=2)
            n == length(iters) && Colorbar(fig[:, ncol+1]; colormap=cmap, limits=extrema(cfun.(ps)),
                label = cname == "acc" ? "acceptance rate" : "# slab dims")
        end
        save(joinpath(imgdir, "$(run_str)_$(name)_vs_stepsize_by_$(cname).png"), fig)
    end
end

# Region dependence: for each iteration, median step size / ESJD of each spike-slab pattern (by # slab dims)
for k in iters[[1, end÷2, end]]
    ps = all_particles[k]
    @info "iter $(k-1)" 
    for m in sort(unique(nslab.(ps)))
        sub = filter(p -> nslab(p) == m, ps)
        length(sub) < 10 && continue
        @info "  nslab=$m" n=length(sub) median_ss=median(p.stepsize for p in sub) median_esjd=median(p.info.curr_esjd_per_grad for p in sub) median_acc=median(p.info.curr_acc_rate for p in sub)
    end
end
# Spread of ESJD at fixed step size: correlation of log step with log ESJD per iteration
for (n, k) in enumerate(iters)
    ps = all_particles[k]
    x = [log(p.stepsize) for p in ps]; y = [log(max(p.info.curr_esjd_per_grad, 1e-12)) for p in ps]
    @info "iter $(k-1)" γ=γs[n] cor_logss_logesjd=cor(x, y) frac_low_acc=mean(p.info.curr_acc_rate < 0.5 for p in ps)
end
