# Fit two-slope Fisher maps to stored SMC populations and plot them (exploratory; does not run or change the sampler).
#
# Usage (from insect/):
#   julia --project=.. fit_fisher_maps.jl <data idx> <run name> [iterations...] [--fit=trim|full|both] [--zeta_min=0.2]
#                                         [--variant=SMCnnn]
# e.g. julia --project=.. fit_fisher_maps.jl 1 SMC505_4k
#      julia --project=.. fit_fisher_maps.jl 1 SMC305_4k 22 25 28 30 35 --fit=both
#
# For each selected SMC iteration i, the stored population (equal weights, targeting π_(γ_i)) is used to fit, for every
# coordinate k independently, the map of the appendix "Fisher divergence of a transformed target",
#   F(u) = θ* + ζ [ s_high softplus(u/ζ) - s_low softplus(-u/ζ) ],
# by minimising D_k = Var_n(F'(u_n) α_{n,k} + (log F')'(u_n) + u_n), u_n = F^{-1}(θ_{n,k}). Scores α are recomputed with the
# run's own `prior_path` and `ldp_builder`, which are taken from the variant script `<run_str>.jl` itself (evaluated
# without its `run_SMC`/`resume_SMC` call), so no path formula is re-derived here.
#
# `--fit=trim` (default) fits on the population without the 1% of particles with the lowest log target (as
# `compute_invmass` does); `--fit=full` uses all particles; `--fit=both` does both.
#
# Outputs: output/fisher_maps/data<d>/<run name>/ (iteration_table.csv, summary.csv, iter<i>[_<fit>].png).

# ---------------------------------------------------------------------------------------------------------------------
# Command line. ARGS[1] is the data index read by SMC.jl.
fm_flags = Dict(begin
        kv = split(a[3:end], "=")
        kv[1] => (length(kv) > 1 ? kv[2] : "true")
    end for a in ARGS if startswith(a, "--"))
fm_pos = [a for a in ARGS if !startswith(a, "--")]
length(fm_pos) >= 2 || error("usage: fit_fisher_maps.jl <data idx> <run name> [iterations...] [--fit=trim|full|both]")
fm_run_name = fm_pos[2]
fm_iters_override = parse.(Int, fm_pos[3:end])
fm_fit_pops = Dict("trim" => ["trim"], "full" => ["full"], "both" => ["trim", "full"])[get(fm_flags, "fit", "trim")]
const ZETA_MIN = parse(Float64, get(fm_flags, "zeta_min", "0.2"))
empty!(ARGS); push!(ARGS, fm_pos[1])

include(joinpath(@__DIR__, "SMC.jl"))   # model, data, make_ldp, ss_idxs, thres, load_SMC, ...

# Run's own path: evaluate its variant script, minus `include(... SMC.jl)` and the run/resume call. Statements ending in
# `;` are parsed as nested `:toplevel` blocks, so the filter recurses into them. Running the variant's `run_SMC` would
# overwrite the stored run (`fname`), so refuse to evaluate if any call to it survives the filter.
# `--variant=SMCnnn` takes the path from another variant with the same path (e.g. SMC505 for SMC501_4k, whose script is
# gone); the log target check in `scores` confirms it reproduces the stored values.
fm_run_str = get(fm_flags, "variant", split(fm_run_name, "_")[1])
fm_variant = joinpath(@__DIR__, "$(fm_run_str).jl")
let ex = Meta.parseall(read(fm_variant, String); filename=fm_variant)
    is_skipped(e) = e isa Expr && e.head == :call && e.args[1] in (:include, :run_SMC, :resume_SMC)
    function strip_calls!(e)
        e isa Expr && e.head == :toplevel || return e
        filter!(x -> !is_skipped(x), e.args)
        foreach(strip_calls!, e.args)
        return e
    end
    calls_run(e) = e isa Expr && ((e.head == :call && e.args[1] in (:run_SMC, :resume_SMC)) || any(calls_run, e.args))
    strip_calls!(ex)
    calls_run(ex) && error("$(fm_variant) still calls run_SMC/resume_SMC after filtering; not evaluating it")
    Base.eval(Main, ex)
end
const FM_PATH_NAMES = Dict('2' => "final prior (likelihood tempered only)", '3' => "moment-interpolated mixture",
                           '5' => "geometric", '4' => "Fisher-Rao (FRPath)")
fm_path = get(FM_PATH_NAMES, fm_run_str[4], "unknown")

fm_in = joinpath(@__DIR__, "output/data$(dir_idx)/$(fm_run_name).jld2")
fm_out = joinpath(@__DIR__, "output/fisher_maps/data$(dir_idx)/$(fm_run_name)")
mkpath(fm_out)
@info "Fisher maps" fm_run_name fm_path fm_in fm_out fm_fit_pops ZETA_MIN

all_particles, n_iter, targetinfos, _, _ = load_SMC(fm_in)
γs = [ti.γ for ti in targetinfos]               # γs[i+1] is γ_i; all_particles[i+1] is the population of iteration i
n_dim = length(all_particles[1][1].state)
const COORD_NAMES = ["λ_EL", "λ_LA", "ρ", "δ_E", "δ_L", "δ_A", "κ_E", "κ_L", "κ_A", "σ"]   # order of `param_labels`
@assert length(COORD_NAMES) == n_dim

# ---------------------------------------------------------------------------------------------------------------------
# Two-slope map

struct TwoSlope
    θs::Float64
    slo::Float64
    shi::Float64
    ζ::Float64
end

fmap(m::TwoSlope, u) = m.θs + m.ζ * (m.shi * log1pexp(u / m.ζ) - m.slo * log1pexp(-u / m.ζ))
dfmap(m::TwoSlope, u) = m.slo * logistic(-u / m.ζ) + m.shi * logistic(u / m.ζ)
dlogdfmap(m::TwoSlope, u) = (m.shi - m.slo) * logistic(u / m.ζ) * logistic(-u / m.ζ) / (m.ζ * dfmap(m, u))

# F^{-1}(θ) by safeguarded Newton inside a bracket, with bisection when a Newton step leaves the bracket.
# F' ≥ min(s) and F(0) = θ* + ζ log 2 (s_high - s_low) give the bracket [0, (θ - F(0))/min(s)] (or its mirror).
function finv(m::TwoSlope, θ; rtol=1e-12, maxit=200)
    F0 = fmap(m, 0.0)
    a, b = minmax(0.0, (θ - F0) / min(m.slo, m.shi))
    u = θ > m.θs ? (θ - m.θs) / m.shi : (θ - m.θs) / m.slo   # inverse of the asymptotes
    u = clamp(u, a, b)
    tol = rtol * max(1.0, abs(θ))
    for _ in 1:maxit
        g = fmap(m, u) - θ
        abs(g) <= tol && return u
        g > 0 ? (b = u) : (a = u)
        un = u - g / dfmap(m, u)
        u = (a < un < b) ? un : 0.5 * (a + b)
        b - a <= eps(max(abs(a), abs(b), 1.0)) && return u
    end
    return u
end

# Parameter vector p = (θ*, log s_low, log s_high, log(ζ - ζ_min)).
to_map(p) = TwoSlope(p[1], exp(p[2]), exp(p[3]), ZETA_MIN + exp(p[4]))
from_map(m::TwoSlope) = [m.θs, log(m.slo), log(m.shi), log(m.ζ - ZETA_MIN)]

popvar(x) = var(x; corrected=false)

# latent values u_n and latent scores a_n = F'(u_n) α_n + (log F')'(u_n)
function latent(m::TwoSlope, θ, α)
    u = finv.(Ref(m), θ)
    a = dfmap.(Ref(m), u) .* α .+ dlogdfmap.(Ref(m), u)
    return u, a
end
Dk(m::TwoSlope, θ, α) = ((u, a) = latent(m, θ, α); popvar(a .+ u))

# Affine nutpie map θ = m + s u (the centre does not enter D): D = Var(s α + θ/s), minimised at s^2 = sqrt(Var θ / Var α)
nutpie_scale(θ, α) = (popvar(θ) / max(popvar(α), eps()))^(1 / 4)
D_affine(s, θ, α) = popvar(s .* α .+ θ ./ s)

# Parts used for the split start, the flags and the per-part latent scales.
split_point(k, θ) = k in ss_idxs ? thres : median(θ)

function fit_twoslope(k, θ, α; ζ0=1.0)
    s_glob = nutpie_scale(θ, α)
    cut = split_point(k, θ)
    lo, hi = θ .<= cut, θ .> cut
    n_lo, n_hi = count(lo), count(hi)
    # Split start: part-wise nutpie scale and centre (Seyboldt et al., Thm 2.2); a part with fewer than 20 particles
    # takes the other part's scale.
    slo = n_lo >= 20 ? nutpie_scale(θ[lo], α[lo]) : NaN
    shi = n_hi >= 20 ? nutpie_scale(θ[hi], α[hi]) : NaN
    isnan(slo) && (slo = isnan(shi) ? s_glob : shi)
    isnan(shi) && (shi = slo)
    mlo = n_lo >= 2 ? mean(θ[lo]) + slo^2 * mean(α[lo]) : NaN
    mhi = n_hi >= 2 ? mean(θ[hi]) + shi^2 * mean(α[hi]) : NaN
    d = 3 + ζ0 * abs(log(slo / shi))
    θs = shi < slo ? mhi - shi * d : mlo + slo * d
    isfinite(θs) || (θs = median(θ))

    starts = [:split => TwoSlope(θs, slo, shi, ζ0), :affine => TwoSlope(median(θ), s_glob, s_glob, ζ0)]
    obj(p) = (v = Dk(to_map(p), θ, α); isfinite(v) ? v : Inf)
    best = nothing
    for (name, m0) in starts
        p = from_map(m0)
        res = nothing
        for _ in 1:3   # restart Nelder-Mead from its own optimum
            res = optimize(obj, p, NelderMead(), Optim.Options(iterations=4000, f_reltol=1e-12, x_abstol=1e-10))
            p = Optim.minimizer(res)
        end
        D = Optim.minimum(res)
        if isnothing(best) || D < best.D
            best = (D=D, m=to_map(p), start=name, D_start=obj(from_map(m0)))
        end
    end

    m = best.m
    # flags: ζ at its lower bound; θ* within 2 local sds (part-wise Fisher scale) of either part's centre
    centres = filter(c -> isfinite(c.m), [(m=mlo, s=slo), (m=mhi, s=shi)])
    flag_ζ = m.ζ <= ZETA_MIN + 1e-2
    flag_θs = any(abs(m.θs - c.m) < 2c.s for c in centres)

    # per-part latent scale (Var_j(u)/Var_j(a))^{1/4} for the two-slope and the affine map (≈ 1: unit curvature)
    u2, a2 = latent(m, θ, α)
    ua, aa = θ ./ s_glob, s_glob .* α
    lscale(u, a, idx) = count(idx) >= 2 ? (popvar(u[idx]) / popvar(a[idx]))^(1 / 4) : NaN
    return (; best..., cut, n_lo, n_hi, m_lo=mlo, m_hi=mhi, s_lo_part=slo, s_hi_part=shi,
            flag_ζ, flag_θs, s_affine=s_glob, D_affine=D_affine(s_glob, θ, α),
            ls2_lo=lscale(u2, a2, lo), ls2_hi=lscale(u2, a2, hi), lsa_lo=lscale(ua, aa, lo), lsa_hi=lscale(ua, aa, hi),
            max_inv_err=maximum(abs.(fmap.(Ref(m), u2) .- θ)), u=u2, a=a2)
end

# ---------------------------------------------------------------------------------------------------------------------
# Step 3: tests

function run_tests(θtest, αtest)
    rng = StableRNG(1)
    println("\n== Tests ==")
    # finite differences of F' and (log F')'
    err1, err2 = 0.0, 0.0
    for _ in 1:2000
        m = TwoSlope(randn(rng) * 5, exp(2randn(rng)), exp(2randn(rng)), ZETA_MIN + exp(randn(rng)))
        u = 4randn(rng); h = 1e-5
        fd1 = (fmap(m, u + h) - fmap(m, u - h)) / 2h
        fd2 = (log(dfmap(m, u + h)) - log(dfmap(m, u - h))) / 2h
        err1 = max(err1, abs(fd1 - dfmap(m, u)) / max(1, abs(fd1)))
        err2 = max(err2, abs(fd2 - dlogdfmap(m, u)) / max(1, abs(fd2)))
    end
    println("max rel. error, F' vs central difference:        ", err1)
    println("max rel. error, (log F')' vs central difference: ", err2)

    # inverse accuracy on real coordinate values, for several (including extreme) maps
    worst = 0.0
    for m in [TwoSlope(thres, 0.5, 5., ZETA_MIN), TwoSlope(thres, 50., 0.05, ZETA_MIN),
              TwoSlope(median(θtest), 1., 1., 1.), TwoSlope(0., 1e-3, 1e2, 3.)]
        u = finv.(Ref(m), θtest)
        worst = max(worst, maximum(abs.(fmap.(Ref(m), u) .- θtest)))
    end
    println("max |F(F^-1(θ)) - θ| over test maps:            ", worst)

    # optimising D over the affine family must give the closed-form nutpie scale
    s_cf = nutpie_scale(θtest, αtest)
    res = optimize(ls -> D_affine(exp(ls), θtest, αtest), log(s_cf) - 3, log(s_cf) + 3, Brent(); abs_tol=1e-12)
    s_opt = exp(Optim.minimizer(res))
    println("affine scale: closed form $(s_cf), numerical $(s_opt), rel. diff $(abs(s_opt - s_cf) / s_cf)")
    # the two-slope map with s_low = s_high is the affine map
    m_eq = TwoSlope(median(θtest), s_cf, s_cf, 1.0)
    println("two-slope with equal slopes: D = $(Dk(m_eq, θtest, αtest)), affine D = $(D_affine(s_cf, θtest, αtest))")
    println()
end

# ---------------------------------------------------------------------------------------------------------------------
# Step 4: iteration table and default selection (states only)

slab_prop(states, k) = mean(s[k] > thres for s in states)
gap_occ(states, k) = mean(abs(s[k] - thres) < 4 for s in states)

iter_rows = map(0:n_iter) do i
    states = [p.state for p in all_particles[i+1]]
    (i=i, γ=γs[i+1], slab=[slab_prop(states, k) for k in ss_idxs], gap=[gap_occ(states, k) for k in ss_idxs])
end

println("\nRun $(fm_run_name) on data $(dir_idx), path: $(fm_path); spike-and-slab coordinates $(ss_idxs), threshold $(thres)")
println(rpad("iter", 6), rpad("γ", 10), join(rpad.("slab " .* COORD_NAMES[ss_idxs], 11)), join(rpad.("gap " .* COORD_NAMES[ss_idxs], 10)))
for r in iter_rows
    println(rpad(r.i, 6), rpad(round(r.γ, sigdigits=3), 10), join(rpad.(round.(r.slab, digits=3), 11)),
            join(rpad.(round.(r.gap, digits=3), 10)))
end
open(joinpath(fm_out, "iteration_table.csv"), "w") do io
    println(io, join(["iter", "gamma", ("slab_" .* COORD_NAMES[ss_idxs])..., ("gap_" .* COORD_NAMES[ss_idxs])...], ","))
    for r in iter_rows
        println(io, join([r.i, r.γ, r.slab..., r.gap...], ","))
    end
end

# 4 iterations with at least 2 spike-and-slab coordinates of slab proportion in [0.1, 0.9], spread from high to low
# mean gap occupancy, plus the final iteration.
function default_iters(rows)
    cand = [r for r in rows if 1 <= r.i < n_iter && count(0.1 .<= r.slab .<= 0.9) >= 2]
    sel = Int[]
    if !isempty(cand)
        cand = sort(cand; by=r -> -mean(r.gap))
        pos = unique(round.(Int, range(1, length(cand), length=min(4, length(cand)))))
        sel = sort([cand[j].i for j in pos])
    end
    return unique([sel; n_iter])
end
sel_iters = isempty(fm_iters_override) ? default_iters(iter_rows) : fm_iters_override
println("Selected iterations: ", sel_iters, " (γ = ", round.(γs[sel_iters .+ 1], sigdigits=3), ")")

# ---------------------------------------------------------------------------------------------------------------------
# Step 1: scores at the stored particles, with the run's own target (as in `compute_invmass`)

function scores(i)
    γ = γs[i+1]
    target = ldp_builder(prior_path, γ)
    logp_func = Base.Fix1(LogDensityProblems.logdensity, target)
    particles = all_particles[i+1]
    n = length(particles)
    X, G, logp = Matrix{Float64}(undef, n_dim, n), Matrix{Float64}(undef, n_dim, n), Vector{Float64}(undef, n)
    result = DiffResults.GradientResult(particles[1].state)
    for (j, p) in enumerate(particles)
        X[:, j] .= p.state
        result = ForwardDiff.gradient!(result, logp_func, p.state)
        logp[j] = DiffResults.value(result)
        G[:, j] .= DiffResults.gradient(result)
    end
    # the stored log targets were evaluated with the run's own path at γ_i, so they must match
    stored = [p.logtarget for p in particles]
    ok = isfinite.(logp) .& vec(all(isfinite, G; dims=1))
    both = ok .& isfinite.(stored)
    max_diff = maximum(abs.(logp[both] .- stored[both]); init=0.0)
    @info "Iteration $i scores" γ n_nonfinite = count(!, ok) max_abs_diff_vs_stored_logtarget = max_diff
    max_diff < 1e-6 || error("Recomputed log target differs from the stored one (max $(max_diff)): wrong path or γ?")
    return X[:, ok], G[:, ok], logp[ok]
end

# ---------------------------------------------------------------------------------------------------------------------
# Steps 2, 5, 6: fits, plots, summary

function plot_iteration(i, γ, fits, X, fit_pop)
    ncol = 5
    fig = Figure(size=(ncol * 430, cld(n_dim, ncol) * 470), fontsize=11)
    for k in 1:n_dim
        f, m, θ, u = fits[k], fits[k].m, X[k, :], fits[k].u
        gl = GridLayout(fig[fld1(k, ncol), mod1(k, ncol)])
        ax_top = Axis(gl[1, 1]; ylabel="density (u)", xticklabelsvisible=false)
        ax = Axis(gl[2, 1]; xlabel="u", ylabel="θ")
        ax_right = Axis(gl[2, 2]; xlabel="density (θ)", yticklabelsvisible=false)
        linkxaxes!(ax_top, ax); linkyaxes!(ax_right, ax)
        rowsize!(gl, 1, Relative(0.28)); colsize!(gl, 2, Relative(0.28))
        rowgap!(gl, 4); colgap!(gl, 4)

        ugrid = range(minimum(u), maximum(u), length=400)
        lines!(ax, ugrid, fmap.(Ref(m), ugrid); color=:black)
        hlines!(ax, [m.θs]; color=:gray, linestyle=:dash)
        vlines!(ax, [0.0]; color=:gray, linestyle=:dash)
        k in ss_idxs && hlines!(ax, [thres]; color=:red, linestyle=:dot)

        ĉ = mean(f.a .+ u)
        hist!(ax_top, u; bins=60, normalization=:pdf, color=(:steelblue, 0.6))
        lines!(ax_top, ugrid, pdf.(Normal(ĉ, 1.0), ugrid); color=:black)
        hist!(ax_right, θ; bins=60, normalization=:pdf, direction=:x, color=(:darkorange, 0.6))
        k in ss_idxs && hlines!(ax_right, [thres]; color=:red, linestyle=:dot)
        # keep θ* and threshold lines from stretching the axis beyond the particles
        lo, hi = extrema(θ)
        pad = 0.04 * (hi - lo)
        ylims!(ax, lo - pad, hi + pad)

        flags = (f.flag_ζ ? "  [ζ at bound]" : "") * (f.flag_θs ? "  [θ* near centre]" : "")
        Label(gl[0, :],
            "$(COORD_NAMES[k])   γ = $(round(γ, sigdigits=3))$(flags)\n" *
            "θ* = $(round(m.θs, digits=2)), s_low = $(round(m.slo, sigdigits=3)), s_high = $(round(m.shi, sigdigits=3)), ζ = $(round(m.ζ, sigdigits=3))\n" *
            "D two-slope = $(round(f.D, sigdigits=4)), D affine = $(round(f.D_affine, sigdigits=4))";
            fontsize=11, tellwidth=false)
    end
    Label(fig[0, :], "$(fm_run_name), data $(dir_idx), path: $(fm_path), iteration $(i), fit population: $(fit_pop)";
          fontsize=14, font=:bold)
    fpath = joinpath(fm_out, "iter$(i)" * (length(fm_fit_pops) > 1 ? "_$(fit_pop)" : "") * ".png")
    save(fpath, fig; px_per_unit=1.5)
    return fpath
end

summary_cols = ["run", "path", "data", "iter", "gamma", "fit_pop", "n_fit", "coord", "coord_name", "is_ss", "slab_prop",
    "gap_occ", "theta_star", "s_low", "s_high", "zeta", "start", "flag_zeta_at_bound", "flag_theta_star_near_centre",
    "D_twoslope", "D_start", "D_affine", "D_affine_full_scale", "s_affine", "s_affine_full", "split", "n_low", "n_high",
    "m_low", "m_high", "s_part_low", "s_part_high", "latent_scale_low_twoslope", "latent_scale_high_twoslope",
    "latent_scale_low_affine", "latent_scale_high_affine", "max_inv_err"]
summary_rows = Vector{Vector{Any}}()
png_paths = String[]

for (ii, i) in enumerate(sel_iters)
    γ = γs[i+1]
    X, G, logp = scores(i)
    keep_trim = logp .>= quantile(logp, 0.01)   # as in `compute_invmass`
    states = [X[:, j] for j in axes(X, 2)]
    ii == 1 && run_tests(X[ss_idxs[1], keep_trim], G[ss_idxs[1], keep_trim])
    for fit_pop in fm_fit_pops
        keep = fit_pop == "trim" ? keep_trim : trues(length(logp))
        fits = map(1:n_dim) do k
            θ, α = X[k, keep], G[k, keep]
            t = @elapsed f = fit_twoslope(k, θ, α)
            s_full = nutpie_scale(X[k, :], G[k, :])   # affine baseline from the untrimmed population
            D_full = D_affine(s_full, θ, α)
            @info "iter $i $(COORD_NAMES[k]) ($fit_pop)" t f.start f.m f.D f.D_affine D_full f.flag_ζ f.flag_θs f.max_inv_err
            push!(summary_rows, Any[fm_run_name, fm_path, dir_idx, i, γ, fit_pop, count(keep), k, COORD_NAMES[k],
                k in ss_idxs, k in ss_idxs ? slab_prop(states, k) : NaN, k in ss_idxs ? gap_occ(states, k) : NaN,
                f.m.θs, f.m.slo, f.m.shi, f.m.ζ, f.start, f.flag_ζ, f.flag_θs,
                f.D, f.D_start, f.D_affine, D_full, f.s_affine, s_full, f.cut, f.n_lo, f.n_hi,
                f.m_lo, f.m_hi, f.s_lo_part, f.s_hi_part, f.ls2_lo, f.ls2_hi, f.lsa_lo, f.lsa_hi, f.max_inv_err])
            f
        end
        push!(png_paths, plot_iteration(i, γ, fits, X[:, keep], fit_pop))
    end
    # rewrite after every iteration so partial results survive
    open(joinpath(fm_out, "summary.csv"), "w") do io
        println(io, join(summary_cols, ","))
        foreach(r -> println(io, join(r, ",")), summary_rows)
    end
end

println("\nOutputs:")
println("  ", joinpath(fm_out, "iteration_table.csv"))
println("  ", joinpath(fm_out, "summary.csv"))
foreach(p -> println("  ", p), png_paths)
