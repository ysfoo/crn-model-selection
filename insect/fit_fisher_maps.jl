# Fit two-slope Fisher maps to stored SMC populations and plot them (exploratory; does not run or change the sampler).
#
# Usage (from insect/):
#   julia --project=.. fit_fisher_maps.jl <data idx> <run name> [iterations...] [--fit=trim|full|both] [--zeta_min=0.2]
#                                         [--variant=SMCnnn] [--min_gamma=0.01]
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

fm_run_str = get(fm_flags, "variant", split(fm_run_name, "_")[1])
fm_variant = joinpath(@__DIR__, "$(fm_run_str).jl")
# Run's own path; `load_variant!` never evaluates the variant's run_SMC/resume_SMC call (see fisher_maps.jl).
# `--variant=SMCnnn` takes the path from another variant with the same path (e.g. SMC505 for SMC501_4k); the log
# target check in `scores` confirms it reproduces the stored values.
include(joinpath(@__DIR__, "fisher_maps.jl"))
load_variant!(fm_variant)
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
# mean gap occupancy, plus the final iteration. Iterations with γ < min_gamma are skipped: there the population still
# follows the initial distribution, so slab proportions near 0.5 say nothing.
const MIN_GAMMA = parse(Float64, get(fm_flags, "min_gamma", "0.01"))
function default_iters(rows)
    cand = [r for r in rows if 1 <= r.i < n_iter && r.γ >= MIN_GAMMA && count(0.1 .<= r.slab .<= 0.9) >= 2]
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

# ---------------------------------------------------------------------------------------------------------------------
# Curvature diagnostics for NUTS (skip with --no_curv)
#
# Along coordinate k, with the other coordinates fixed, the latent potential is U(u) = -log π(θ(u)) - log F'(u) with
# θ_k = F(u), so
#   U''(u) = -[F'(u)^2 h + F''(u) g + (log F')''(u)],   g = ∂_k log π, h = ∂²_k log π, F'' = F' (log F')'.
# NUTS on the transformed target would use a diagonal mass matrix estimated in u-space (nutpie, M^{-1}_kk =
# sqrt(Var u / Var a)), so the leapfrog integrator sees the scaled curvature κ = M^{-1}_kk U''. It is unstable where
# ε √κ > 2, and the number of steps per U-turn grows with √(κ_max / κ_min). For the affine nutpie map, which is the
# current sampler, κ = -s^2 h with s^2 = sqrt(Var θ / Var α) (the u-space mass matrix is then the identity).
# Stored step sizes are those of the current (affine) sampler at that iteration, in the same scaled units.

d2logdfmap(m::TwoSlope, u) = ForwardDiff.derivative(v -> dlogdfmap(m, v), u)

function line_func(logp_func, θ, k)
    return t -> begin
        x = similar(θ, typeof(t)); x .= θ; x[k] += t
        logp_func(x)
    end
end
# Second derivative of log π along coordinate k. Nested forward mode; if that fails through the ODE solver, a central
# difference of the forward-mode first derivative.
const NESTED_AD = Ref(true)
function line_h(logp_func, θ, k; δ=1e-4)
    df = t -> ForwardDiff.derivative(line_func(logp_func, θ, k), t)
    if NESTED_AD[]
        try
            return ForwardDiff.derivative(df, 0.0)
        catch err
            @warn "Nested forward mode failed; using central differences of the first derivative" exception = err
            NESTED_AD[] = false
        end
    end
    return (df(δ) - df(-δ)) / 2δ
end
line_vgh(logp_func, θ, k) = begin
    f = line_func(logp_func, θ, k)
    (f(0.0), ForwardDiff.derivative(f, 0.0), line_h(logp_func, θ, k))
end

latent_minv(f) = sqrt(popvar(f.u) / popvar(f.a))
κ_twoslope(f, u, g, h) = begin
    m = f.m
    Fp, L1, L2 = dfmap(m, u), dlogdfmap(m, u), d2logdfmap(m, u)
    -latent_minv(f) * (Fp^2 * h + Fp * L1 * g + L2)
end
κ_affine(f, h) = -f.s_affine^2 * h

# Scaled curvature of every (trimmed) particle and coordinate, for both maps.
function population_curvature(logp_func, X, G, fits)
    n = size(X, 2)
    K2, KA = Matrix{Float64}(undef, n_dim, n), Matrix{Float64}(undef, n_dim, n)
    for j in 1:n, k in 1:n_dim
        h = line_h(logp_func, X[:, j], k)
        K2[k, j] = κ_twoslope(fits[k], fits[k].u[j], G[k, j], h)
        KA[k, j] = κ_affine(fits[k], h)
    end
    return K2, KA
end

const KFLOOR = 1e-4   # κ ≤ 0 (no restoring force) is drawn at this floor
logκ(κ) = log10(max(κ, KFLOOR))
qs_pos(κ, ps) = (p = filter(>(0), κ); isempty(p) ? fill(NaN, length(ps)) : quantile(p, ps))
cond_spread(κ) = (p = filter(>(0), vec(κ)); isempty(p) ? NaN : sqrt(quantile(p, 0.99) / quantile(p, 0.01)))

function plot_curvature(i, γ, X, fits, K2, KA, ε)
    ncol = 5
    fig = Figure(size=(ncol * 400, cld(n_dim, ncol) * 380), fontsize=11)
    lim = (2 / ε)^2
    for k in 1:n_dim
        f = fits[k]
        ax = Axis(fig[fld1(k, ncol), mod1(k, ncol)]; xlabel="θ", ylabel="log10 κ",
            title="$(COORD_NAMES[k])   neg. κ: two-slope $(round(100mean(K2[k, :] .<= 0), digits=1)) %, " *
                  "affine $(round(100mean(KA[k, :] .<= 0), digits=1)) %", titlesize=11)
        scatter!(ax, X[k, :], logκ.(KA[k, :]); color=(:gray50, 0.35), markersize=3, label="affine (current)")
        scatter!(ax, X[k, :], logκ.(K2[k, :]); color=(:steelblue, 0.5), markersize=3, label="two-slope")
        hlines!(ax, [log10(lim)]; color=:red, linestyle=:dash, label="(2/ε)², ε = median stored")
        hlines!(ax, [0.0]; color=:black, linestyle=:dot, label="κ = 1")
        lo, hi = extrema(X[k, :])
        lo <= f.m.θs <= hi && vlines!(ax, [f.m.θs]; color=:gray30, linestyle=:dash)
        k in ss_idxs && vlines!(ax, [thres]; color=:red, linestyle=:dot)
        k == 1 && axislegend(ax; position=:lt, labelsize=9, framevisible=false)
    end
    Label(fig[0, :], "$(fm_run_name), data $(dir_idx), iteration $(i), γ = $(round(γ, sigdigits=3)): scaled latent " *
        "curvature κ = M⁻¹U'' per particle (κ ≤ 0 drawn at 1e-4; dashed grey: θ*, dotted red: threshold)\n" *
        "pooled spread √(q99/q01) of κ > 0: two-slope $(round(cond_spread(K2), sigdigits=3)), " *
        "affine $(round(cond_spread(KA), sigdigits=3)); stability limit (2/ε)² = $(round(lim, sigdigits=3))";
        fontsize=13, font=:bold)
    fpath = joinpath(fm_out, "curv_iter$(i).png")
    save(fpath, fig; px_per_unit=1.5)
    return fpath
end

# Conditional scans across the bend: for a few particles on each side of a spike-and-slab coordinate, move only that
# coordinate over the population's range and record U (barrier) and κ (curvature) for both maps, against θ.
function plot_scans(i, γ, logp_func, X, fits; n_per_side=5, n_grid=150, rng=StableRNG(i))
    ks = [k for k in ss_idxs if fits[k].n_lo >= 20 && fits[k].n_hi >= 20]
    isempty(ks) && return nothing
    fig = Figure(size=(length(ks) * 380, 2 * 330 + 80), fontsize=11)
    for (c, k) in enumerate(ks)
        f, m = fits[k], fits[k].m
        θk = X[k, :]
        ugrid = range(minimum(f.u), maximum(f.u), length=n_grid)
        θgrid = fmap.(Ref(m), ugrid)
        axU = Axis(fig[1, c]; ylabel="U - U(start)  [nats]", xticklabelsvisible=false,
            title="$(COORD_NAMES[k]): θ* = $(round(m.θs, digits=2)), ζ = $(round(m.ζ, sigdigits=2)), " *
                  "M⁻¹ = $(round(latent_minv(f), sigdigits=3))", titlesize=11)
        axK = Axis(fig[2, c]; xlabel="θ", ylabel="log10 κ")
        linkxaxes!(axU, axK)
        for (side, idx, col) in (("spike", findall(θk .<= f.cut), :darkorange), ("slab", findall(θk .> f.cut), :steelblue))
            for j in idx[randperm(rng, length(idx))[1:n_per_side]]
                vgh = [line_vgh(logp_func, X[:, j] .+ (θ - θk[j]) .* (1:n_dim .== k), k) for θ in θgrid]
                v, g, h = getindex.(vgh, 1), getindex.(vgh, 2), getindex.(vgh, 3)
                U2 = -v .- log.(dfmap.(Ref(m), ugrid))
                UA = -v
                U2_0 = -logp_func(X[:, j]) - log(dfmap(m, f.u[j]))
                UA_0 = -logp_func(X[:, j])
                lines!(axU, θgrid, U2 .- U2_0; color=(col, 0.8))
                lines!(axU, θgrid, UA .- UA_0; color=(col, 0.5), linestyle=:dash)
                lines!(axK, θgrid, logκ.(κ_twoslope.(Ref(f), ugrid, g, h)); color=(col, 0.8))
                lines!(axK, θgrid, logκ.(κ_affine.(Ref(f), h)); color=(col, 0.5), linestyle=:dash)
                scatter!(axU, [θk[j]], [0.0]; color=col, markersize=6)
            end
        end
        ylims!(axU, -8, 30)
        xlims!(axK, max(minimum(θgrid), thres - 8), maximum(θgrid))   # spike centre to the slab; axes are linked
        hlines!(axK, [0.0]; color=:black, linestyle=:dot)
        for ax in (axU, axK)
            vlines!(ax, [m.θs]; color=:gray30, linestyle=:dash)
            vlines!(ax, [thres]; color=:red, linestyle=:dot)
        end
    end
    Label(fig[0, :], "$(fm_run_name), data $(dir_idx), iteration $(i), γ = $(round(γ, sigdigits=3)): conditional scans of " *
        "one coordinate (others fixed) from $(n_per_side) spike-side (orange) and $(n_per_side) slab-side (blue) particles\n" *
        "solid: two-slope latent potential and scaled curvature; dashed: affine (current); dots: start particles; " *
        "κ ≤ 0 drawn at 1e-4"; fontsize=13, font=:bold)
    fpath = joinpath(fm_out, "scan_iter$(i).png")
    save(fpath, fig; px_per_unit=1.5)
    return fpath
end

function plot_curv_summary(curv)
    isempty(curv) && return nothing
    fig = Figure(size=(1500, 300 * cld(length(curv), 2) + 60), fontsize=11)
    for (c, r) in enumerate(curv)
        ax = Axis(fig[fld1(c, 2), mod1(c, 2)]; yticks=(1:n_dim, COORD_NAMES), xlabel="log10 √κ (κ > 0)",
            title="iteration $(r.i), γ = $(round(r.γ, sigdigits=3)); spread √(q99/q01): two-slope " *
                  "$(round(cond_spread(r.K2), sigdigits=3)), affine $(round(cond_spread(r.KA), sigdigits=3))",
            titlesize=11, yreversed=true)
        for (K, off, col) in ((r.KA, 0.17, :gray40), (r.K2, -0.17, :steelblue))
            for k in 1:n_dim
                q = log10.(sqrt.(qs_pos(K[k, :], [0.01, 0.1, 0.5, 0.9, 0.99])))
                all(isfinite, q) || continue
                lines!(ax, [q[1], q[5]], [k + off, k + off]; color=col, linewidth=1)
                lines!(ax, [q[2], q[4]], [k + off, k + off]; color=col, linewidth=4)
                scatter!(ax, [q[3]], [k + off]; color=:black, markersize=5)
            end
        end
        vlines!(ax, [log10(2 / r.ε)]; color=:red, linestyle=:dash)
    end
    Label(fig[0, :], "$(fm_run_name), data $(dir_idx): scaled curvature per coordinate (grey: affine, current; blue: " *
        "two-slope). Thin: 1-99 %, thick: 10-90 %, dot: median; red dashed: 2/ε for the stored step size";
        fontsize=13, font=:bold)
    fpath = joinpath(fm_out, "curv_summary.png")
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
curv_cols = ["run", "data", "iter", "gamma", "coord", "coord_name", "map", "eps_median_stored", "stability_limit",
    "frac_nonpositive", "q01_pos", "q10_pos", "q50_pos", "q90_pos", "q99_pos", "frac_above_limit"]
curv_rows = Vector{Vector{Any}}()
curv_results = []

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

        # curvature diagnostics, for the first fit population only
        if fit_pop == fm_fit_pops[1] && !haskey(fm_flags, "no_curv")
            logp_func = Base.Fix1(LogDensityProblems.logdensity, ldp_builder(prior_path, γ))
            ε = median(p.stepsize for p in all_particles[i+1])
            Xk, Gk = X[:, keep], G[:, keep]
            t = @elapsed((K2, KA) = population_curvature(logp_func, Xk, Gk, fits))
            @info "iter $i curvature" t ε nested_AD = NESTED_AD[] spread_twoslope = cond_spread(K2) spread_affine = cond_spread(KA)
            push!(curv_results, (i=i, γ=γ, ε=ε, K2=K2, KA=KA))
            for k in 1:n_dim, (map_name, K) in (("twoslope", K2), ("affine", KA))
                push!(curv_rows, Any[fm_run_name, dir_idx, i, γ, k, COORD_NAMES[k], map_name, ε, (2 / ε)^2,
                    mean(K[k, :] .<= 0), qs_pos(K[k, :], [0.01, 0.1, 0.5, 0.9, 0.99])..., mean(K[k, :] .> (2 / ε)^2)])
            end
            push!(png_paths, plot_curvature(i, γ, Xk, fits, K2, KA, ε))
            t = @elapsed p = plot_scans(i, γ, logp_func, Xk, fits)
            @info "iter $i scans" t
            isnothing(p) || push!(png_paths, p)
            open(joinpath(fm_out, "curvature.csv"), "w") do io
                println(io, join(curv_cols, ","))
                foreach(r -> println(io, join(r, ",")), curv_rows)
            end
        end
    end
    # rewrite after every iteration so partial results survive
    open(joinpath(fm_out, "summary.csv"), "w") do io
        println(io, join(summary_cols, ","))
        foreach(r -> println(io, join(r, ",")), summary_rows)
    end
end

p = plot_curv_summary(curv_results)
isnothing(p) || push!(png_paths, p)

println("\nOutputs:")
println("  ", joinpath(fm_out, "iteration_table.csv"))
println("  ", joinpath(fm_out, "summary.csv"))
isempty(curv_rows) || println("  ", joinpath(fm_out, "curvature.csv"))
foreach(p -> println("  ", p), png_paths)
