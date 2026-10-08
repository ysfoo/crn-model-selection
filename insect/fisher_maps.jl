# Two-slope Fisher maps (appendix "Fisher divergence of a transformed target"): the map, its fit to a population, safe
# loading of a run's variant script, guards against degenerate fits and the transformed log density for NUTS.
# Expects `ZETA_MIN`, `ss_idxs` and `thres` to be defined (by the including script and SMC.jl).

using LogDensityProblems

# Evaluate a variant script (e.g. SMC505.jl) for its `prior_path` and settings, minus `include(... SMC.jl)` and the
# run/resume call. Statements ending in `;` are parsed as nested `:toplevel` blocks, so the filter recurses into them.
# Running the variant's `run_SMC` would overwrite the stored run (`fname`), so refuse to evaluate if any call survives.
function load_variant!(path)
    ex = Meta.parseall(read(path, String); filename=path)
    is_skipped(e) = e isa Expr && e.head == :call && e.args[1] in (:include, :run_SMC, :resume_SMC)
    function strip_calls!(e)
        e isa Expr && e.head == :toplevel || return e
        filter!(x -> !is_skipped(x), e.args)
        foreach(strip_calls!, e.args)
        return e
    end
    calls_run(e) = e isa Expr && ((e.head == :call && e.args[1] in (:run_SMC, :resume_SMC)) || any(calls_run, e.args))
    strip_calls!(ex)
    calls_run(ex) && error("$(path) still calls run_SMC/resume_SMC after filtering; not evaluating it")
    Base.eval(Main, ex)
    return nothing
end


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
# Guards for use in a sampler
#
# A fit is only constrained where there are particles. With one side (nearly) empty, the slope on that side is
# unidentified and can come out as 1e-224 or 4e7 (for δ_A on data 4 SMC505_4k iteration 26, with 9 spike particles, all
# particles sit at u > 35, so F'(u) ≈ s_high + s_low e^{-u/ζ} on the population and s_low trades off against θ*; D is flat
# for s_low ≥ 1000). Such a map is fine on the population but extreme beyond it: F' grows towards s_low, so the latent
# curvature F'^2 h explodes in the tail, or F is nearly constant and truncates the support in floating point.
# Fall back to the equal-slope optimum (the affine nutpie map, s^2 = sqrt(Var θ / Var α), which minimises D over
# s_low = s_high) unless
#   - both parts (split at `thres` for spike-and-slab coordinates) have at least n_min particles,
#   - at least n_min particles lie on each side of θ*,
#   - s_high / s_low is within [1/max_ratio, max_ratio], and ζ is finite.
# NUTS then runs in u with an identity mass matrix: the D-optimal map already sets the scale of each coordinate.
equal_slope_map(θ, α) = (s = nutpie_scale(θ, α); TwoSlope(median(θ), s, s, 1.0))   # F(u) = median + s u
function guard_map(f, θ, α; n_min=50, max_ratio=1e4)
    m = f.m
    reason = if min(f.n_lo, f.n_hi) < n_min
        "part with < $(n_min) particles"
    elseif min(count(θ .< m.θs), count(θ .> m.θs)) < n_min
        "< $(n_min) particles on one side of θ*"
    elseif !(1 / max_ratio <= m.shi / m.slo <= max_ratio) || !isfinite(m.ζ)
        "slope ratio outside [1/$(max_ratio), $(max_ratio)]"
    else
        ""
    end
    return isempty(reason) ? (m, "") : (equal_slope_map(θ, α), reason)
end

# Target in u-space: log π(F(u)) + Σ_k log F_k'(u_k). Exact change of variables for any fixed set of bijective maps.
struct TransformedTarget{T}
    target::T
    maps::Vector{TwoSlope}
end
LogDensityProblems.dimension(t::TransformedTarget) = LogDensityProblems.dimension(t.target)
LogDensityProblems.capabilities(::Type{<:TransformedTarget}) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(t::TransformedTarget, u)
    θ = [fmap(m, x) for (m, x) in zip(t.maps, u)]
    logjac = sum(log(dfmap(m, x)) for (m, x) in zip(t.maps, u))
    return LogDensityProblems.logdensity(t.target, θ) + logjac
end
to_theta(maps, u) = [fmap(m, x) for (m, x) in zip(maps, u)]
to_latent(maps, θ) = [finv(m, x) for (m, x) in zip(maps, θ)]
