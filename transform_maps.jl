# Per-coordinate two-slope maps for NUTS on a transformed target (thesis appendix "Fisher divergence of a transformed
# target"). θ_k = F_k(u_k) with
#   F(u)  = θ* + ζ [ s_high softplus(u/ζ) - s_low softplus(-u/ζ) ],
#   F'(u) = s_low logistic(-u/ζ) + s_high logistic(u/ζ),
# fitted to a population (θ_n, α_n = ∇ log π(θ_n)) by minimising the sample Fisher divergence of the pull-back to a
# standard normal with a free centre, D = Var_n(F'(u_n) α_n + (log F')'(u_n) + u_n), u_n = F^{-1}(θ_n).

using ForwardDiff, LogDensityProblems, LogExpFunctions, Optim, Statistics

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

to_theta(maps, u) = [fmap(m, x) for (m, x) in zip(maps, u)]
to_latent(maps, θ) = [finv(m, x) for (m, x) in zip(maps, θ)]
logjac(maps, u) = sum(log(dfmap(m, x)) for (m, x) in zip(maps, u))

# Parameter vector p = (θ*, log s_low, log s_high, log(ζ - ζ_min)).
to_map(p, ζ_min) = TwoSlope(p[1], exp(p[2]), exp(p[3]), ζ_min + exp(p[4]))
from_map(m::TwoSlope, ζ_min) = [m.θs, log(m.slo), log(m.shi), log(m.ζ - ζ_min)]

popvar(x) = var(x; corrected=false)

# latent values u_n and latent scores a_n = F'(u_n) α_n + (log F')'(u_n)
function latent(m::TwoSlope, θ, α)
    u = finv.(Ref(m), θ)
    a = dfmap.(Ref(m), u) .* α .+ dlogdfmap.(Ref(m), u)
    return u, a
end
Dk(m::TwoSlope, θ, α) = ((u, a) = latent(m, θ, α); popvar(a .+ u))

# Affine nutpie map θ = m + s u (the centre does not enter D): D = Var(s α + θ/s), minimised at s^2 = sqrt(Var θ / Var α).
# It is also the minimiser of D over two-slope maps with s_low = s_high.
nutpie_scale(θ, α) = (popvar(θ) / max(popvar(α), eps()))^(1 / 4)
D_affine(s, θ, α) = popvar(s .* α .+ θ ./ s)
equal_slope_map(θ, α) = (s = nutpie_scale(θ, α); TwoSlope(median(θ), s, s, 1.0))   # F(u) = median + s u

# Penalty on the log slope ratio, λ (log s_high - log s_low)^2, added to D. D is a population average: a part holding a
# fraction w_j of the particles, roughly Gaussian in latent space with its slope off by a factor e^δ_j, adds about
# w_j (2 sinh δ_j)^2 ≈ 4 w_j δ_j^2. Forcing a log slope ratio L to equal slopes then costs about 4 w_lo w_hi L^2, so the
# penalised log ratio is about L · 4 w_lo w_hi / (4 w_lo w_hi + λ). With λ = 4 n0 / N and one small part this is
# L · n_lo / (n_lo + n0): n0 is roughly the number of particles at which a moderate log ratio is shrunk halfway. Large,
# well-supported ratios shrink much less (2 sinh δ grows exponentially), while a slope with no particles behind it
# costs D nothing and is pulled to the other one, which removes the ridge of unidentified fits (e.g. s_low = 4e7 with no
# particle below the bend). Default λ = 100 / N (n0 = 25): on stored insect populations this resolved every degenerate
# fit while changing well-supported log ratios by about 0.05 (median) and D by under 0.1 % (median).
penalty_lambda(n0, N) = 4n0 / N
const DEFAULT_λN = 100.0   # λ = DEFAULT_λN / N

# Split start at `cut`: part-wise nutpie scale and centre (Seyboldt et al., Thm 2.2), ζ = ζ0, and θ* placed d local
# scales from the centre of the narrower part. A part with fewer than 20 particles takes the other part's scale.
function split_start(θ, α, cut; ζ0=1.0)
    s_glob = nutpie_scale(θ, α)
    lo, hi = θ .<= cut, θ .> cut
    n_lo, n_hi = count(lo), count(hi)
    slo = n_lo >= 20 ? nutpie_scale(θ[lo], α[lo]) : NaN
    shi = n_hi >= 20 ? nutpie_scale(θ[hi], α[hi]) : NaN
    isnan(slo) && (slo = isnan(shi) ? s_glob : shi)
    isnan(shi) && (shi = slo)
    mlo = n_lo >= 2 ? mean(θ[lo]) + slo^2 * mean(α[lo]) : NaN
    mhi = n_hi >= 2 ? mean(θ[hi]) + shi^2 * mean(α[hi]) : NaN
    d = 3 + ζ0 * abs(log(slo / shi))
    θs = shi < slo ? mhi - shi * d : mlo + slo * d
    isfinite(θs) || (θs = median(θ))
    return (m=TwoSlope(θs, slo, shi, ζ0), n_lo, n_hi, m_lo=mlo, m_hi=mhi)
end

# Penalised objective D + λ (log s_high - log s_low)^2 at p = (θ*, log s_low, log s_high, log(ζ - ζ_min)), written for
# ForwardDiff. u_n is found by Newton on Float64 values (`finv`), then one Newton correction with the (dual) parameters,
# which at the solution gives the exact derivative du/dp = -(∂F/∂p) / F'(u). The variance is computed in two passes:
# the one-pass formula (mean of squares minus squared mean) loses all precision at large ζ, where a gradient-based
# optimiser then finds spurious negative "variances". Returns Inf where the value is not finite.
function penalised_D(p::AbstractVector{S}, θ, α, λ, ζ_min) where S
    θs, slo, shi, ζ = p[1], exp(p[2]), exp(p[3]), ζ_min + exp(p[4])
    v = ForwardDiff.value
    mv = TwoSlope(v(θs), v(slo), v(shi), v(ζ))
    n = length(θ)
    xs = Vector{S}(undef, n)
    for i in 1:n
        u0 = finv(mv, θ[i])
        F = θs + ζ * (shi * log1pexp(u0 / ζ) - slo * log1pexp(-u0 / ζ))
        Fp0 = slo * logistic(-u0 / ζ) + shi * logistic(u0 / ζ)
        u = u0 - (F - θ[i]) / Fp0
        Fp = slo * logistic(-u / ζ) + shi * logistic(u / ζ)
        L1 = (shi - slo) * logistic(u / ζ) * logistic(-u / ζ) / (ζ * Fp)
        xs[i] = Fp * α[i] + L1 + u
    end
    μ = sum(xs) / n
    val = sum(x -> (x - μ)^2, xs) / n + λ * (p[3] - p[2])^2
    return isfinite(v(val)) ? val : S(Inf)
end

# Fit a two-slope map to one coordinate: one BFGS run (ForwardDiff gradients of `penalised_D`, stopping when the largest
# gradient component is below 1e-8) from each of three starts (split at the 25% and 75% quantiles, and the affine nutpie
# map), keeping the start with the best penalised objective, evaluated with the reference `Dk`. On 150 coordinate fits
# of stored insect populations BFGS was about 4 times faster than Nelder-Mead (2.7 s against 11.1 s for 10 coordinates of
# about 4000 particles, plus about 4 s of compilation once per process) and at least as accurate: Nelder-Mead stalled
# with ζ at its lower bound in a few fits (up to 0.5% worse). The three starts were chosen with Nelder-Mead (they reached
# the best of five starts with two restarts each to within 6e-8). A start whose optimisation fails is skipped; if all do,
# the equal-slope map is used. The fit uses no threshold: `cut` only sets the split for the diagnostic part-wise numbers
# returned with the fit (n_lo, n_hi, part centres and scales, flags). f.D is the unpenalised D at the penalised optimum,
# f.D_pen the penalised objective.
function fit_twoslope(θ::AbstractVector, α::AbstractVector; cut=median(θ), ζ0=1.0, λ=DEFAULT_λN / length(θ), ζ_min=0.2)
    s_glob = nutpie_scale(θ, α)
    starts = [:q25 => split_start(θ, α, quantile(θ, 0.25); ζ0).m, :q75 => split_start(θ, α, quantile(θ, 0.75); ζ0).m,
        :affine => TwoSlope(median(θ), s_glob, s_glob, ζ0)]
    obj(p) = (v = Dk(to_map(p, ζ_min), θ, α) + λ * (p[3] - p[2])^2; isfinite(v) ? v : Inf)
    best = nothing
    for (name, m0) in starts
        res = try
            optimize(p -> penalised_D(p, θ, α, λ, ζ_min), from_map(m0, ζ_min), BFGS(),
                Optim.Options(iterations=1000, g_tol=1e-8); autodiff=:forward)
        catch
            continue
        end
        D_pen = obj(Optim.minimizer(res))
        if isfinite(D_pen) && (isnothing(best) || D_pen < best.D_pen)
            m = to_map(Optim.minimizer(res), ζ_min)
            best = (D=Dk(m, θ, α), D_pen=D_pen, m=m, start=name, D_start=obj(from_map(m0, ζ_min)))
        end
    end
    if isnothing(best)
        m = equal_slope_map(θ, α)
        best = (D=Dk(m, θ, α), D_pen=Dk(m, θ, α), m=m, start=:equal_slope_fallback, D_start=NaN)
    end

    # diagnostics relative to the split at `cut`
    parts = split_start(θ, α, cut; ζ0)
    lo, hi = θ .<= cut, θ .> cut
    n_lo, n_hi, mlo, mhi, slo, shi = parts.n_lo, parts.n_hi, parts.m_lo, parts.m_hi, parts.m.slo, parts.m.shi
    m = best.m
    # flags: ζ at its lower bound; θ* within 2 local sds (part-wise Fisher scale) of either part's centre
    centres = filter(c -> isfinite(c.m), [(m=mlo, s=slo), (m=mhi, s=shi)])
    flag_ζ = m.ζ <= ζ_min + 1e-2
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

# Backstop for a sampler: an asymptote far from the slopes the particles see (beyond max_ratio either way) means a slope
# is not pinned down by data, and F' would grow or vanish in the tails. With the default penalty this did not occur on
# the stored insect populations. Returns `m`, or the equal-slope map with a reason.
function check_asymptotes(m::TwoSlope, u, θ, α; max_ratio=10.0)
    Fp = dfmap.(Ref(m), u)
    ok = isfinite(m.ζ) && max(m.slo, m.shi) <= max_ratio * maximum(Fp) && min(m.slo, m.shi) >= minimum(Fp) / max_ratio
    return ok ? (m, "") : (equal_slope_map(θ, α), "asymptote more than $(max_ratio)x from the local slopes")
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
    return LogDensityProblems.logdensity(t.target, θ) + logjac(t.maps, u)
end
