# Per-coordinate two-slope maps for NUTS on a transformed target (thesis appendix "Fisher divergence of a transformed
# target"). θ_k = F_k(u_k), where F is the piecewise-affine map θ0 + s̄ u + (Δs/2)|u| (slope s_low for u < 0, s_high for
# u > 0) with |u| = ζ|u/ζ| smoothed to ζ log cosh(u/ζ):
#   F(u)  = θ0 + s̄ u + (Δs/2) ζ log cosh(u/ζ),    s̄ = (s_low + s_high)/2, Δs = s_high - s_low,
#   F'(u) = s̄ + (Δs/2) tanh(u/ζ),
# so θ0 = F(0) and ζ is the half-width of the bend. Since log cosh x = |x| - g(|x|) with g(x) = log 2 - log(1 + e^{-2x}),
#   F(u) = θ0 + s_± u - (Δs/2) ζ g(|u|/ζ),    s_± = s_low for u < 0, s_high for u ≥ 0,
# which is how it is evaluated: the two terms have the same sign on the shallow side, and the correction is at most half
# the first term on the steep side, so F is accurate to a few ulps for any ζ and slope ratio. (The log cosh form loses
# about log10(s_high/s_low) digits far out on the shallow side, where s̄u and (Δs/2)|u| cancel; the softplus form
# θ* + (ζ/2)[s_high softplus(2u/ζ) - s_low softplus(-2u/ζ)] is a difference of two terms of size ζ s, and for ζ ~ 1e23
# every particle rounded to the same u.) The map is fitted to a population (θ_n, α_n =
# ∇ log π(θ_n)) by minimising the sample Fisher divergence of the pull-back to a standard normal with a free centre,
# D = Var_n(F'(u_n) α_n + (log F')'(u_n) + u_n), u_n = F^{-1}(θ_n).

using ForwardDiff, LogDensityProblems, LogExpFunctions, Optim, Statistics

struct TwoSlope{T<:Real}
    θ0::T    # F(0), the centre of the bend
    slo::T
    shi::T
    ζ::T     # half-width of the bend
end

# g(x) = |x| - log cosh x for x ≥ 0, accurate relative to its value (≈ x near 0, → log 2)
bend_g(x) = -log1p(expm1(-2x) / 2)

fmap(m::TwoSlope, u) = m.θ0 + (u < 0 ? m.slo : m.shi) * u - (m.shi - m.slo) / 2 * m.ζ * bend_g(abs(u) / m.ζ)
# s̄ + (Δs/2) tanh(u/ζ), written as a weighted mean of the slopes so that a small slope keeps its relative accuracy
dfmap(m::TwoSlope, u) = m.slo * logistic(-2u / m.ζ) + m.shi * logistic(2u / m.ζ)
# (log F')' = (Δs/2) sech²(u/ζ) / (ζ F')
dlogdfmap(m::TwoSlope, u) = 2 * (m.shi - m.slo) * logistic(2u / m.ζ) * logistic(-2u / m.ζ) / (m.ζ * dfmap(m, u))

# F^{-1}(θ) by Newton's method from the inverse of the nearer asymptote, θ* + s_± u with θ* = θ0 - (Δs/2) ζ log 2. F is
# increasing and convex (s_high > s_low; concave otherwise, by symmetry) and lies above both asymptotes, so the start is
# not left of the root and the iterates decrease monotonically to it. On 364 fitted maps and stress cases (slope ratio
# up to 1e6, ζ up to 5e23) this took at most 9 iterations, against up to 54 for the bracketed Newton/bisection used before.
function finv(m::TwoSlope, θ; rtol=1e-12, maxit=100)
    θs = m.θ0 - (m.shi - m.slo) / 2 * m.ζ * log(2)
    u = θ > θs ? (θ - θs) / m.shi : (θ - θs) / m.slo
    tol = rtol * max(1.0, abs(θ))
    for _ in 1:maxit
        g = fmap(m, u) - θ
        abs(g) <= tol && return u
        du = g / dfmap(m, u)
        u -= du
        abs(du) <= 4eps(abs(u)) && return u
    end
    return u
end

to_theta(maps, u) = [fmap(m, x) for (m, x) in zip(maps, u)]
to_latent(maps, θ) = [finv(m, x) for (m, x) in zip(maps, θ)]
logjac(maps, u) = sum(log(dfmap(m, x)) for (m, x) in zip(maps, u))

# Parameter vector p = (θ0, log s_low, log s_high, log(ζ - ζ_min)).
to_map(p, ζ_min) = TwoSlope(p[1], exp(p[2]), exp(p[3]), ζ_min + exp(p[4]))
from_map(m::TwoSlope, ζ_min) = [m.θ0, log(m.slo), log(m.shi), log(m.ζ - ζ_min)]

popvar(x) = var(x; corrected=false)

# D for the map m (Float64 or with dual-number parameters). u_n is found by Newton on Float64 values (`finv`), then one
# Newton step with the parameters of m, which at the solution gives the exact derivative du/dp = -(∂F/∂p) / F'(u). The
# variance is computed in two passes (the one-pass formula loses all precision when the terms are large).
function fisher_D(m::TwoSlope, θ, α)
    v = ForwardDiff.value
    mv = TwoSlope(v(m.θ0), v(m.slo), v(m.shi), v(m.ζ))
    xs = map(eachindex(θ)) do i
        u0 = finv(mv, θ[i])
        u = u0 - (fmap(m, u0) - θ[i]) / dfmap(m, u0)
        dfmap(m, u) * α[i] + dlogdfmap(m, u) + u
    end
    μ = sum(xs) / length(xs)
    return sum(x -> (x - μ)^2, xs) / length(xs)
end

# Affine nutpie map θ = m + s u (the centre does not enter D): D = Var(s α + θ/s), minimised at s^2 = sqrt(Var θ / Var α).
# It is also the minimiser of D over two-slope maps with s_low = s_high.
nutpie_scale(θ, α) = (popvar(θ) / max(popvar(α), eps()))^(1 / 4)
equal_slope_map(θ, α; ζ0=2.0) = (s = nutpie_scale(θ, α); TwoSlope(median(θ), s, s, ζ0))   # F(u) = median + s u

# Penalty on the log slope ratio, λ (log s_high - log s_low)^2, added to D. D is a population average: a part holding a
# fraction w_j of the particles, roughly Gaussian in latent space with its slope off by a factor e^δ_j, adds about
# w_j (2 sinh δ_j)^2 ≈ 4 w_j δ_j^2. Forcing a log slope ratio L to equal slopes then costs about 4 w_lo w_hi L^2, so the
# penalised log ratio is about L · 4 w_lo w_hi / (4 w_lo w_hi + λ). With λ = 4 n0 / N and one small part this is
# L · n_lo / (n_lo + n0): n0 is roughly the number of particles at which a moderate log ratio is shrunk halfway. Large,
# well-supported ratios shrink much less (2 sinh δ grows exponentially), while a slope with no particles behind it
# costs D nothing and is pulled to the other one, which removes the ridge of unidentified fits (e.g. s_low = 4e7 with no
# particle below the bend). Default λ = 100 / N (n0 = 25): on stored insect populations this resolved every degenerate
# fit while changing well-supported log ratios by about 0.05 (median) and D by under 0.1 % (median).
const DEFAULT_λN = 100.0   # λ = DEFAULT_λN / N

# Penalised objective at p = (θ0, log s_low, log s_high, log(ζ - ζ_min)), for ForwardDiff; Inf where it is not finite.
function penalised_D(p::AbstractVector{S}, θ, α, λ, ζ_min) where S
    val = fisher_D(to_map(p, ζ_min), θ, α) + λ * (p[3] - p[2])^2
    return isfinite(ForwardDiff.value(val)) ? val : S(Inf)
end

# Split start at `cut`: part-wise nutpie scale and centre (Seyboldt et al., Thm 2.2), ζ = ζ0, and the intersection of
# the asymptotes, θ0 - (ζ/2) log 2 (s_high - s_low), placed d local scales from the centre of the narrower part. A part
# with fewer than 20 particles takes the other part's scale.
function split_start(θ, α, cut; ζ0=2.0)
    s_glob = nutpie_scale(θ, α)
    lo, hi = θ .<= cut, θ .> cut
    slo = count(lo) >= 20 ? nutpie_scale(θ[lo], α[lo]) : NaN
    shi = count(hi) >= 20 ? nutpie_scale(θ[hi], α[hi]) : NaN
    isnan(slo) && (slo = isnan(shi) ? s_glob : shi)
    isnan(shi) && (shi = slo)
    mlo = count(lo) >= 2 ? mean(θ[lo]) + slo^2 * mean(α[lo]) : NaN
    mhi = count(hi) >= 2 ? mean(θ[hi]) + shi^2 * mean(α[hi]) : NaN
    d = 3 + ζ0 / 2 * abs(log(slo / shi))
    θs = shi < slo ? mhi - shi * d : mlo + slo * d
    isfinite(θs) || (θs = median(θ))
    return TwoSlope(θs + ζ0 / 2 * log(2) * (shi - slo), slo, shi, ζ0)
end

# Fit a two-slope map to one coordinate: one BFGS run (ForwardDiff gradients of `penalised_D`, stopping when the largest
# gradient component is below 1e-8) from each of three starts (split at the 25% and 75% quantiles, and the affine nutpie
# map), keeping the start with the lowest penalised objective. A start whose optimisation fails is skipped; if all do,
# the equal-slope map is used. Returns the map, the unpenalised D at it, the penalised objective and the start used.
function fit_twoslope(θ::AbstractVector, α::AbstractVector; ζ0=2.0, λ=DEFAULT_λN / length(θ), ζ_min=0.4)
    s_glob = nutpie_scale(θ, α)
    starts = [:q25 => split_start(θ, α, quantile(θ, 0.25); ζ0), :q75 => split_start(θ, α, quantile(θ, 0.75); ζ0),
        :affine => TwoSlope(median(θ), s_glob, s_glob, ζ0)]
    best = nothing
    for (name, m0) in starts
        res = try
            optimize(p -> penalised_D(p, θ, α, λ, ζ_min), from_map(m0, ζ_min), BFGS(),
                Optim.Options(iterations=1000, g_tol=1e-8); autodiff=:forward)
        catch
            continue
        end
        p = Optim.minimizer(res)
        D_pen = penalised_D(p, θ, α, λ, ζ_min)
        if isfinite(D_pen) && (isnothing(best) || D_pen < best.D_pen)
            best = (m=to_map(p, ζ_min), D_pen=D_pen, start=name)
        end
    end
    if isnothing(best)
        m = equal_slope_map(θ, α; ζ0)
        return (m=m, D=fisher_D(m, θ, α), D_pen=fisher_D(m, θ, α), start=:equal_slope_fallback)
    end
    return (m=best.m, D=fisher_D(best.m, θ, α), D_pen=best.D_pen, start=best.start)
end

# Backstop for a sampler. Falls back to the equal-slope map, with a reason, if
#   - an asymptote is far from the slopes the particles see (beyond `max_ratio` either way): that slope is not pinned
#     down by data, and F' would grow or vanish in the tails (with the default penalty this did not occur on the stored
#     insect populations), or
#   - the map does not invert the particles to within `rtol` (relative to max |θ|).
function check_map(m::TwoSlope, θ, α; max_ratio=10.0, rtol=1e-8)
    u = finv.(Ref(m), θ)
    Fp = dfmap.(Ref(m), u)
    if !(isfinite(m.ζ) && max(m.slo, m.shi) <= max_ratio * maximum(Fp) && min(m.slo, m.shi) >= minimum(Fp) / max_ratio)
        return equal_slope_map(θ, α), "asymptote more than $(max_ratio)x from the local slopes"
    end
    if maximum(abs.(fmap.(Ref(m), u) .- θ)) > rtol * max(1.0, maximum(abs, θ))
        return equal_slope_map(θ, α), "map does not invert the particles"
    end
    return m, ""
end

# Target in u-space: log π(F(u)) + Σ_k log F_k'(u_k). Exact change of variables for any fixed set of bijective maps.
struct TransformedTarget{T}
    target::T
    maps::Vector{TwoSlope{Float64}}
end
LogDensityProblems.dimension(t::TransformedTarget) = LogDensityProblems.dimension(t.target)
LogDensityProblems.capabilities(::Type{<:TransformedTarget}) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(t::TransformedTarget, u)
    θ = [fmap(m, x) for (m, x) in zip(t.maps, u)]
    return LogDensityProblems.logdensity(t.target, θ) + logjac(t.maps, u)
end
