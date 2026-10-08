# Insect-specific helpers for the two-slope Fisher maps (the generic map code is in ../transform_maps.jl, included by
# SMC_functions.jl): safe loading of a run's variant script, the split used for each coordinate, and the older hard guard
# against degenerate fits. Expects `ZETA_MIN`, `ss_idxs` and `thres` to be defined (by the including script and SMC.jl).

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

# Parts used for the split start, the flags and the per-part latent scales.
split_point(k, θ) = k in ss_idxs ? thres : median(θ)

# Fit for coordinate k (default penalty λ = 100 / N, see `penalty_lambda` in ../transform_maps.jl).
fit_twoslope(k::Integer, θ::AbstractVector, α::AbstractVector; kwargs...) =
    fit_twoslope(θ, α, split_point(k, θ); ζ_min=ZETA_MIN, kwargs...)

# Older hard guard (superseded by the log-slope penalty plus `check_asymptotes`): fall back to the equal-slope optimum
# unless both parts (split at `thres` for spike-and-slab coordinates) have at least n_min particles, at least n_min
# particles lie on each side of θ*, and s_high / s_low is within [1/max_ratio, max_ratio].
# Example of what it guarded against: δ_A on data 4 SMC505_4k iteration 26 has 9 spike particles; without the penalty
# all particles sit at u > 35, so F'(u) ≈ s_high + s_low e^{-u/ζ} on the population and s_low (4e7) trades off against
# θ*, with D flat for s_low ≥ 1000. The map is fine on the population but F' grows towards s_low in the tail.
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
