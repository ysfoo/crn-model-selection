# Log posterior of the insect models as an `OrdinaryDiffEqLDP` (replaces the PEtab problems). Include after `setup.jl`.
# θ holds log10 of the parameters in the order of `parameters(model)` (σ last), with priors defined on the log10 scale
# (Normal(0, 2) for ODE parameters, Normal(-1, 1) for σ), so no Jacobian is needed. This matches PEtab with
# `prior_on_linear_scale = false`, which the old scripts used.
include(joinpath(@__DIR__, "../ODE_LDP.jl"));

using Distributions, ForwardDiff, LinearAlgebra, LogDensityProblems, Optim, Random

prior_dists(n_θ) = [fill(Normal(0., 2.), n_θ - 1); Normal(-1., 1.)]

# `tol = nothing` uses the OrdinaryDiffEq default tolerances (for log density values only); pass `tol = 1e-6` where
# gradients are taken. `maxiters = 10^4` matches the PEtab default.
function make_insect_ldp(model, data; tol=nothing, solver=AutoTsit5(Rodas5P()), maxiters=10^4)
    ps = parameters(model)
    n_θ = length(ps)
    tol_kwargs = isnothing(tol) ? (;) : (abstol=tol, reltol=tol)
    oprob = ODEProblem(model, u0, (0.0, t_end), [p => 1. for p in ps]; tol_kwargs...)
    param_idxs = map((x)->parameter_index(oprob, x).idx, ps)

    ode_params!(buf, θ) = for i in 1:n_θ
        buf[i] = exp10(θ[i])
    end

    dists = prior_dists(n_θ)
    logprior_func(θ) = sum(logpdf(dist, val) for (dist, val) in zip(dists, θ))

    data_mat = hcat(data.data_E, data.data_L, data.data_A) # N x 3
    n_obs = length(data.t)
    # Local copy of the global `n_u`: reading a non-const global makes the loop type-unstable, and with dual numbers
    # this tripled the allocations per gradient.
    nu = n_u
    function loglike_func(sol, θ::AbstractVector{T}) where T
        σ = exp10(θ[n_θ])
        ll = T(-nu*n_obs*log(2π)/2)
        for i in 1:n_obs
            for j in 1:nu
                y_pred = max(0., sol.u[i][j])
                s = 0.01 + σ*y_pred
                ll -= ((data_mat[i,j] - y_pred) / s)^2 / 2 + log(s)
            end
        end
        return ll
    end

    return OrdinaryDiffEqLDP(
        oprob, param_idxs, ode_params!, logprior_func, loglike_func, n_θ;
        solver = solver,
        solve_kwargs = (saveat=data.t, verbose=false, maxiters=maxiters),
    )
end

loglike(ldp::OrdinaryDiffEqLDP, θ) = LogDensityProblems.logdensity(ldp, θ) - ldp.log_prior(θ)

# Draws from the prior until the log posterior (and, with `grad = true`, also its gradient) is finite.
function sample_feasible(rng, ldp; grad=false)
    dists = prior_dists(ldp.n_θ)
    while true
        θ = [rand(rng, dist) for dist in dists]
        val = grad ? LogDensityProblems.logdensity_and_gradient(ldp, θ)[1] : LogDensityProblems.logdensity(ldp, θ)
        isfinite(val) && return θ
    end
end

struct FailStreak <: Exception end

const DEFAULT_OPT = Optim.Options(iterations = 1000, show_trace = false, show_warnings = false,
                                  allow_f_increases = true, successive_f_tol = 3,
                                  f_reltol = 1e-8, g_tol = 1e-6, x_abstol = 0.0)

# Multistart MAP estimation from prior draws, as PEtab's `calibrate_multistart` with `sample_prior = true`: BFGS inside
# `Fminbox` (unbounded here, but its outer iterations restart BFGS when it stalls). Starts are drawn until `n_starts`
# runs are healthy, at most `max_attempts` in total.
# - Value and gradient come from one ForwardDiff solve (cached for the last x). The objective is Inf where that solve
#   fails, so the line search shortens the step. At extreme parameters the gradient solve can fail (`Unstable`, the
#   partials overflow in the error norm) where the Float64 solve succeeds; a finite value with a zero gradient there
#   made BFGS stop early.
# - A start is aborted after `max_fail_streak` consecutive failed evaluations (it is stuck where gradients fail).
# - A run is healthy if Optim reports convergence and max|gradient| at its minimiser is at most `healthy_gtol`. Runs
#   that stall where the line search cannot progress report convergence with max|gradient| of 1e2 and above, against
#   at most 4e-3 at true minima.
function fit_MAP(ldp, n_starts; rng=Random.default_rng(), options=DEFAULT_OPT,
                 max_fail_streak=20, max_attempts=10n_starts, healthy_gtol=0.1)
    cache_x = fill(NaN, ldp.n_θ)
    cache_val = Ref(NaN)
    cache_grad = zeros(ldp.n_θ)
    fail_streak = Ref(0)
    function eval!(x)
        if x != cache_x
            val, grad = LogDensityProblems.logdensity_and_gradient(ldp, x)
            copyto!(cache_x, x)
            cache_val[] = val
            copyto!(cache_grad, grad)
            fail_streak[] = isfinite(val) ? 0 : fail_streak[] + 1
            fail_streak[] >= max_fail_streak && throw(FailStreak())
        end
        return cache_val[]
    end
    f(x) = (val = eval!(x); isfinite(val) ? -val : Inf)
    g!(G, x) = (val = eval!(x); G .= isfinite(val) ? .-cache_grad : 0.)

    unbounded = fill(Inf, ldp.n_θ)
    xmins = Vector{Float64}[]
    fmins = Float64[]
    n_attempts = 0
    n_aborted = 0
    n_unhealthy = 0
    while length(fmins) < n_starts && n_attempts < max_attempts
        n_attempts += 1
        x0 = sample_feasible(rng, ldp; grad=true)
        fill!(cache_x, NaN)
        fail_streak[] = 0
        res = try
            optimize(f, g!, -unbounded, unbounded, x0, Fminbox(BFGS(linesearch = Optim.BackTracking())), options)
        catch e
            e isa FailStreak || rethrow(e)
            n_aborted += 1
            continue
        end
        xm = collect(Optim.minimizer(res))
        val, grad = LogDensityProblems.logdensity_and_gradient(ldp, xm)
        if Optim.converged(res) && isfinite(val) && maximum(abs, grad) <= healthy_gtol
            push!(xmins, xm)
            push!(fmins, Optim.minimum(res))
        else
            n_unhealthy += 1
        end
    end
    isempty(fmins) && error("fit_MAP: no healthy run in $n_attempts attempts")
    length(fmins) < n_starts && @warn "fit_MAP: only $(length(fmins)) healthy runs in $n_attempts attempts"
    xmin = xmins[argmin(fmins)]
    return (xmin = xmin, fmin = minimum(fmins), loglik = loglike(ldp, xmin), fmins = fmins,
            n_attempts = n_attempts, n_aborted = n_aborted, n_unhealthy = n_unhealthy)
end

# Hessian of the negative log posterior.
neglogpost_hessian(ldp, θ) = -ForwardDiff.hessian(x -> LogDensityProblems.logdensity(ldp, x), θ)
