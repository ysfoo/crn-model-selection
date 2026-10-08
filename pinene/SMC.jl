include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using DiffResults, Optim, StatsFuns
using BSplineKit, SparseArrays
import RegularizationTools

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

LinearAlgebra.BLAS.set_num_threads(1)

# Single dataset (see `setup.jl`). All `n_rx` candidate reactions have a spike-and-slab prior (none are fixed), 
# one noise parameter per species, logit-transformed inclusion probability for each candidate reaction.
n_θ = n_rx + n_species + 1;
std_idxs = 1:0; # ODE parameters with standard prior (none)
ss_idxs = 1:n_rx; # ODE parameters with spike-and-slab prior
noise_idxs = (n_rx+1):(n_rx+n_species);
p_idx = n_θ
n_ss = length(ss_idxs);

param_idxs = collect(ss_idxs);
u0_idxs = map((x)->parameter_index(base_oprob, Initial(x)).idx, unknowns(model));

### Parameter scaling, from `src/inference.jl` of crn-inference
# Smooth each species' time series with a regularised cubic B-spline, then set the scale factor of each
# reaction to (largest range of the species' derivative per unit stoichiometry) / (range of the rate law
# along the smoothed trajectories, with all rate constants equal to 1).

# u      : Univariate time series
# t      : Timepoints corresponding to the time series
# d      : Order of derivative to penalise
# t_itp  : Interpolation points to penalise d-th order derivative at
# alg    : Algorithm for determining smoothing hyperparameter, see RegularizationSmooth from DataInterpolations.jl
function smooth_data(u, t; d=2, t_itp=range(extrema(t)..., 50), alg=:L_curve)
    basis = BSplineBasis(BSplineKit.BSplineOrder(4), copy(t_itp));
    B = collocation_matrix(basis, t, BSplineKit.Derivative(0), SparseMatrixCSC{Float64});
    D = collocation_matrix(basis, t_itp, BSplineKit.Derivative(d), SparseMatrixCSC{Float64});
    Ψ = RegularizationTools.setupRegularizationProblem(B, collect(D))
    scale = sqrt(sum(abs2, Ψ.Ā) / length(Ψ.Ā) * length(t))
    sol = RegularizationTools.solve(Ψ, u; alg=alg, λ₁=1e-4scale, λ₂=scale)
    β = sol.x
    return (basis, β)
end

function eval_spline(basis, β, t, d=0)
    B = collocation_matrix(basis, t, BSplineKit.Derivative(d), SparseMatrixCSC{Float64});
    return B*β
end

function get_scale_fcts(smooth_resvec, t, species_vec, rx_vec, k)
    est_derivs = Dict(
        x => eval_spline(basis, β, t, 1)
        for (x, (basis, β)) in zip(species_vec, smooth_resvec)
    );
    est_trajs = [eval_spline(basis, β, t) for (basis, β) in smooth_resvec];

    rates_vec = substitute.(oderatelaw.(rx_vec, combinatoric_ratelaw=false), Ref([k => ones(length(k))]));
    n_itp = length(est_trajs[1]);
    return [
        begin
            itp_rates = substitute.(
                Ref(rates), [
                    Dict(x => traj[i] for (x, traj) in zip(species_vec, est_trajs))
                    for i in 1:n_itp]);
            rate_min, rate_max = extrema(itp_rates);
            deriv_ranges = [
                begin
                    deriv_min, deriv_max = extrema(est_derivs[x] ./ stoich);
                    deriv_max - deriv_min
                end for (x, stoich) in rx.netstoich]
                eval(Symbolics.toexpr(maximum(deriv_ranges) / (rate_max - rate_min)))
        end for (rx, rates) in zip(rx_vec, rates_vec)
    ]
end

# `scale_factors[r]` maps the rate constant of reaction `r` at θ[r] = 0 to the ODE parameter. It is mutated
# in place to revise the scaling; `ode_params!` reads it at call time.
smooth_resvec = [smooth_data(data[:,j], t_obs) for j in 1:n_species];
scale_factors = Float64.(get_scale_fcts(smooth_resvec, t_obs, species_vec, rx_vec, only(parameters(model))));

ode_params!(buf, θ) = for i in 1:n_rx
    buf[i] = exp10(θ[i]) * scale_factors[i]
end;

μ_noise, σ_noise = 0., 1.;
μ_slab, σ_slab = 0., 2.;
μ_spike = -16.; σ_spike = σ_slab;
thres = 0.5 * (μ_spike + μ_slab);
μ0 = -8.;

noise_prior = Normal(μ_noise, σ_noise)
slab_prior = Normal(μ_slab, σ_slab)
spike_prior = Normal(μ_spike, σ_spike)
p_prior = Logistic()

# Prior on non spike-and-slab parameters
function add_logprior(θ)
    lp = logpdf(p_prior, θ[p_idx])
    @inbounds for i in noise_idxs
        lp += logpdf(noise_prior, θ[i])
    end
    return lp
end

ss_bimodal_logprior(θ) = begin
    logp, log1mp = loglogistic(θ[p_idx]), log1mlogistic(θ[p_idx])
    sum(
        begin
            x = θ[d]
            logaddexp(
                normlogpdf(μ_spike, σ_spike, x) + log1mp, 
                normlogpdf(μ_slab, σ_slab, x) + logp
            )
        end for d in ss_idxs
    )
end

final_logprior_func(θ) = ss_bimodal_logprior(θ) + add_logprior(θ)

function make_ldp(logprior_func::LPF, γ::Float64, loglike_offset::Function=(θ)->0.) where LPF
    # The closures read the globals through local copies: reading non-const globals makes them type-unstable, which
    # with dual numbers multiplies the allocations per gradient. `scale_factors` is the same array, so revising it in
    # place still takes effect.
    loglike_func = let n_species = n_species, n_obs = n_obs, data = data, noise_idxs = noise_idxs
        function (sol, θ::AbstractVector{T}) where T
            ll = T(-n_species*n_obs*log(2π)/2)
            for i in 1:n_obs
                for j in 1:n_species
                    y_obs = data[i,j]
                    y_pred = sol.u[i][j]
                    s = exp10(θ[noise_idxs[j]])
                    ll -= ((y_obs - y_pred) / s)^2 / 2 + log(s)
                end
            end
            return (ll + loglike_offset(θ)) * γ
        end
    end
    ldp_params! = let n_rx = n_rx, scale_factors = scale_factors
        (buf, θ) -> for i in 1:n_rx
            buf[i] = exp10(θ[i]) * scale_factors[i]
        end
    end
    return OrdinaryDiffEqLDP(
        base_oprob, param_idxs, ldp_params!, θ -> logprior_func(θ, γ), loglike_func, n_θ;
        solver = AutoTsit5(Rodas5P()),
        solve_kwargs = (saveat=t_obs, verbose=false, abstol=1e-6, reltol=1e-6),
    )
end

function run_10(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 1 ? 10 : 0
end

function run_20(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 1 ? 20 : 0
end

function repeat_5(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    if npass < 1
        return 5
    end
    nzeros = mean(p.info.sum_sqjdist == 0. for p in particles)
    return nzeros > 0.01 ? 5 : 0
end

function run_5_3(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 3 ? 5 : 0
end

function partition_by_thres(states::AbstractVector, thres::Float64; ss_idxs=ss_idxs)
    classes = Dict{BitVector, Vector{Int}}()
    for (i, θ) in enumerate(states)
        key = BitVector(θ[j] > thres for j in ss_idxs)
        push!(get!(classes, key, Int[]), i)
    end
    return classes
end


## Nutpie-style mass matrix
function compute_invmass(particles, target, npass)
    logp_func = Base.Fix1(LogDensityProblems.logdensity, target)

    x1 = particles[1].state
    d, n = length(x1), length(particles)
    T = eltype(x1)

    X = Matrix{T}(undef, d, n)
    G = Matrix{T}(undef, d, n)
    logp = Vector{T}(undef, n)

    result = DiffResults.GradientResult(x1)
    for (i, p) in enumerate(particles)
        X[:, i] .= p.state
        result = ForwardDiff.gradient!(result, logp_func, p.state)  # value + grad in one pass
        logp[i] = DiffResults.value(result)
        G[:, i] .= DiffResults.gradient(result)
    end

    keep = logp .>= quantile(logp, 0.01)   # drop bottom 1% by log target density
    var_x     = vec(var(view(X, :, keep); dims = 2))
    var_alpha = vec(var(view(G, :, keep); dims = 2))
    invmass = sqrt.(var_x ./ var_alpha)
    # display([sqrt.(var_x) sqrt.(var_alpha) invmass])
    return (invmass = invmass,)   # inverse mass diagonal
end

## Population info for `nuts_invmass_move`: nutpie-style mass matrix plus the pool of step sizes
# Use previous pool of step sizes weighted by how close the acceptance rate of the ancestor's last move was to the target.
acc_weight(a; p=4, q=1) = isfinite(a) ? clamp(a, 0., 1.)^p * (1 - clamp(a, 0., 1.))^q : 0.

# Particles without acceptance info (first iteration, no usable weights) give a uniform pool with a wide jitter.
# Particles with `curr_acc_rate < min_acc_rate` can additionally be excluded outright (disabled by default); if 
# that would exclude everyone, the filter is ignored.
function compute_FT_info(
    particles, target, npass; jitter=0.2, init_jitter=0.2, min_acc_rate=0.0,
    weight_func=acc_weight,
)
    stepsizes = [p.stepsize for p in particles]
    accs = [
        hasproperty(p.info, :curr_acc_rate) ? p.info.curr_acc_rate : NaN
        for p in particles
    ]
    valid_step = isfinite.(stepsizes) .& (stepsizes .> 0)
    acc_ok = .!(accs .< min_acc_rate)
    any(valid_step .& acc_ok) && (valid_step = valid_step .& acc_ok)
    ws = [valid_step[i] ? weight_func(accs[i]) : 0. for i in eachindex(particles)]
    if sum(ws) > 0
        used_jitter = jitter
    else
        ws = Float64.(valid_step)
        used_jitter = init_jitter
    end
    cum_probs = cumsum(ws ./ sum(ws))
    cum_probs[end] = 1.0
    return merge(
        compute_invmass(particles, target, npass),
        (stepsizes = stepsizes, cum_probs = cum_probs, jitter = used_jitter),
    )
end


## Population info for `nuts_invmass_move` with a regression-based step size.
# Regress acceptance rates from previous population against log step size. Fit model for E[a | log ε] using quasi-binomial
# regression a(ε) = logistic(β0 + β1 (log ε - mean log ε)) by Fisher scoring. Returns the step size solution to a(ε)=`acc_target`.
# `nuts_invmass_move` multiplies this stepsize by exp(jitter * randn()).
# - `weights`: if `use_ngrad`, particles are weighted by `info.curr_n_grad`, default off.
# - `max_dlog`: cap on |log new ε - mean log ε| (default: no cap), to limit extrapolation beyond the spread of the step sizes.
# - Fallback (no acceptance information, too few particles, no spread in step sizes, or a non-negative slope): the
#   geometric mean of the particles' step sizes is used, i.e. the step size is left unchanged.
function fit_acc_stepsize(stepsizes, accs; acc_target=0.8, weights=nothing, min_obs=20, max_dlog=Inf)
    ok = findall(eachindex(stepsizes)) do i
        isfinite(stepsizes[i]) && stepsizes[i] > 0 && isfinite(accs[i]) &&
            (isnothing(weights) || (isfinite(weights[i]) && weights[i] > 0))
    end
    length(ok) < min_obs && return nothing
    x = log.(stepsizes[ok]); y = clamp.(accs[ok], 0., 1.)
    w = isnothing(weights) ? ones(length(ok)) : Float64.(weights[ok])
    x̄ = sum(w .* x) / sum(w); xc = x .- x̄
    maximum(abs, xc) > 1e-8 || return nothing

    β = [logit(clamp(sum(w .* y) / sum(w), 0.01, 0.99)), 0.]
    # Fisher scoring
    for _ in 1:100
        η = β[1] .+ β[2] .* xc; μ = logistic.(η)
        v = max.(μ .* (1 .- μ), 1e-6)
        z = η .+ (y .- μ) ./ v                  # working response
        W = w .* v
        S = [sum(W) sum(W .* xc); sum(W .* xc) sum(W .* xc .^ 2)]
        β_new = S \ [sum(W .* z), sum(W .* xc .* z)]
        all(isfinite, β_new) || return nothing
        converged = maximum(abs, β_new .- β) < 1e-8
        β = β_new
        converged && break
    end
    β[2] < 0 || return nothing
    dlog = clamp((logit(acc_target) - β[1]) / β[2], -max_dlog, max_dlog)
    return (stepsize = exp(x̄ + dlog), slope = β[2], centre = exp(x̄))
end

function compute_QB_info(
    particles, target, npass; acc_target=0.8, jitter=0.2, max_dlog=Inf, use_ngrad=false, min_obs=20,
)
    stepsizes = [p.stepsize for p in particles]
    valid = filter(s -> isfinite(s) && s > 0, stepsizes)
    stepsize = exp(mean(log.(valid))) # nominal stepsize is previous stepsize

    accs = [hasproperty(p.info, :curr_acc_rate) ? p.info.curr_acc_rate : NaN for p in particles]
    weights = use_ngrad ? [hasproperty(p.info, :curr_n_grad) ? p.info.curr_n_grad : NaN for p in particles] : nothing
    fit = fit_acc_stepsize(stepsizes, accs; acc_target, weights, min_obs, max_dlog)
    if npass > 1
        stepsize *= 0.95 # decrease if multiple NUTS rounds
        if !isnothing(fit)
            stepsize = min(stepsize, fit.stepsize) # don't increase stepsize
        end
    elseif !isnothing(fit)
        stepsize = fit.stepsize
    end
    return merge(
        compute_invmass(particles, target, npass),
        (stepsize = stepsize, jitter = jitter),
    )
end