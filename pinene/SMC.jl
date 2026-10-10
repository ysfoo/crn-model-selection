include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../process_setup.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using DiffResults, Optim, StatsFuns
using BSplineKit, SparseArrays
import RegularizationTools

# Workers for `parallel = :distributed`, thread pinning and single-threaded BLAS (see process_setup.jl)
setup_SMC_process()

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

function repeat_5_min_2(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    if npass < 2
        return 5
    end
    nzeros = mean(p.info.sum_sqjdist == 0. for p in particles)
    return nzeros > 0.01 ? 5 : 0
end

function run_5_3(particles, last_n_nuts, npass, states_before_move; verbose = 0, rng=Random.default_rng())
    return npass < 3 ? 5 : 0
end

