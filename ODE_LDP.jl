using Accessors, LogDensityProblems
using Catalyst, ModelingToolkit, OrdinaryDiffEq
using ForwardDiff, DiffResults
using PreallocationTools: DiffCache, get_tmp
using SymbolicIndexingInterface: parameter_index

# ForwardDiff tag for the gradient of an `OrdinaryDiffEqLDP`. A `nothing` tag cannot be ordered against the tag that
# OrdinaryDiffEq uses for the Jacobian of an implicit solver (e.g. the Rodas5P part of AutoTsit5(Rodas5P())).
struct OrdinaryDiffEqLDPTag end
ForwardDiff.checktag(::Type{ForwardDiff.Tag{OrdinaryDiffEqLDPTag,V}}, f::F, x::AbstractArray{V}) where {F,V} = true
make_gradconfig(n_θ) = ForwardDiff.GradientConfig(
    nothing, zeros(n_θ), ForwardDiff.Chunk(n_θ), ForwardDiff.Tag(OrdinaryDiffEqLDPTag(), Float64)
)

struct OrdinaryDiffEqLDP{PR<:ODEProblem, LP, LL, UP, UU, ALG, KW, CR, CC, TC1, TC2}
    base_prob          :: PR
    n_θ                :: Int
    log_prior          :: LP
    log_likelihood     :: LL
    update_param!      :: UP
    update_u0!         :: UU
    param_idxs         :: Vector{Int}
    u0_idxs            :: Vector{Int}
    solver             :: ALG
    solve_kwargs       :: KW
    tunable_diffcache  :: TC1
    initials_diffcache :: TC2
    diffresult         :: CR
    gradconfig         :: CC
end

function OrdinaryDiffEqLDP(
    oprob          :: ODEProblem,
    param_idxs     :: Vector{Int},
    update_param!  :: UP,
    log_prior      :: LP,
    log_likelihood :: LL,
    n_θ            :: Int;
    solver         :: ALG = AutoVern7(Rodas5P()),
    solve_kwargs   :: NamedTuple = (;),
    u0_idxs        :: Vector{Int} = Int[],
    update_u0!     :: UU = (buf, θ) -> nothing,
) where {LP, LL, UP, UU, ALG}
    tunable_diffcache  = DiffCache(copy(oprob.p.tunable))
    initials_diffcache = DiffCache(copy(oprob.p.initials))
    diffresult = DiffResults.GradientResult(zeros(n_θ))
    gradconfig = make_gradconfig(n_θ)
    return OrdinaryDiffEqLDP(
        oprob, n_θ, log_prior, log_likelihood,
        update_param!, update_u0!,
        param_idxs, u0_idxs,
        solver, solve_kwargs,
        tunable_diffcache, initials_diffcache, diffresult, gradconfig
    )
end

function customcopy(ldprob::OrdinaryDiffEqLDP)
    return setproperties(ldprob, (
        base_prob          = deepcopy(ldprob.base_prob), 
        tunable_diffcache  = DiffCache(copy(ldprob.base_prob.p.tunable)),
        initials_diffcache = DiffCache(copy(ldprob.base_prob.p.initials)),
        diffresult         = DiffResults.GradientResult(zeros(ldprob.n_θ)),
        gradconfig         = make_gradconfig(ldprob.n_θ),
    ))
end

function _logdensity(ldprob::OrdinaryDiffEqLDP, θ::AbstractVector{T}; show=false) where T
    lp = ldprob.log_prior(θ)
    isfinite(lp) || return T(-Inf)

    tunable_buf = get_tmp(ldprob.tunable_diffcache, θ)
    copyto!(tunable_buf, ldprob.base_prob.p.tunable)
    ldprob.update_param!(view(tunable_buf, ldprob.param_idxs), θ)

    if isempty(ldprob.u0_idxs)
        p_new = setproperties(ldprob.base_prob.p, (tunable = tunable_buf,))
    else
        initials_buf = get_tmp(ldprob.initials_diffcache, θ)
        copyto!(initials_buf, ldprob.base_prob.p.initials)
        ldprob.update_u0!(view(initials_buf, ldprob.u0_idxs), θ)
        p_new = setproperties(ldprob.base_prob.p, (tunable = tunable_buf, initials = initials_buf))
    end

    prob = remake(ldprob.base_prob; p = p_new)
    sol  = solve(prob, ldprob.solver; ldprob.solve_kwargs...)
    if show
        display(prob.ps)
        display(sol.u[end])
        f = Makie.series(stack(petab_sol(data.t).u))
        display(f)
    end

    sol.retcode === ReturnCode.Success || return T(-Inf)

    return T(lp + ldprob.log_likelihood(sol, θ))
end

LogDensityProblems.capabilities(::Type{<:OrdinaryDiffEqLDP}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(ldprob::OrdinaryDiffEqLDP)      = ldprob.n_θ
LogDensityProblems.logdensity(ldprob::OrdinaryDiffEqLDP, θ)  = _logdensity(ldprob, θ)

function LogDensityProblems.logdensity_and_gradient(ldprob::OrdinaryDiffEqLDP, θ::AbstractVector{<:Real})
    ForwardDiff.gradient!(ldprob.diffresult, Base.Fix1(_logdensity, ldprob), θ, ldprob.gradconfig)
    # Copy the gradient, as callers (e.g. AdvancedHMC phase points) keep it after the next call overwrites the buffer.
    return DiffResults.value(ldprob.diffresult), copy(DiffResults.gradient(ldprob.diffresult))
end

# function deriv2(ℓ, θ, i)
#     θi = θ[i]
#     f = t -> ℓ([j == i ? θi + t : θ[j] for j in eachindex(θ)])
#     ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), 0.0)
# end