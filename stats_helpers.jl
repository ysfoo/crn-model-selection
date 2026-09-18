using LinearAlgebra, LogExpFunctions, MCMCChains
using Distributions, QuadGK, Random, StatsBase


# Effective sample size
function compute_ess(logws)
    logws[.!isfinite.(logws)] .= -Inf
    return exp(2logsumexp(logws) - logsumexp(2 .* logws))
end


# Sample categorical distribution using {i/n}_{i=0}^{n-1} instead of drawing from Unif(0, 1)
function stratified_sampling(ws, n=length(ws); rng=Random.default_rng())
    edges = cumsum(ws ./ sum(ws))
    edges[end] = 1.0
    u = (rand(rng, n) .+ (0:(n-1))) ./ n
    idxs = searchsortedfirst.(Ref(edges), u)
    return idxs
end


# Bisection search for a monotonic function
function bisection_search(trg, func, lo, hi; tol=1e-10, is_increasing = func(hi) >= func(lo))
    (is_increasing == (func(hi) < trg)) && return hi
    (is_increasing == (func(lo) > trg)) && return lo    
        
    left, right = lo, hi
    while right - left > tol
        mid = (left + right) / 2
        val = func(mid)
        if (val <= trg) == is_increasing
            left = mid
        else
            right = mid
        end
    end
    
    return right
end

# Search based on intermediate value theorem
function IVT_search(trg, func, lo, hi; tol=1e-10)    
    left, right = lo, hi
    val_left, val_right = func(left), func(right)
    @assert (func(hi) - trg) * (func(lo) - trg) <= 0 "trg is not in the range of func"
    while right - left > tol
        mid = (left + right) / 2
        val = func(mid)
        if (val - trg) * (val_left - trg) > 0
            left = mid
            val_left = val
        else
            right = mid
            val_right = val
        end
    end    
    return right
end


# Returns indices corresponding to largest log weights that covers 1-α of the total
function select_top(logws, α; n_max=length(logws))
    lse = logsumexp(logws)
    ord = sortperm(logws, rev=true)
    tmp = -Inf
    for (upto, idx) in enumerate(ord)
        tmp = logaddexp(logws[idx], tmp)
        if tmp >= lse + log1p(-α)
            return ord[1:min(n_max,upto)]
        end
    end
    return ord[1:n_max]
end


# Return `top` largest entries of `v`
function sortview(v; top=5)
    top = min(top, length(v))
    return sort(collect(enumerate(v)), by=last, rev=true)[1:top]
end


# Turing sometimes returns Chains with :logjoint or :lp for log ldproberior
function extract_logp(chn::Chains)
    sym = :logjoint ∈ chn.name_map.internals ? :logjoint : :lp
    return collect(vec(chn[sym]))
end


# Approximate upper quantile of Kendall correlation coefficient given sample size and significance level
function kendall_qt(n, sig, ties=Int64[])
    sum1 = sum(t*(t-1) for t in ties if t > 1; init=0)
    sum2 = sum(t*(t-1)*(2t+5) for t in ties if t > 1; init=0)
    opairs = n*(n-1)
    v = (2 * (opairs*(2n + 5) - sum2)) / (9*opairs*(opairs - sum1))
    q = quantile(Normal(), 1 - sig)
    return q*sqrt(v)
end


# function kendall_qt_old(n, sig)
#     v = 2 * (2*n + 5) / (9 * n * (n - 1))
#     q = quantile(Normal(), 1 - sig)
#     return q*sqrt(v)
# end


## LogDensityProblem for Distribution
using Distributions, LogDensityProblems, LogDensityProblemsAD
import ForwardDiff

LogDensityProblems.capabilities(::Type{<:Distribution}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(dist::Distribution)      = length(dist)
LogDensityProblems.logdensity(dist::Distribution, x)  = logpdf(dist, x)

make_ldprob(dist::Distribution) = ADgradient(:ForwardDiff, dist)


## LogDensityProblem for generic logpdf
struct BasicLDP{F}
    f::F
    d::Int
end

LogDensityProblems.capabilities(::Type{<:BasicLDP}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(ldp::BasicLDP)      = ldp.d
LogDensityProblems.logdensity(ldp::BasicLDP, x)  = ldp.f(x)

customcopy(ldp::BasicLDP) = BasicLDP(ldp.f, ldp.d)

## LogDensityProblem for tuple of distributions
struct PriorLogDensity{V<:AbstractVector{<:UnivariateDistribution}}
    dists::V
end

LogDensityProblems.logdensity(p::PriorLogDensity, x) = sum(logpdf(p.dists[i], x[i]) for i in eachindex(p.dists))
LogDensityProblems.dimension(p::PriorLogDensity) = length(p.dists)
LogDensityProblems.capabilities(::Type{<:PriorLogDensity}) = LogDensityProblems.LogDensityOrder{0}()

customcopy(p::PriorLogDensity) = PriorLogDensity(p.dists)


function compute_BC(d1, d2, lo=-Inf, hi=Inf)
    f(x) = exp((logpdf(d1, x) + logpdf(d2, x))/2)
    return quadgk(f, lo, hi)[1]
end
