# Calibrate the log-slope penalty of the two-slope fit (λ = 4 n0 / N, see `penalty_lambda` in fisher_maps.jl) on stored
# SMC populations. Read-only apart from one CSV.
#
# Usage (from insect/): julia --project=.. penalty_sweep.jl <data idx> <run name> <iterations...> [--variant=SMCnnn]
# Output: output/penalty_sweep/data<d>/<run name>.csv, one row per (iteration, coordinate, n0).

ps_flags = Dict(begin
        kv = split(a[3:end], "=")
        kv[1] => (length(kv) > 1 ? kv[2] : "true")
    end for a in ARGS if startswith(a, "--"))
ps_pos = [a for a in ARGS if !startswith(a, "--")]
ps_run_name, ps_iters = ps_pos[2], parse.(Int, ps_pos[3:end])
const ZETA_MIN = 0.2
const N0S = [0, 5, 20, 50, 200]
empty!(ARGS); push!(ARGS, ps_pos[1])

include(joinpath(@__DIR__, "SMC.jl"))
include(joinpath(@__DIR__, "fisher_maps.jl"))
load_variant!(joinpath(@__DIR__, get(ps_flags, "variant", split(ps_run_name, "_")[1]) * ".jl"))

all_particles, _, targetinfos, _, _ = load_SMC(joinpath(@__DIR__, "output/data$(dir_idx)/$(ps_run_name).jld2"))
out = joinpath(@__DIR__, "output/penalty_sweep/data$(dir_idx)")
mkpath(out)
const COORD_NAMES = ["λ_EL", "λ_LA", "ρ", "δ_E", "δ_L", "δ_A", "κ_E", "κ_L", "κ_A", "σ"]

cols = ["run", "data", "iter", "gamma", "coord", "coord_name", "is_ss", "n0", "lambda", "n_lo", "n_hi", "theta_star",
    "s_low", "s_high", "zeta", "logratio", "logratio_parts", "D", "D_pen", "D_unpen_fit", "D_affine",
    "local_slope_min", "local_slope_max", "asym_over_local_max", "asym_under_local_min", "n_below_bend_u",
    "n_above_bend_u", "guard_ok", "guard_reason", "start"]
rows = Vector{Vector{Any}}()
for i in ps_iters
    γ = targetinfos[i+1].γ
    target = ldp_builder(prior_path, γ)
    lp = Base.Fix1(LogDensityProblems.logdensity, target)
    ps = all_particles[i+1]
    X = reduce(hcat, [p.state for p in ps]); G = similar(X); logp = zeros(size(X, 2))
    for j in axes(X, 2)
        r = DiffResults.GradientResult(X[:, j]); r = ForwardDiff.gradient!(r, lp, X[:, j])
        logp[j] = DiffResults.value(r); G[:, j] .= DiffResults.gradient(r)
    end
    maximum(abs.(logp .- [p.logtarget for p in ps])) < 1e-6 || error("log target mismatch: wrong path or γ?")
    keep = logp .>= quantile(logp, 0.01)
    N = count(keep)
    for k in 1:size(X, 1)
        θ, α = X[k, keep], G[k, keep]
        cut = split_point(k, θ)
        lo, hi = θ .<= cut, θ .> cut
        L_parts = (count(lo) >= 20 && count(hi) >= 20) ? log(nutpie_scale(θ[hi], α[hi]) / nutpie_scale(θ[lo], α[lo])) : NaN
        D_unpen = NaN
        for n0 in N0S
            λ = penalty_lambda(n0, N)
            t = @elapsed f = fit_twoslope(k, θ, α; λ)
            n0 == 0 && (D_unpen = f.D)
            m = f.m
            Fp = dfmap.(Ref(m), f.u)
            _, reason = guard_map(f, θ, α)
            push!(rows, Any[ps_run_name, dir_idx, i, γ, k, COORD_NAMES[k], k in ss_idxs, n0, λ, f.n_lo, f.n_hi, m.θs,
                m.slo, m.shi, m.ζ, log(m.shi / m.slo), L_parts, f.D, f.D_pen, D_unpen, f.D_affine,
                minimum(Fp), maximum(Fp), max(m.slo, m.shi) / maximum(Fp), min(m.slo, m.shi) / minimum(Fp),
                count(f.u .< -2m.ζ), count(f.u .> 2m.ζ), isempty(reason), reason, f.start])
        end
        @info "iter $i $(COORD_NAMES[k])" rows[end-length(N0S)+1:end] .|> (r -> round(r[16], digits=2))
    end
    open(joinpath(out, "$(ps_run_name).csv"), "w") do io
        println(io, join(cols, ","))
        foreach(r -> println(io, join(r, ",")), rows)
    end
end
println("Output: ", joinpath(out, "$(ps_run_name).csv"))
