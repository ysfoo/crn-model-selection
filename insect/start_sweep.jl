# Which Nelder-Mead starts and restarts does the two-slope fit need? For every coordinate of stored SMC populations, fit
# the penalised objective (λ = 100 / N) from several starts, recording the objective after one Nelder-Mead run and after
# each of two restarts from the run's own optimum. Read-only apart from one CSV.
#
# Usage (from insect/): julia --project=.. start_sweep.jl <data idx> <run name> <iterations...> [--variant=SMCnnn]
# Output: output/start_sweep/data<d>/<run name>.csv, one row per (iteration, coordinate, start).

ss_flags = Dict(begin
        kv = split(a[3:end], "=")
        kv[1] => (length(kv) > 1 ? kv[2] : "true")
    end for a in ARGS if startswith(a, "--"))
ss_pos = [a for a in ARGS if !startswith(a, "--")]
ss_run_name, ss_iters = ss_pos[2], parse.(Int, ss_pos[3:end])
const ZETA_MIN = 0.2
empty!(ARGS); push!(ARGS, ss_pos[1])

include(joinpath(@__DIR__, "SMC.jl"))
include(joinpath(@__DIR__, "fisher_maps.jl"))
load_variant!(joinpath(@__DIR__, get(ss_flags, "variant", split(ss_run_name, "_")[1]) * ".jl"))

all_particles, _, targetinfos, _, _ = load_SMC(joinpath(@__DIR__, "output/data$(dir_idx)/$(ss_run_name).jld2"))
out = joinpath(@__DIR__, "output/start_sweep/data$(dir_idx)")
mkpath(out)
const COORD_NAMES = ["λ_EL", "λ_LA", "ρ", "δ_E", "δ_L", "δ_A", "κ_E", "κ_L", "κ_A", "σ"]

cols = ["run", "data", "iter", "gamma", "coord", "coord_name", "is_ss", "start", "obj_start", "obj_run1", "obj_run2",
    "obj_run3", "D_run1", "D_run3", "f_calls_run1", "converged_run1", "time_run1", "time_run3", "logratio_run1",
    "logratio_run3", "zeta_run1", "zeta_run3"]
rows = Vector{Vector{Any}}()
for i in ss_iters
    γ = targetinfos[i+1].γ
    lp = Base.Fix1(LogDensityProblems.logdensity, ldp_builder(prior_path, γ))
    ps = all_particles[i+1]
    X = reduce(hcat, [p.state for p in ps]); G = similar(X); logp = zeros(size(X, 2))
    for j in axes(X, 2)
        r = DiffResults.GradientResult(X[:, j]); r = ForwardDiff.gradient!(r, lp, X[:, j])
        logp[j] = DiffResults.value(r); G[:, j] .= DiffResults.gradient(r)
    end
    maximum(abs.(logp .- [p.logtarget for p in ps])) < 1e-6 || error("log target mismatch: wrong path or γ?")
    keep = logp .>= quantile(logp, 0.01)
    for k in axes(X, 1)
        θ, α = X[k, keep], G[k, keep]
        λ = DEFAULT_λN / length(θ)
        obj(p) = (v = Dk(to_map(p, ZETA_MIN), θ, α) + λ * (p[3] - p[2])^2; isfinite(v) ? v : Inf)
        starts = Pair{String,TwoSlope}["q25" => split_start(θ, α, quantile(θ, 0.25)).m,
            "median" => split_start(θ, α, median(θ)).m, "q75" => split_start(θ, α, quantile(θ, 0.75)).m,
            "affine" => TwoSlope(median(θ), nutpie_scale(θ, α), nutpie_scale(θ, α), 1.0)]
        k in ss_idxs && push!(starts, "thres" => split_start(θ, α, thres).m)
        for (name, m0) in starts
            p = from_map(m0, ZETA_MIN)
            objs, ts, res1 = Float64[], Float64[], nothing
            p1 = p
            for r in 1:3
                t = @elapsed res = optimize(obj, p, NelderMead(), Optim.Options(iterations=4000, f_reltol=1e-12, x_abstol=1e-10))
                p = Optim.minimizer(res)
                push!(objs, Optim.minimum(res)); push!(ts, t)
                if r == 1
                    res1 = res; p1 = p
                end
            end
            m1, m3 = to_map(p1, ZETA_MIN), to_map(p, ZETA_MIN)
            push!(rows, Any[ss_run_name, dir_idx, i, γ, k, COORD_NAMES[k], k in ss_idxs, name, obj(from_map(m0, ZETA_MIN)),
                objs..., Dk(m1, θ, α), Dk(m3, θ, α), Optim.f_calls(res1), Optim.converged(res1), ts[1], sum(ts),
                log(m1.shi / m1.slo), log(m3.shi / m3.slo), m1.ζ, m3.ζ])
        end
    end
    @info "iteration $i done"
    open(joinpath(out, "$(ss_run_name).csv"), "w") do io
        println(io, join(cols, ","))
        foreach(r -> println(io, join(r, ",")), rows)
    end
end
println("Output: ", joinpath(out, "$(ss_run_name).csv"))
