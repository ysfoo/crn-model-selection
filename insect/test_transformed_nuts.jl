# Offline test of NUTS on the two-slope transformed target against the current move (`nuts_invmass_move`), on a stored
# SMC population. Nothing is written except CSV files; the stored run is only read.
#
# Usage (from insect/):
#   julia --project=.. test_transformed_nuts.jl <data idx> <run name> <iteration> [--n_sub=400] [--seed=1]
#                                               [--variant=SMCnnn] [--zeta_min=0.2]
#
# The population of iteration i (equal weights, targeting π_(γ_i)) is moved at its own γ_i by two methods:
#   affine:      NUTS on θ with the nutpie diagonal mass matrix of the population (the current sampler);
#   transformed: NUTS on u, θ_k = F_k(u_k), with an identity mass matrix. Every coordinate gets the two-slope map
#                fitted to the population with the default log-slope penalty (λ = 100 / N), or the equal-slope optimum
#                (the affine nutpie map) if `check_asymptotes` fails. This is the rule `compute_TQB_info` uses in SMC.
# Maps and the affine mass matrix come from the whole (trimmed) population; the moves are run on a random subset of n_sub
# particles. Each method gets two rounds of 5 NUTS transitions (`repeat_5`): round A, then round B with the step size
# from the quasi-binomial fit to round A's acceptance (as `compute_QB_info` does in pass 1). The affine round A uses the
# stored particles' step sizes and acceptance rates, as the sampler would; the transformed round A has no history, so it
# starts from ε = 0.3 with a wider jitter (0.5) to give the fit a range of step sizes. Compare the methods on round B.
#
# Outputs: output/transformed_nuts/data<d>/<run name>/iter<i>_transitions.csv and iter<i>_maps.csv.

tn_flags = Dict(begin
        kv = split(a[3:end], "=")
        kv[1] => (length(kv) > 1 ? kv[2] : "true")
    end for a in ARGS if startswith(a, "--"))
tn_pos = [a for a in ARGS if !startswith(a, "--")]
length(tn_pos) == 3 || error("usage: test_transformed_nuts.jl <data idx> <run name> <iteration> [--n_sub=400] [--seed=1]")
tn_run_name, tn_iter = tn_pos[2], parse(Int, tn_pos[3])
const N_SUB = parse(Int, get(tn_flags, "n_sub", "400"))
const TN_SEED = parse(Int, get(tn_flags, "seed", "1"))
const ZETA_MIN = parse(Float64, get(tn_flags, "zeta_min", "0.2"))
empty!(ARGS); push!(ARGS, tn_pos[1])

include(joinpath(@__DIR__, "SMC.jl"))
include(joinpath(@__DIR__, "fisher_maps.jl"))
load_variant!(joinpath(@__DIR__, get(tn_flags, "variant", split(tn_run_name, "_")[1]) * ".jl"))

tn_in = joinpath(@__DIR__, "output/data$(dir_idx)/$(tn_run_name).jld2")
tn_out = joinpath(@__DIR__, "output/transformed_nuts/data$(dir_idx)/$(tn_run_name)")
mkpath(tn_out)

all_particles, _, targetinfos, _, _ = load_SMC(tn_in)
particles = all_particles[tn_iter+1]
γ = targetinfos[tn_iter+1].γ
target = ldp_builder(prior_path, γ)
logp_func = Base.Fix1(LogDensityProblems.logdensity, target)
n_dim = LogDensityProblems.dimension(target)
const COORD_NAMES = ["λ_EL", "λ_LA", "ρ", "δ_E", "δ_L", "δ_A", "κ_E", "κ_L", "κ_A", "σ"]
@info "Transformed NUTS test" tn_run_name tn_iter γ N_SUB TN_SEED tn_in tn_out

# ---------------------------------------------------------------------------------------------------------------------
# Scores, maps, mass matrices

X = reduce(hcat, [p.state for p in particles])
G = similar(X); logp = Vector{Float64}(undef, size(X, 2))
let result = DiffResults.GradientResult(X[:, 1])
    for j in axes(X, 2)
        result = ForwardDiff.gradient!(result, logp_func, X[:, j])
        logp[j] = DiffResults.value(result)
        G[:, j] .= DiffResults.gradient(result)
    end
end
max_diff = maximum(abs.(logp .- [p.logtarget for p in particles]))
max_diff < 1e-6 || error("Recomputed log target differs from the stored one (max $(max_diff)): wrong path or γ?")
keep = logp .>= quantile(logp, 0.01)

maps = [equal_slope_map(X[k, keep], G[k, keep]) for k in 1:n_dim]
map_rows = Vector{Vector{Any}}()
for k in 1:n_dim
    f = fit_twoslope(k, X[k, keep], G[k, keep])
    m, reason = check_asymptotes(f.m, f.u, X[k, keep], G[k, keep])
    maps[k] = m
    @info "map $(COORD_NAMES[k])" f.m f.D f.D_affine used_twoslope = isempty(reason) reason
    push!(map_rows, Any[k, COORD_NAMES[k], f.m.θs, f.m.slo, f.m.shi, f.m.ζ, f.D, f.D_affine, isempty(reason), reason])
end
open(joinpath(tn_out, "iter$(tn_iter)_maps.csv"), "w") do io
    println(io, "coord,coord_name,theta_star,s_low,s_high,zeta,D_twoslope,D_affine,used,fallback_reason")
    foreach(r -> println(io, join(r, ",")), map_rows)
end

ttarget = TransformedTarget(target, maps)
mkparticles(states) = [SMCParticle(s, NaN, NaN, NaN, 0, NamedTuple()) for s in states]
U = reduce(hcat, [to_latent(maps, X[:, j]) for j in axes(X, 2)])
@info "inverse map accuracy" max_err = maximum(abs.(reduce(hcat, [to_theta(maps, U[:, j]) for j in axes(U, 2)]) .- X))
invmass_aff = compute_invmass(particles, target, 1).invmass
# the transformed move uses M = I; the u-space nutpie diagonal is only logged, to check that the maps set the scale
invmass_tr = ones(n_dim)
nutpie_u = compute_invmass(mkparticles([U[:, j] for j in axes(U, 2)]), ttarget, 1).invmass
@info "u-space nutpie diagonal (not used)" nutpie_u = round.(nutpie_u, sigdigits=3)

# ---------------------------------------------------------------------------------------------------------------------
# Moves

# One particle, n_nuts transitions; returns the states, the per-transition statistics and the jittered step size.
function nuts_transitions(seed, tgt, state, n_nuts, invmass, base_ε, jitter)
    rng = StableRNG(seed)
    h = Hamiltonian(DiagEuclideanMetric(invmass), tgt, ForwardDiff)
    ε = base_ε * exp(jitter * randn(rng))
    kernel = HMCKernel(Trajectory{MultinomialTS}(Leapfrog(ε), GeneralisedNoUTurn()))
    states, stats = sample(rng, h, kernel, state, n_nuts, NoAdaptation(), 0; verbose=false, progress=false)
    return states, stats, ε
end

n_slab(θ) = count(θ[k] > thres for k in ss_idxs)
classes(θ) = [θ[k] > thres for k in ss_idxs]
const SD = vec(std(X[:, keep]; dims=2))   # population sd per coordinate, to standardise jumps

rows = Vector{Vector{Any}}()
# Runs one round for one method from θ-states `starts`; records each transition in θ-space; returns the end states and
# (step sizes, mean acceptance) per particle for the step-size fit of the next round.
function run_round(method, round, starts, seeds, base_ε, jitter)
    tgt, invmass = method == "transformed" ? (ttarget, invmass_tr) : (target, invmass_aff)
    ends = similar(starts)
    εs, accs = zeros(length(starts)), zeros(length(starts))
    t_round = @elapsed for (j, θ0) in enumerate(starts)
        s0 = method == "transformed" ? to_latent(maps, θ0) : θ0
        t = @elapsed((states, stats, ε) = nuts_transitions(seeds[j], tgt, s0, 5, invmass, base_ε, jitter))
        θs = method == "transformed" ? [to_theta(maps, s) for s in states] : states
        prev = θ0
        for (n, (θ, st)) in enumerate(zip(θs, stats))
            push!(rows, Any[tn_run_name, tn_iter, γ, method, round, j, n, ε, st.n_steps, st.tree_depth,
                st.numerical_error, st.acceptance_rate, n_slab(prev), count(classes(θ) .!= classes(prev)),
                sum(abs2, (θ .- prev) ./ SD), t / 5])
            prev = θ
        end
        ends[j] = θs[end]
        εs[j] = ε
        accs[j] = mean(isfinite(st.acceptance_rate) ? st.acceptance_rate : 0.0 for st in stats)
    end
    # consistency: the end states are valid θ-space states of the original target
    lps = [logp_func(θ) for θ in ends]
    @info "$method round $round" t_round base_ε jitter mean_acc = mean(accs) n_nonfinite_end = count(!isfinite, lps)
    return ends, εs, accs
end

# Step size for round B: quasi-binomial fit of acceptance on log step size, target 0.8 (as in `compute_QB_info`).
function qb_stepsize(εs, accs)
    fit = fit_acc_stepsize(εs, accs; acc_target=0.8, min_obs=20)
    return isnothing(fit) ? exp(mean(log.(εs))) : fit.stepsize
end

rng = StableRNG(hash((dir_idx, tn_run_name, tn_iter, TN_SEED)))
sub = randperm(rng, length(particles))[1:N_SUB]
starts = [X[:, j] for j in sub]
seeds_A, seeds_B = rand(rng, UInt64, N_SUB), rand(rng, UInt64, N_SUB)

# affine round A: step size from the stored population, as the sampler's pass 1 would choose it
stored_ε = [p.stepsize for p in particles]
stored_acc = [hasproperty(p.info, :curr_acc_rate) ? p.info.curr_acc_rate : NaN for p in particles]
ε_aff_A = qb_stepsize(stored_ε, stored_acc)

results = Dict{String,Any}()
for (method, εA, jitA) in (("affine", ε_aff_A, 0.2), ("transformed", 0.3, 0.5))
    endsA, εs, accs = run_round(method, "A", starts, seeds_A, εA, jitA)
    εB = qb_stepsize(εs, accs)
    run_round(method, "B", endsA, seeds_B, εB, 0.2)
    results[method] = (εA=εA, εB=εB)
    open(joinpath(tn_out, "iter$(tn_iter)_transitions.csv"), "w") do io
        println(io, "run,iter,gamma,method,round,particle,transition,stepsize,n_steps,tree_depth,divergent,acc,",
            "n_slab_start,n_class_flips,sqjump_std,time_per_transition")
        foreach(r -> println(io, join(r, ",")), rows)
    end
end
@info "step sizes" results

# ---------------------------------------------------------------------------------------------------------------------
# Summary of round B

using DataFrames
df = DataFrame([Symbol(c) => [r[i] for r in rows] for (i, c) in enumerate(split(
    "run,iter,gamma,method,round,particle,transition,stepsize,n_steps,tree_depth,divergent,acc,n_slab_start," *
    "n_class_flips,sqjump_std,time_per_transition", ","))])
println("\n$(tn_run_name), data $(dir_idx), iteration $(tn_iter), γ = $(round(γ, sigdigits=3)), $(N_SUB) particles x 5 transitions")
for rnd in ("A", "B")
    println("Round $rnd")
    for method in ("affine", "transformed")
        d = df[(df.method .== method) .& (df.round .== rnd), :]
        g = d.n_steps
        println(rpad("  " * method, 15),
            "grads/trans mean ", rpad(round(mean(g), sigdigits=4), 7), " median ", rpad(round(median(g), sigdigits=4), 6),
            " at cap ", rpad(round(100mean(g .>= 1023), digits=1), 5), "%",
            "  acc ", rpad(round(mean(d.acc), digits=3), 6), " div/trans ", rpad(round(mean(d.divergent), digits=4), 7),
            " flips/trans ", rpad(round(mean(d.n_class_flips), digits=4), 7),
            " sqjump/1k grads ", rpad(round(1000sum(d.sqjump_std) / sum(g), sigdigits=3), 7),
            " s/trans ", round(mean(d.time_per_transition), sigdigits=3))
    end
end
println("Round B gradients per transition by number of slab coordinates at the start of the transition:")
for method in ("affine", "transformed")
    d = df[(df.method .== method) .& (df.round .== "B"), :]
    println(rpad("  " * method, 15), join(["$(c) slab: $(round(mean(d.n_steps[d.n_slab_start .== c]), sigdigits=3)) " *
        "(n=$(count(d.n_slab_start .== c)))" for c in sort(unique(d.n_slab_start))], ", "))
end
println("\nOutputs: ", tn_out)
