# Convergence diagnostics of the MCMC chains (`chains_8k_model*.jld2`) of all insect datasets and models.
# Usage (from insect/): julia --project=.. MCMC_diagnostics.jl [first_data last_data]    (default: datasets 1 to 44)
# Writes output/MCMC_diagnostics.csv (one row per dataset, model and coordinate: bulk and tail ESS, R-hat) and prints
# - the number of chain files per dataset (64 when a dataset is complete);
# - summary statistics of bulk ESS, tail ESS and R-hat over all coordinates, and of the worst coordinate of each model;
# - every dataset, model and coordinate with bulk ESS < ESS_MIN or R-hat > RHAT_MAX;
# - for each flagged model, a comparison of its chains, marking a chain as stuck if its mean log density is more than
#   STUCK_LP_GAP below the median of the other chains or its final step size is below STUCK_STEP_RATIO times theirs.
# ESS and R-hat are the MCMCChains defaults: bulk ESS (also tail ESS) and rank-normalised split R-hat (maximum of bulk
# and tail), with the 5 chains pooled. Reads the outputs only; loading about 2800 files takes a few minutes.
using JLD2, MCMCChains, Printf, Statistics

const ESS_MIN = 500
const RHAT_MAX = 1.01
const STUCK_LP_GAP = 50
const STUCK_STEP_RATIO = 0.1
const N_MODELS = 64

data_idxs = isempty(ARGS) ? (1:44) : (parse(Int, ARGS[1]):parse(Int, ARGS[2]))
OUTDIR = joinpath(@__DIR__, "output")

rows = NamedTuple[]
chains = Dict{Tuple{Int,Int},Chains}()  # only flagged models are kept
n_files = Dict{Int,Int}()
for d in data_idxs, m in 1:N_MODELS
    fname = joinpath(OUTDIR, "data$d", "chains_8k_model$m.jld2")
    isfile(fname) || continue
    n_files[d] = get(n_files, d, 0) + 1
    chn = load(fname, "chn")
    eb, et, r = ess(chn; kind=:bulk), ess(chn; kind=:tail), rhat(chn)
    flagged = false
    for (k, p) in enumerate(eb.nt.parameters)
        row = (data=d, model=m, coord=String(p), ess_bulk=eb.nt.ess[k], ess_tail=et.nt.ess[k], rhat=r.nt.rhat[k])
        push!(rows, row)
        flagged |= row.ess_bulk < ESS_MIN || row.rhat > RHAT_MAX
    end
    flagged && (chains[(d, m)] = chn)
end

open(joinpath(OUTDIR, "MCMC_diagnostics.csv"), "w") do io
    println(io, "data,model,coord,ess_bulk,ess_tail,rhat")
    for r in rows
        @printf(io, "%d,%d,%s,%.2f,%.2f,%.6f\n", r.data, r.model, r.coord, r.ess_bulk, r.ess_tail, r.rhat)
    end
end

println("Chain files per dataset (", N_MODELS, " when complete):")
incomplete = [d => get(n_files, d, 0) for d in data_idxs if get(n_files, d, 0) != N_MODELS]
println(isempty(incomplete) ? "  all complete" : "  incomplete: $incomplete")

function print_summary(name, v)
    q = quantile(v, [0.01, 0.05, 0.5, 0.95])
    @printf("  %-22s min %9.4g  1%% %9.4g  5%% %9.4g  median %9.4g  mean %9.4g  95%% %9.4g  max %9.4g\n",
        name, minimum(v), q[1], q[2], q[3], mean(v), q[4], maximum(v))
end
models = unique((r.data, r.model) for r in rows)
println("\nSummary over ", length(rows), " coordinates in ", length(models), " models:")
print_summary("bulk ESS", [r.ess_bulk for r in rows])
print_summary("tail ESS", [r.ess_tail for r in rows])
print_summary("R-hat", [r.rhat for r in rows])
by_model = Dict(dm => NamedTuple[] for dm in models)
foreach(r -> push!(by_model[(r.data, r.model)], r), rows)
print_summary("min bulk ESS per model", [minimum(r.ess_bulk for r in v) for v in values(by_model)])
print_summary("max R-hat per model", [maximum(r.rhat for r in v) for v in values(by_model)])

flagged = sort([r for r in rows if r.ess_bulk < ESS_MIN || r.rhat > RHAT_MAX]; by=r -> (r.data, r.model, r.coord))
println("\n", length(flagged), " coordinates in ", length(chains), " models with bulk ESS < $ESS_MIN or R-hat > $RHAT_MAX:")
for r in flagged
    @printf("  data %2d  model %2d  %-6s  bulk ESS %8.1f  tail ESS %8.1f  R-hat %.4f\n",
        r.data, r.model, r.coord, r.ess_bulk, r.ess_tail, r.rhat)
end

println("\nChains of the flagged models (* = stuck):")
for (d, m) in sort(collect(keys(chains)))
    chn = chains[(d, m)]
    n_chains = size(chn, 3)
    lp = [mean(chn[:lp][:, c]) for c in 1:n_chains]
    ϵ = [chn[:step_size][end, c] for c in 1:n_chains]
    println("  data $d model $m")
    for c in 1:n_chains
        others = setdiff(1:n_chains, c)
        stuck = lp[c] < median(lp[others]) - STUCK_LP_GAP || ϵ[c] < STUCK_STEP_RATIO * median(ϵ[others])
        means = join([@sprintf("%.2f", mean(chn[p][:, c])) for p in names(chn, :parameters)], " ")
        @printf("    %s chain %d: mean lp %10.2f  step size %.3g  mean leapfrog steps %6.1f  divergences %5d  acceptance %.3f  means %s\n",
            stuck ? "*" : " ", c, lp[c], ϵ[c], mean(chn[:n_steps][:, c]), sum(chn[:numerical_error][:, c]),
            mean(chn[:acceptance_rate][:, c]), means)
    end
end
