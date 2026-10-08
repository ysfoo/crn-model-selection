include(joinpath(@__DIR__, "SMC.jl"))
using StatsFuns

# SMC345 (the 3xx path, moment-interpolated spike-and-slab mixture, with the :distributed backend) with NUTS on the
# two-slope transformed target: maps fitted to the population before every round (`compute_TQB_info`), identity mass
# matrix, same step-size rule (`nuts_transformed_move`).
# Usage: julia --project=.. SMC445.jl <data idx> [1k|2k|4k|8k]   (population size, default 4k)
run_str = "SMC445"
σ0 = sqrt(68);
unimodal_prior = Normal(μ0, σ0);
unimodal_dists = [
    fill(slab_prior, length(std_idxs));
    fill(unimodal_prior,  length(ss_idxs));
    noise_prior
];
init_dists = unimodal_dists;

move_func = nuts_transformed_move;  init_stepsize = 0.5;
n_nuts_func = repeat_5;
ldp_builder = make_ldp;
prior_path(θ, γ) = begin
    sum(
        begin
            x = θ[d]
            logaddexp(
                normlogpdf(μ0 + sqrt(γ)*(μ_spike-μ0), sqrt((1-γ)*σ0^2 + γ*σ_spike^2), x),
                normlogpdf(μ0 + sqrt(γ)*(μ_slab-μ0), sqrt((1-γ)*σ0^2 + γ*σ_slab^2), x)
            ) - log(2)
        end for d in ss_idxs
    ) + add_logprior(θ)
end

init_sampler = (rng) -> rand.(Ref(rng), init_dists);

psize_str = length(ARGS) >= 2 ? ARGS[2] : "4k"
pop_size = Dict("1k" => 1000, "2k" => 2000, "4k" => 4000, "8k" => 8000)[psize_str]; target_ess = 0.8pop_size; init_pop_size = pop_size;

fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str)_$(psize_str).jld2");
display(fname)
# run_SMC overwrites `fname` after every iteration; never start over an existing run (use resume_SMC for that)
myid() == 1 && isfile(fname) && error("$(fname) exists; move it away or resume it instead")

rng = StableRNG(hash((dir_idx, run_str, pop_size)))
run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_stepsize=init_stepsize, init_pop_size=init_pop_size, max_npass=10, compute_pop_info=compute_TQB_info,
    verbose=1, rng=rng, parallel=:distributed, pbar_lines=20,  # errors if SMC_NWORKERS is unset
);
