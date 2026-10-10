include(joinpath(@__DIR__, "SMC.jl"))
using StatsFuns

run_str = "SMC335"
σ0 = sqrt(68);
unimodal_prior = Normal(μ0, σ0);
unimodal_dists = [
    fill(slab_prior, length(std_idxs));
    fill(unimodal_prior,  length(ss_idxs));
    noise_prior
];
init_dists = unimodal_dists; 

move_func = nuts_invmass_move;  init_stepsize = 0.5;
n_nuts_func = repeat_5_min_2;
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
pop_size = Dict("1k" => 1000, "2k" => 2000, "4k" => 4000, "8k" => 8000)[psize_str]; 
target_ess = 0.8pop_size; init_pop_size = pop_size;

fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str)_$(psize_str).jld2");
display(fname)

rng = StableRNG(hash((dir_idx, run_str, pop_size)))
run_SMC(
    pop_size, target_ess, init_sampler, prior_path, ldp_builder, move_func, n_nuts_func, fname;
    init_stepsize=init_stepsize, init_pop_size=init_pop_size, max_npass=10, compute_pop_info=compute_QB_info,
    verbose=1, rng=rng, parallel=:distributed, pbar_lines=20,  # errors if SMC_NWORKERS is unset
);

# resume_SMC(
#     pop_size, target_ess, prior_path, ldp_builder, move_func, n_nuts_func, fname;
#     max_npass=10, compute_pop_info=compute_QB_info,
#     verbose=1, rng=rng, parallel=:distributed, pbar_lines=20,  # errors if SMC_NWORKERS is unset
# );