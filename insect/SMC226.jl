include(joinpath(@__DIR__, "SMC.jl"))

run_str = "SMC226"
init_dists = final_dists; 
move_func = nuts_NDA_move; init_stepsize = 0.5;
n_nuts_func = run_10;
ldp_builder = make_ldp;

logprior_func(θ, β) = sum(logpdf(dist, val) for (dist, val) in zip(init_dists, θ));

function update!(logprior_func, curr_β, iter)
    nothing
end

init_sampler = (rng) -> rand.(Ref(rng), init_dists);

psize_str = "8k"; pop_size = 8000; target_ess = 0.8pop_size; init_pop_size = pop_size;

fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str)_$(psize_str).jld2");
display(fname)

rng = StableRNG(hash((dir_idx, run_str, pop_size)))
run_SMC(
    pop_size, target_ess, init_sampler, logprior_func, ldp_builder, move_func, n_nuts_func, fname;
    init_stepsize=init_stepsize, init_pop_size=init_pop_size, max_npass=5,
    verbose=1, rng=rng, parallel=true, pbar_lines=20,
);