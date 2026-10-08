## Setup
include(joinpath(@__DIR__, "../SMC_functions.jl"));
include(joinpath(@__DIR__, "../ODE_LDP.jl"));
include(joinpath(@__DIR__, "setup.jl"));

using Distributions, LinearAlgebra, Optim
using JLD2

using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);

LinearAlgebra.BLAS.set_num_threads(1)

rx_sys = models[end]
n_θ = length(parameters(rx_sys))
std_idxs = 1:3 # ODE parameters with standard prior
ss_idxs = 4:9 # ODE parameters with spike-and-slab prior
noise_idx = 10
n_ss = length(ss_idxs)

base_oprob = ODEProblem(rx_sys, u0, (0.0, 10.0), [p => 1. for p in parameters(rx_sys)]);

param_idxs = map((x)->parameter_index(base_oprob, x).idx, parameters(rx_sys))
u0_idxs = map((x)->parameter_index(base_oprob, Initial(x)).idx, unknowns(rx_sys))

ode_params!(buf, θ) = for i in 1:n_θ
    buf[i] = exp10(θ[i]) 
end;

ps = parameters(models[end])
param_labels = [
    L"\lambda_{EL}", L"\lambda_{LA}", L"\rho", 
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A", L"\sigma"
];
sym2label = Dict(zip(Symbol.(ps), param_labels))

μ_noise, σ_noise = -1., 1.;
μ_slab, σ_slab = 0., 2.;
μ_spike = -16.; σ_spike = σ_slab;
thres = 0.5 * (μ_spike + μ_slab);
μ0 = -8.;

noise_prior = Normal(μ_noise, σ_noise)
slab_prior = Normal(μ_slab, σ_slab)
spike_prior = Normal(μ_spike, σ_spike)

final_ss_prior = MixtureModel([slab_prior, spike_prior]);
final_dists = [
    fill(slab_prior, length(std_idxs));
    fill(final_ss_prior, length(ss_idxs));
    noise_prior
];
final_logprior_func(θ) = sum(logpdf(dist, val) for (dist, val) in zip(final_dists, θ));

extract_γ(targetinfo) = begin
    hasproperty(targetinfo, :γ) && return targetinfo.γ
    hasproperty(targetinfo, :β) && return targetinfo.β
    return last(targetinfo)    
end

convert_HMS(s) = string(floor(Int, s÷3600), ":", lpad(floor(Int, s%3600÷60), 2, '0'), ":", lpad(floor(Int, s%60), 2, '0'));

# Load previous results
logZs_fname = joinpath(@__DIR__, "output/logZs.jld2");
@load logZs_fname rAMIS_logZvecs BS_logZvecs all_times;
pvecs_BS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in BS_logZvecs];
pvecs_rAMIS = [exp.(logZvec .- logsumexp(logZvec)) for logZvec in rAMIS_logZvecs];
tvds_rAMIS = [0.5sum(abs, pvec_BS .- pvec_rAMIS) for (pvec_BS, pvec_rAMIS) in zip(pvecs_BS, pvecs_rAMIS)];

### End setup

## Load SMC results
run_str = "SMC505"
psize_str = "4k"
SMC_fname = joinpath(@__DIR__, "output/$(run_str)_$(psize_str).jld2");

pvecs_SMC = [Float64[] for _ in 1:n_feasible];
hours_SMC = [0. for _ in 1:n_feasible];
ns_nuts = [0 for _ in 1:n_feasible];
targetinfos_vec = [[] for _ in 1:n_feasible];

if isfile(SMC_fname)
    @load SMC_fname pvecs_SMC hours_SMC ns_nuts targetinfos_vec;
end;

tmp = 0;
@showprogress for dir_idx in 1:n_feasible
    fname = joinpath(@__DIR__, "output/data$(dir_idx)/$(run_str)_$(psize_str).jld2")
    if hours_SMC[dir_idx] > 0
        tmp += 1
        sleep(0.01)
        continue
    end
    if isfile(fname)
        all_particles, iter, targetinfos, npass_vec, smc_times = load_SMC(fname);
        # println(targetinfos[end])
        if extract_γ(targetinfos[end]) == 1.
            tmp += 1
            pvecs_SMC[dir_idx] = get_pvec(all_particles[end], thres)
            hours_SMC[dir_idx] = sum(smc_times)/3600
            ns_nuts[dir_idx] = sum(ps[1].n_nuts for ps in all_particles[2:end])
            targetinfos_vec[dir_idx] = targetinfos
        else
            # @info dir_idx iter targetinfos[end] 
        end
    end
end
tvds_SMC = [dir_idx => 0.5sum(abs, pvec_BS .- pvec_SMC) for (dir_idx, pvec_BS, pvec_SMC) in zip(1:n_feasible, pvecs_BS, pvecs_SMC) if !isempty(pvec_SMC)];

tmp
# findall(isempty.(pvecs_SMC))

summarystats(last.(tvds_SMC)) |> display
tvds_SMC
sort(tvds_SMC, by=last)[end-9:end]

println(round.(Int, filter(!iszero, hours_SMC)))
println(filter(!iszero, ns_nuts))
println(filter(!iszero, length.(targetinfos_vec)) .- 1)

summarystats(filter(!iszero, hours_SMC)) |> display
summarystats(filter(!iszero, ns_nuts)) |> display
summarystats(filter(!iszero, length.(targetinfos_vec)) .- 1) |> display

hist(filter(!iszero, hours_SMC), axis=(xlabel="Computational time (hours)",))
hist(filter(!iszero, ns_nuts), axis=(xlabel="Number of NUTS iterations",))
hist(filter(!iszero, length.(targetinfos_vec)) .- 1, axis=(xlabel="Number of SMC iterations", xticks=1:100))

@save SMC_fname pvecs_SMC hours_SMC ns_nuts targetinfos_vec;

## Check some individual runs

dir_idx = 4

# SMC872_5k
run_str = "SMC872_5k"
OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
fname = "$OUTDIR/$(run_str).jld2"
@load fname all_particles iter targetinfos smc_times;
var(reduce(hcat, getproperty.(all_particles[1], :state)), dims=2)

pvec_SMC = get_pvec(all_particles[end], thres);
0.5sum(abs, pvecs_BS[dir_idx] .- pvec_SMC)
esjds_872 = [[esjd_per_nuts(particle) for particle in particles] for particles in all_particles[2:end]];

fig = make_fig(all_particles[end], iter)
vid_path = joinpath(@__DIR__, "imgs/$(run_str)/data$(dir_idx)") |> mkpath;
save(joinpath(vid_path, "iter$(iter).png"), fig, px_per_unit=4);

# SMC762 (5k)
run_str = "SMC762_5k"
OUTDIR = joinpath(@__DIR__, "output/data$(dir_idx)");
fname = "$OUTDIR/$(run_str).jld2"
@load fname all_particles iter targetinfos smc_times;
var(reduce(hcat, getproperty.(all_particles[1], :state)), dims=2)

length(all_particles[end])
pvec_SMC = get_pvec(all_particles[end], thres);
0.5sum(abs, pvecs_BS[dir_idx] .- pvec_SMC)
esjds_762 = [[esjd_per_nuts(particle) for particle in particles] for particles in all_particles[2:end]];

fig = make_fig(all_particles[end], iter)
vid_path = joinpath(@__DIR__, "imgs/$(run_str)/data$(dir_idx)") |> mkpath;
save(joinpath(vid_path, "iter$(iter).png"), fig, px_per_unit=4);

begin
    f = Figure()
    ax = Axis(f[1,1], xlabel="SMC iteration", ylabel=L"\log_{10} \text{ESJD}",)
    color = Makie.wong_colors()[1]
    for (iter, esjd_vec) in enumerate(esjds_872)
        boxplot!(
            fill(iter - 0.2, length(esjd_vec)), esjd_vec .|> log10, width=0.5, 
            color=color, whiskercolor=whiskercolor,
            outliercolor=(color, 0.5), markersize=6
        )
    end
    color = Makie.wong_colors()[2]
    for (iter, esjd_vec) in enumerate(esjds_762)
        boxplot!(
            fill(iter + 0.2, length(esjd_vec)), esjd_vec .|> log10, width=0.5, 
            color=Makie.wong_colors()[2], whiskercolor=Makie.wong_colors()[2],
            outliercolor=(:grey, 0.5), markersize=6
        )
    end    
    display(f)
end


## Plots of TVD and pvec
using SpecialFunctions
mad_func(N, p) = exp(floor(N*p)*log(p)+(N-floor(N*p))*log(1-p)+logabsbinomial(N-1, floor(Int, N*p))[1])

# etvds = map(pvec -> sum(sqrt, pvec)/sqrt(2π*10^4), pvecs_BS);
etvds = map(pvec -> sum(p->p*mad_func(5000,p), pvec), pvecs_BS);
begin
    f = Figure()
    ax = Axis(
        f[1,1], limits=((0., 1.025*maximum(etvds)), (0., 1.025*maximum(tvds_SMC .|> last))),
        xlabel="Expected TVD assuming ideal sampling", ylabel="Actual TVD (SMC)"
    )
    lines!(
        [0, maximum(tvds_SMC .|> last)], [0, maximum(tvds_SMC .|> last)], 
        color=:grey30, alpha=0.7, linestyle=:dash
    )
    plot_idxs = first.(tvds_SMC)
    sc = scatter!(etvds[plot_idxs], last.(tvds_SMC), alpha=0.7, color=ns_nuts[plot_idxs])
    Colorbar(f[1,2], sc, label="Number of NUTS iterations")
    display(f)
    save(joinpath(@__DIR__, "imgs/$(run_str)_TVDs.png"), f, px_per_unit=4)
end

sort(tvds_SMC, by=last)

hist(last.(tvds_SMC), axis=(xlabel="TVD relative to bridge sampling",), bins=0:0.005:(0.005+maximum(last.(tvds_SMC))))

begin
    # Representative case looks ok
    # dir_idx = n_feasible

    # ?
    # dir_idx = 20

    # AMIS and SMC agree
    # dir_idx = 21

    # AMIS and bridge agree
    dir_idx = 39

    # AMIS and bridge agree
    # dir_idx = 3

    # AMIS and bridge agree
    # dir_idx = 31 

    n_plot = 20
    plot_order = sortperm(pvecs_BS[dir_idx], rev=true)[1:n_plot]

    COLORS = Makie.wong_colors()[[4,2,5]]

    f = Figure(size=(400, 640))
    ax = Axis(
        f[1,1], yreversed=true,
        ylabel="Models ranked by bridge sampling", xlabel="Model posterior probability",
        limits=((0, nothing), (0.3, n_plot+0.7)), yticks = (1:n_plot, string.(plot_order))
    )
    width = 0.25
    barplot!(
        (1:n_plot) .- width, pvecs_rAMIS[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[1], label="Robust AMIS"
    )
    barplot!(
        (1:n_plot) , pvecs_BS[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[2], label="Bridge sampling"
    )
    barplot!(
        (1:n_plot) .+ width, pvecs_SMC[dir_idx][plot_order], direction=:x,
        gap=0, width=width, color=COLORS[3], label="Spike-and-slab SMC"
    )
    axislegend(ax, position=:rb)
    display(f)
end

## Plot model posterior probability estimates with identifiability as colour
vrats_fname = joinpath(@__DIR__, "output/var_ratios.jld2");
@load vrats_fname var_ratios;

begin
    f = Figure()
    ax = Axis(
        f[1,1], aspect=DataAspect(), xticks=0:0.2:1,
        xlabel="SMC", xlabelsize=18, 
        xticklabelsize=16, yticklabelsize=16,
        ylabel="Bridge sampling", ylabelsize=18
    )
    vcat_pvecs_SMC = reduce(vcat, pvecs_SMC)
    pmax = pvecs_BS .|> maximum |> maximum
    lines!([0, pmax], [0, pmax], color=(:grey10, 0.8), linestyle=:dash)
    sc = scatter!(
        vcat_pvecs_SMC, reduce(vcat, pvecs_BS),
        color=sqrt.(vec(var_ratios')), colorrange=(0, 0.6), #colormap=Reverse(:viridis),
        highclip=:yellow,
        alpha=0.6, markersize=7
    )
    Colorbar(
        f[1,2], sc, ticklabelsize=16, alignmode=Mixed(right=0),
        label="Max posterior-to-prior SD ratio", labelsize=18
    )
    display(f)
    save(joinpath(@__DIR__, "imgs/SMC_10000_modelprobs.png"), f, px_per_unit=4)
end

# keep = findall(p -> all(p.state .> -8), all_particles[end])
# hist([p.logtarget for p in all_particles[end]][keep] .+ 6log(2), bins=65:0.5:85, alpha=0.5, normalization=:pdf);
# hist!(vec(chn[:lp].data), bins=65:0.5:85, alpha=0.5, normalization=:pdf);
# display(current_figure())

