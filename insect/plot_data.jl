include(joinpath(@__DIR__, "setup.jl"));

# Fetch packages.
using Distributions, LinearAlgebra, LogExpFunctions, Optim, OrdinaryDiffEq, PDMats, Random
using JLD2, ProgressMeter, Suppressor

include(joinpath(@__DIR__, "../plot_helpers.jl"));

@load joinpath(@__DIR__, "params.jld2") tuned_params;
@load joinpath(@__DIR__, "data.jld2") all_data;

COLORS = Makie.wong_colors()[5:7];
MARKERS = [:circle, :diamond, :star5];

param_labels = [
    L"\lambda_{EL}", L"\lambda_{LA}", L"\rho", 
    L"\delta_E", L"\delta_L", L"\delta_A", 
    L"\kappa_E", L"\kappa_L", L"\kappa_A", L"\sigma"
];
sym2label = Dict(zip(Symbol.(parameters(models[end])), param_labels))

# Plot datasets
function make_plot!(model, u0, ps, data)
    oprob = ODEProblem(model, u0, data.t[end], ps)
    sol = solve(oprob)
    for d in eachindex(sol.u[1])
        lines!(sol.t, getindex.(sol.u, d), color=COLORS[d])
        scatter!(data.t, data[d], color=COLORS[d], alpha=0.8, marker=MARKERS[d])
    end
end
function make_plot!(idx, models, u0, param_sets, all_data)
    make_plot!(models[idx], u0, param_sets[idx], all_data[idx])
end

f = Figure(size=(1080, 1440));
to_exclude = Symbol.(parameters(models[1]))
for (i, midx) in enumerate(feasible_idxs)
    ps = parameters(models[midx])
    p_title = join([sym2label[p].s for p in Symbol.(ps) if p ∉ to_exclude], ", ")
    println(p_title)
    ax = Axis(
        f[cld(i, 5), mod1(i, 5)],
        title=latexstring(p_title), titlesize=18
    );
    make_plot!(midx, models, u0, tuned_params, all_data)
    ax.xticklabelsvisible = i >= 40
    if i >= 40
        ax.xlabel = "Time (a.u.)"
        ax.xlabelsize = 18
    end
    if i == 40
        ax.alignmode=Mixed(bottom=-42)
    end
end
Legend(
    # f[end+1,:], 
    f[:,6],
    [[
        LineElement(color=COLORS[d], linestyle=nothing), 
        MarkerElement(color=COLORS[d], alpha=0.8, marker=MARKERS[d])
    ] for d in 1:3][end:-1:begin],
    ["Egg", "Larva", "Adult"][end:-1:begin],
    # orientation=:horizontal, 
    # tellwidth=false, 
    labelsize=18
);
Label(f[0,:], "Synthetic datasets (population size against time)", font=:bold, fontsize=20);

for i in 1:8
    rowgap!(f.layout, i, 12)
end
display(f);

save_dir = mkpath(joinpath(@__DIR__, "imgs"));
save("$(save_dir)/insect_datasets.png", f, px_per_unit=4);