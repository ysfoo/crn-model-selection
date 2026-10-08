# Fetch packages.
using Catalyst, Combinatorics
using DataFrames, Distributions
using Logging, LoggingExtras

nowarn_logger = EarlyFilteredLogger(global_logger()) do log
    log.level != Logging.Warn
end

macro nowarn_load(filename, vars...)
    quote
        ($([esc(v) for v in vars]...),) =
            with_logger(nowarn_logger) do
                ($([:(load($(esc(filename)), $(string(v)))) for v in vars]...),)
            end

        $(Symbol[v for v in vars])
    end
end

# Generates all potential models.
begin
    t = default_t()
    all_ps = @parameters λ1 λ2 ρ δ1 δ2 δ3 κ1 κ2 κ3 σ
    @species E(t) L(t) A(t)
    rxs_base = [
        Reaction(λ1, [E], [L]),
        Reaction(λ2, [L], [A]),
        Reaction(ρ, [A], [A, E]),
    ]
    rxs_extra = [
        Reaction(δ1, [E], []),
        Reaction(δ2, [L], []),
        Reaction(δ3, [A], []),
        Reaction(κ1, [E], [E], [2], [1]),
        Reaction(κ2, [L], [L], [2], [1]),
        Reaction(κ3, [A], [A], [2], [1]),
    ]
    models = []
    for rxs in collect(combinations(rxs_extra))
        @named rs = ReactionSystem(
            [rxs_base; rxs], t, [E, L, A], 
            [reduce(vcat, [Symbolics.get_variables(rx.rate) ∩ all_ps for rx in [rxs_base; rxs]])...; σ])
        push!(models, complete(rs))
    end
    n_extra = length(rxs_extra);
    n_models = length(models);

# Integer combinations
combs = combinations(collect(1:n_extra));
rx_boolmat = [Int(rx ∈ comb) for comb in combs, rx in 1:n_extra] # 2^R by R
nparams = length.(combs) .+ 4

# For ODE simulaion
u0 = [:E => 0.0, :L => 0.0, :A => 3.0]
n_u = length(u0)
t_end = 10.
end

function model_is_feasible(model)
    syms = Symbol.(parameters(model))
    return any(s ∈ syms for s in [:κ1, :κ2, :κ3]) && any(s ∈ syms for s in [:δ3, :κ3])
end

feasible_idxs = findall(model_is_feasible, models);
n_feasible = length(feasible_idxs)

# Latex labels
rx_labels = [
        begin 
        s = string(rx)
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        s
    end for rx in rxs_extra
]