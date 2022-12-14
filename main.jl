using DataFramesMeta
using StatsPlots
using GaussianProcesses
using Random
using Statistics

plot_dir = "plots"
isdir(plot_dir) || mkdir(plot_dir)
rng = MersenneTwister(123)

t = float.(1:100)
random_walk(t; x0, σ) = x0 .+ cumsum(map(i -> σ*randn(rng), t))
df = DataFrame(
    :t  => t,
    :y1 => random_walk(t; x0=1, σ=0.1),
    :y2 => random_walk(t; x0=-1, σ=0.15),
    :y3 => random_walk(t; x0=0.5, σ=0.25),
)
df = stack(df, Not(:t))
data_plt = @df df scatter(
    :t,
    :value,
    group=:variable,
    title="Data",
)
savefig(data_plt, joinpath(plot_dir, "data.png"))

function fit_model(df)
    t = df.t
    y = df.value
    signal = first(df.variable)
    kernel = SE(0.0, 0.0)
    gp = GP(t, y, MeanConst(mean(y)), kernel, log(0.5))
    optimize!(gp)
    return gp, t, y, signal
end

function plot_fit((gp, t, y, signal); forecast=10)
    t_pred = vcat(t, t[end]:t[end] + forecast)
    μ, Σ = predict_y(gp, t_pred)
    plt = plot(t_pred, μ, ribbon=2*sqrt.(Σ), label=nothing)
    plt = scatter!(plt, t, y, title=signal, label=nothing)
    return plt
end

N = length(unique(df.variable))
plots = combine(groupby(df, :variable), AsTable(:) => plot_fit ∘ fit_model => :plot)
fit_plt = plot(plots.plot..., layout=(N, 1), size=(600, 200N))
savefig(fit_plt, joinpath(plot_dir, "model_fit.png"))
