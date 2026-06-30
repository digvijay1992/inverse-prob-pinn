# ============================================================
# 1D Wave Equation — Inverse Problem: estimate wave speed c
#
# PDE:
#   u_tt = c^2 u_xx,   x ∈ [0, 1], t ∈ [0, T]
#
# Boundary conditions:
#   u(0, t) = 0
#   u(1, t) = 0
#
# Initial conditions:
#   u(x, 0)   = x(1 - x)
#   u_t(x, 0) = 0
#
# Setup:
#   - True wave speed c_true = 1.0 generates synthetic sensor data.
#   - PINN starts with initial guess c_param = 0.5.
#   - Both u(x, t) and c_param are learned simultaneously.
# ============================================================

using Pkg
Pkg.activate(@__DIR__)

using NeuralPDE
using Lux
using ModelingToolkit
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using LineSearches
using Random
using ComponentArrays
using Plots
using JLD2
using CSV, Tables
using DomainSets
using DomainSets: Interval
using Statistics
using StatsBase

# ------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------

const c_true   = 1.0
const Tfinal   = 1.0
const dx_train = 0.02       # PINN collocation spacing
const dx_eval  = 0.05       # evaluation grid spacing (space)
const dt_eval  = 0.02       # evaluation grid spacing (time)
const Nfourier = 200
const seed     = 1234

Random.seed!(seed)

# ------------------------------------------------------------
# Analytical solution (Fourier sine series)
# ------------------------------------------------------------

fourier_coeff(n::Int) = isodd(n) ? 8.0 / (pi^3 * n^3) : 0.0

function u_true_fn(x, t; c::Float64 = 1.0, N::Int = 200)
    s = 0.0
    @inbounds for n in 1:N
        s += fourier_coeff(n) * cos(c * n * pi * t) * sin(n * pi * x)
    end
    return s
end

u_true_grid(xs, ts; c::Float64 = 1.0, N::Int = 200) =
    [u_true_fn(x, t; c = c, N = N) for x in xs, t in ts]

# ------------------------------------------------------------
# Synthetic sensor data (c_true is unknown to the PINN)
# ------------------------------------------------------------

x_sensors = [0.25, 0.5, 0.75]
t_sensors = [0.2, 0.5, 0.8]

measure_points = [(x, t) for x in x_sensors for t in t_sensors]
u_meas = [u_true_fn(x, t; c = c_true, N = Nfourier) for (x, t) in measure_points]

sensor_input  = reduce(hcat, [[x, t] for (x, t) in measure_points])  # 2 × Nsensor
sensor_target = reshape(u_meas, 1, :)                                # 1 × Nsensor

println("Sensor data generated at $(length(measure_points)) points (c_true = $c_true).")

# ------------------------------------------------------------
# Symbolic PDE with trainable parameter c_param
# ------------------------------------------------------------

@parameters x t c_param
@variables u(..)

Dx  = Differential(x)
Dt  = Differential(t)
Dxx = Dx^2
Dtt = Dt^2

eq = Dtt(u(x, t)) ~ c_param^2 * Dxx(u(x, t))

bcs = [
    u(0.0, t) ~ 0.0,
    u(1.0, t) ~ 0.0,
    u(x, 0.0) ~ x * (1.0 - x),
    Dt(u(x, 0.0)) ~ 0.0,
]

domains = [
    x ∈ Interval(0.0, 1.0),
    t ∈ Interval(0.0, Tfinal),
]

@named pdesys = PDESystem(
    eq,
    bcs,
    domains,
    [x, t],
    [u(x, t)],
    [c_param];
    initial_conditions = Dict(c_param => 0.5),  # initial guess for c
)

# ------------------------------------------------------------
# PINN model (Lux + ComponentArrays)
# ------------------------------------------------------------

rng = Random.default_rng()
Random.seed!(rng, seed)

chain = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 1),
)

ps, st = Lux.setup(rng, chain)
ps = ComponentArray{Float64}(ps)

chains   = [chain]
init_ps  = [ps]

# ------------------------------------------------------------
# Additional data loss (sensor misfit)
# ------------------------------------------------------------

function data_loss(phi, θ, p)
    y_pred = phi[1](sensor_input, θ.u)         # 1 × Nsensor
    return sum(abs2, y_pred .- sensor_target) / length(u_meas)
end

# ------------------------------------------------------------
# PINN discretization
# ------------------------------------------------------------

strategy = GridTraining(dx_train)

discretization = PhysicsInformedNN(
    chains,
    strategy;
    init_params     = init_ps,
    param_estim     = true,
    additional_loss = data_loss,
)

prob = NeuralPDE.discretize(pdesys, discretization)

# ------------------------------------------------------------
# Training: loss / phase / c-history recording
# ------------------------------------------------------------

loss_history  = Float64[]
phase_history = String[]
c_history     = Float64[]

current_c(u) = u.p[1]

function make_callback(phase::String)
    return (state, l) -> begin
        push!(loss_history, l)
        push!(phase_history, phase)
        push!(c_history, current_c(state.u))
        if length(loss_history) % 100 == 0
            println("[$phase] iter $(length(loss_history))  loss = $l  c ≈ $(round(current_c(state.u), digits = 4))")
        end
        return false
    end
end

# Phase 1: Adam
println("\n--- Starting Adam training ---")
res_adam = Optimization.solve(
    prob,
    OptimizationOptimisers.Adam(0.01);
    callback = make_callback("Adam"),
    maxiters = 4000,
)

# Phase 2: BFGS refinement
println("\n--- Starting BFGS refinement ---")
prob_bfgs = remake(prob; u0 = res_adam.u)

res_bfgs = Optimization.solve(
    prob_bfgs,
    OptimizationOptimJL.BFGS(linesearch = BackTracking());
    callback = make_callback("BFGS"),
    maxiters = 3000,
)

θ = res_bfgs.u
@save "trained_theta.jld2" θ

# ------------------------------------------------------------
# Estimated wave speed
# ------------------------------------------------------------

c_est = θ.p[1]

println("\n==========================================")
println("Inverse wave speed estimation results")
println("==========================================")
println("True c      = ", c_true)
println("Estimated c = ", c_est)
println("Absolute error in c = ", abs(c_est - c_true))

# ------------------------------------------------------------
# PINN predictor
# ------------------------------------------------------------

phi = discretization.phi

function u_pinn(xval::Float64, tval::Float64, θ_local)
    X = reshape((xval, tval), 2, 1)
    return phi[1](X, θ_local.depvar.u)[1]
end

u_pinn_grid(xs, ts, θ_local) = [u_pinn(x, t, θ_local) for x in xs, t in ts]

# ------------------------------------------------------------
# Evaluation grid and error metrics
# ------------------------------------------------------------

x_vals = collect(0.0:dx_eval:1.0)
t_vals = collect(0.0:dt_eval:Tfinal)

U_true = u_true_grid(x_vals, t_vals; c = c_true, N = Nfourier)
U_pinn = u_pinn_grid(x_vals, t_vals, θ)
Err    = abs.(U_pinn .- U_true)

mae    = mean(Err)
rmse   = sqrt(mean((U_pinn .- U_true) .^ 2))
maxerr = maximum(Err)

println("\n==========================================")
println("Solution error metrics")
println("==========================================")
println("MAE   = ", mae)
println("RMSE  = ", rmse)
println("MaxErr= ", maxerr)

selected_times = [0.0, 0.25, 0.5, 0.75, 1.0]
println("\nErrors at selected time snapshots:")
for tsnap in selected_times
    j      = argmin(abs.(t_vals .- tsnap))
    e_snap = abs.(U_pinn[:, j] .- U_true[:, j])
    println(
        "t = $(round(t_vals[j], digits = 3)) : ",
        "MAE = $(mean(e_snap)), ",
        "MaxErr = $(maximum(e_snap))",
    )
end

# ------------------------------------------------------------
# Static plots
# ------------------------------------------------------------

default(size = (900, 600), lw = 2)

j0   = argmin(abs.(t_vals .- 0.0))
jend = length(t_vals)

p1 = plot(
    x_vals,
    U_true[:, j0],
    label = "True",
    xlabel = "x",
    ylabel = "u(x,t)",
    title = "Initial Condition (t = 0)",
    legend = :topright,
)
plot!(p1, x_vals, U_pinn[:, j0], label = "PINN", ls = :dash)

p2 = plot(
    x_vals,
    U_true[:, jend],
    label = "True",
    xlabel = "x",
    ylabel = "u(x,t)",
    title = "Final Time t = $(round(t_vals[jend], digits = 2)) | c_est = $(round(c_est, digits = 4))",
    legend = :topright,
)
plot!(p2, x_vals, U_pinn[:, jend], label = "PINN", ls = :dash)

p3 = heatmap(
    t_vals,
    x_vals,
    U_true,
    xlabel = "t",
    ylabel = "x",
    title = "True Solution (c = $c_true)",
    colorbar_title = "u",
)

p4 = heatmap(
    t_vals,
    x_vals,
    U_pinn,
    xlabel = "t",
    ylabel = "x",
    title = "PINN Solution (c_est = $(round(c_est, digits = 4)))",
    colorbar_title = "u",
)

plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1200))
savefig("inverse_c_wave_pinn_comparison.png")

savefig(p1, "inverse_c_initial_condition.png")
savefig(p2, "inverse_c_finaltime.png")
savefig(p3, "inverse_c_true_heatmap.png")
savefig(p4, "inverse_c_pinn_heatmap.png")

adam_idx = findall(==("Adam"), phase_history)
adam_end = isempty(adam_idx) ? 0 : maximum(adam_idx)

p_loss = plot(
    1:length(loss_history),
    loss_history,
    yscale = :log10,
    xlabel = "Iteration",
    ylabel = "Loss",
    title  = "Training Loss (Adam + BFGS)",
    label  = "Loss",
    lw     = 2,
)
if adam_end > 0
    vline!(p_loss, [adam_end]; label = "Adam → BFGS", ls = :dash, lc = :red, lw = 2)
end
savefig(p_loss, "inverse_c_loss_history.png")

p_c = plot(
    1:length(c_history),
    c_history,
    label = "Estimated c",
    lc    = :red,
    lw    = 2,
    xlabel = "Iteration",
    ylabel = "Wave speed c",
    title  = "Estimated vs True Wave Speed",
)
hline!(p_c, [c_true]; label = "True c = $c_true", ls = :dash, lc = :black, lw = 2)
if adam_end > 0
    vline!(p_c, [adam_end]; label = "Adam → BFGS", ls = :dot, lc = :gray, lw = 1)
end
savefig(p_c, "inverse_c_convergence.png")

# ------------------------------------------------------------
# GIFs: line plot and heatmap animations
# ------------------------------------------------------------

step_line = max(1, Int(cld(length(t_vals), 20)))
anim_line = @animate for k in 1:step_line:length(t_vals)
    t = t_vals[k]
    plot(
        x_vals,
        U_true[:, k],
        label  = "True",
        xlabel = "x",
        ylabel = "u(x,t)",
        title  = "True vs PINN  t = $(round(t, digits = 2))  c_est = $(round(c_est, digits = 4))",
        ylim   = (minimum(U_true), maximum(U_true)),
        legend = :topright,
    )
    plot!(x_vals, U_pinn[:, k], label = "PINN", ls = :dash)
end
gif(anim_line, "inverse_c_pinn_vs_true.gif"; fps = 5)

min_u     = minimum([minimum(U_true), minimum(U_pinn)])
max_u     = maximum([maximum(U_true), maximum(U_pinn)])
step_heat = max(1, Int(cld(length(t_vals), 20)))

anim_heat = @animate for i in 1:step_heat:length(t_vals)
    ti  = t_vals[i]
    ph1 = heatmap(
        t_vals[1:i],
        x_vals,
        U_true[:, 1:i],
        xlabel = "t",
        ylabel = "x",
        clim   = (min_u, max_u),
        title  = "True solution (up to t = $(round(ti, digits = 2)))",
        colorbar = false,
    )
    ph2 = heatmap(
        t_vals[1:i],
        x_vals,
        U_pinn[:, 1:i],
        xlabel = "t",
        ylabel = "x",
        clim   = (min_u, max_u),
        title  = "PINN solution (c_est = $(round(c_est, digits = 4)))",
        colorbar = true,
    )
    plot(ph1, ph2, layout = (2, 1), size = (600, 600))
end
gif(anim_heat, "inverse_c_heatmap_animation.gif"; fps = 5)

# ------------------------------------------------------------
# Save simulation data (JLD2 + CSV)
# ------------------------------------------------------------

@save "inverse_c_wave_pinn_simulation.jld2" x_vals t_vals U_true U_pinn Err c_true c_est dx_train dx_eval dt_eval Nfourier measure_points u_meas loss_history phase_history c_history

function build_table(xs, ts, U_true_local, U_pinn_local, Err_local)
    data = NamedTuple[]
    for (i, x) in enumerate(xs), (j, t) in enumerate(ts)
        push!(data, (
            x      = x,
            t      = t,
            u_true = U_true_local[i, j],
            u_pinn = U_pinn_local[i, j],
            error  = Err_local[i, j],
        ))
    end
    return data
end

data_table = build_table(x_vals, t_vals, U_true, U_pinn, Err)
CSV.write("inverse_c_wave_pinn_simulation.csv", Tables.columntable(data_table))

println("\n==========================================")
println("All output files saved:")
println("  inverse_c_wave_pinn_comparison.png")
println("  inverse_c_initial_condition.png")
println("  inverse_c_finaltime.png")
println("  inverse_c_true_heatmap.png")
println("  inverse_c_pinn_heatmap.png")
println("  inverse_c_loss_history.png")
println("  inverse_c_convergence.png")
println("  inverse_c_pinn_vs_true.gif")
println("  inverse_c_heatmap_animation.gif")
println("  inverse_c_wave_pinn_simulation.jld2")
println("  inverse_c_wave_pinn_simulation.csv")
println("==========================================")