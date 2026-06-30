# ============================================================
# 1D Wave Equation — Inverse Problem: Estimate the Wave Speed c
#
# PDE:
#   u_tt = c^2 u_xx,   x in [0, 1], t in [0, T]
#
# Boundary conditions (BCs):
#   u(0, t) = 0
#   u(1, t) = 0
#
# Initial conditions (ICs):
#   u(x, 0)   = x(1 - x)
#   u_t(x, 0) = 0
#
# Setup:
#   - The true wave speed c_true = 1.0 is used to generate sensor data.
#   - The PINN starts from an initial guess c_param = 0.5.
#   - Both the solution u(x, t) and the wave speed parameter are learned jointly.
#
# Note:
#   The wave equation is second-order in time and is challenging for a
#   vanilla PINN. A denser collocation grid and tailored loss weighting
#   are used to encourage convergence toward c = 1.0.
# ============================================================

# ------------------------------------------------------------
# Environment setup — performed once before loading packages.
# Avoid calling Pkg.update inside a script that has already
# begun importing modules.
# ------------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__)

# Instantiate dependencies only if the environment is not yet prepared.
if !isfile(joinpath(@__DIR__, "Project.toml"))
    Pkg.add([
        "NeuralPDE",
        "Lux",
        "ModelingToolkit",
        "Optimization",
        "OptimizationOptimisers",
        "OptimizationOptimJL",
        "LineSearches",
        "Random",
        "ComponentArrays",
        "Plots",
        "JLD2",
        "CSV",
        "Tables",
        "DomainSets",
        "Statistics",
    ])
end

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
using DomainSets: Interval, infimum, supremum
using Statistics

# -----------------------------
# Problem parameters
# -----------------------------
const c_true   = 1.0
const Tfinal   = 1.0
const dx_train = 0.02      # PINN collocation spacing (denser than evaluation grid)
const dx_eval  = 0.05      # spacing for the evaluation and plotting grid
const dt_eval  = 0.02
const Nfourier = 200
const seed     = 1234

Random.seed!(seed)

# -----------------------------
# Analytical reference solution
#   u(x, t) = Σ a_n cos(c n π t) sin(n π x), where a_n = 8 / (π^3 n^3) for odd n
# -----------------------------
fourier_coeff(n::Int) = isodd(n) ? 8.0 / (pi^3 * n^3) : 0.0

function u_true_fn(x, t; c = 1.0, N = 200)
    s = 0.0
    for n in 1:N
        s += fourier_coeff(n) * cos(c * n * pi * t) * sin(n * pi * x)
    end
    return s
end

u_true_grid(xs, ts; c = 1.0, N = 200) = [u_true_fn(x, t; c = c, N = N) for x in xs, t in ts]

# -----------------------------
# Synthetic sensor data
# -----------------------------
x_sensors = [0.25, 0.5, 0.75]
t_sensors = [0.2, 0.5, 0.8]

measure_points = [(x, t) for x in x_sensors for t in t_sensors]   # 9 points stored as tuples
u_meas = [u_true_fn(x, t; c = c_true, N = Nfourier) for (x, t) in measure_points]

# Pack sensor inputs into a 2 × Nsensor matrix for batched phi() calls.
sensor_input = reduce(hcat, [[x, t] for (x, t) in measure_points])   # 2 × 9
sensor_target = reshape(u_meas, 1, :)                                # 1 × 9

println("Sensor data generated at $(length(measure_points)) points using c_true = $c_true")

# -----------------------------
# Symbolic PDE definition
# -----------------------------
@parameters x t c_param
@variables u(..)

Dx  = Differential(x)
Dt  = Differential(t)
Dxx = Dx^2
Dtt = Dt^2

eq = Dtt(u(x, t)) ~ c_param^2 * Dxx(u(x, t))

# Use Float64 literals throughout for consistent numerical types.
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

# The sixth positional argument is the set of PDE parameters to estimate.
# In the current ModelingToolkit setup, the initial guess for c_param is
# supplied through initial_conditions instead of defaults.
@named pdesys = PDESystem(
    eq, bcs, domains, [x, t], [u(x, t)], [c_param];
    initial_conditions = Dict([c_param => 0.5]),
)

# -----------------------------
# Neural network architecture
# -----------------------------
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

# For a single dependent variable with param_estim = true, the network and
# initial parameters must be wrapped in a vector so that NeuralPDE handles
# the internal structure correctly.
chains = [chain]
init_ps = [ps]

# -----------------------------
# Additional data loss term
# -----------------------------
# The sensor measurements are matched through this loss contribution.
function data_loss(phi, θ, p)
    y_pred = phi[1](sensor_input, θ.u)       # 1 × Nsensor
    return sum(abs2, y_pred .- sensor_target) / length(u_meas)
end

# -----------------------------
# PINN discretization
# -----------------------------
strategy = GridTraining(dx_train)

discretization = PhysicsInformedNN(
    chains,
    strategy;
    init_params     = init_ps,
    param_estim     = true,
    additional_loss = data_loss,
)

prob = NeuralPDE.discretize(pdesys, discretization)

# -----------------------------
# Training history tracking
# -----------------------------
loss_history  = Float64[]
phase_history = String[]
c_history     = Float64[]

# Extract the current wave-speed estimate from the optimizer state.
# With param_estim = true, the estimated parameters are stored in u.p.
current_c(u) = u.p[1]

make_cb(phase) = (state, l) -> begin
    push!(loss_history, l)
    push!(phase_history, phase)
    push!(c_history, current_c(state.u))
    if length(loss_history) % 100 == 0
        println("[$phase] iter $(length(loss_history))  loss = $l  c ≈ $(round(state.u[end], digits = 4))")
    end
    return false
end

# -----------------------------
# Phase 1: Adam optimization
# -----------------------------
println("\n--- Starting Adam training ---")
res_adam = Optimization.solve(
    prob,
    OptimizationOptimisers.Adam(0.01);
    callback = make_cb("Adam"),
    maxiters = 4000,
)

# -----------------------------
# Phase 2: BFGS refinement
# -----------------------------
println("\n--- Starting BFGS refinement ---")
prob_bfgs = remake(prob; u0 = res_adam.u)
res_bfgs = Optimization.solve(
    prob_bfgs,
    OptimizationOptimJL.BFGS(linesearch = BackTracking());
    callback = make_cb("BFGS"),
    maxiters = 3000,
)

θ = res_bfgs.u

# Checkpoint the trained parameters immediately so post-processing can be
# re-run without retraining.
@save "trained_theta.jld2" θ
flush(stdout)

# -----------------------------
# Extract the estimated wave speed
# -----------------------------
# The returned state uses θ.depvar for network weights and θ.p for estimated parameters.
c_est = θ.p[1]

println("\n==========================================")
println("Inverse wave speed estimation results")
println("==========================================")
println("True c      = ", c_true)
println("Estimated c = ", c_est)
println("Absolute error in c = ", abs(c_est - c_true))

# -----------------------------
# PINN prediction
# -----------------------------
phi = discretization.phi

function u_pinn(xval, tval, θ)
    X = reshape([xval, tval], 2, 1)
    return phi[1](X, θ.depvar.u)[1]
end

u_pinn_grid(xs, ts, θ) = [u_pinn(x, t, θ) for x in xs, t in ts]

# -----------------------------
# Evaluation grid
# -----------------------------
x_vals = collect(0.0:dx_eval:1.0)
t_vals = collect(0.0:dt_eval:Tfinal)

U_true = u_true_grid(x_vals, t_vals; c = c_true, N = Nfourier)
U_pinn = u_pinn_grid(x_vals, t_vals, θ)
Err    = abs.(U_pinn .- U_true)

# -----------------------------
# Error metrics
# -----------------------------
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
    println("t = $(round(t_vals[j], digits = 3)) : MAE = $(mean(e_snap)), MaxErr = $(maximum(e_snap))")
end

# ============================================================
# Static plots
# ============================================================
default(size = (900, 600), lw = 2)

j0   = argmin(abs.(t_vals .- 0.0))
jend = length(t_vals)

p1 = plot(x_vals, U_true[:, j0], label = "True", xlabel = "x", ylabel = "u(x,t)",
          title = "Initial Condition (t = 0)", legend = :topright)
plot!(p1, x_vals, U_pinn[:, j0], label = "PINN", ls = :dash)

p2 = plot(x_vals, U_true[:, jend], label = "True", xlabel = "x", ylabel = "u(x,t)",
          title = "Final Time t = $(round(t_vals[jend], digits=2))  |  c_est = $(round(c_est, digits=4))",
          legend = :topright)
plot!(p2, x_vals, U_pinn[:, jend], label = "PINN", ls = :dash)

p3 = heatmap(t_vals, x_vals, U_true, xlabel = "t", ylabel = "x",
             title = "True Solution  (c = $c_true)", colorbar_title = "u")

p4 = heatmap(t_vals, x_vals, U_pinn, xlabel = "t", ylabel = "x",
             title = "PINN Solution  (c_est = $(round(c_est, digits=4)))", colorbar_title = "u")

plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1200))
savefig("inverse_c_wave_pinn_comparison.png")

savefig(p1, "inverse_c_initial_condition.png")
savefig(p2, "inverse_c_finaltime.png")
savefig(p3, "inverse_c_true_heatmap.png")
savefig(p4, "inverse_c_pinn_heatmap.png")

# Loss history plot
adam_idx = findall(==("Adam"), phase_history)
adam_end = isempty(adam_idx) ? 0 : maximum(adam_idx)

p_loss = plot(1:length(loss_history), loss_history, yscale = :log10,
              xlabel = "Iteration", ylabel = "Loss",
              title = "Training Loss (Adam + BFGS)", label = "Loss", lw = 2)
if adam_end > 0
    vline!(p_loss, [adam_end]; label = "Adam → BFGS", ls = :dash, lc = :red, lw = 2)
end
savefig(p_loss, "inverse_c_loss_history.png")

# c convergence over iterations (now using the RECORDED history)
p_c = plot(1:length(c_history), c_history, label = "Estimated c", lc = :red, lw = 2,
           xlabel = "Iteration", ylabel = "Wave speed c",
           title = "Estimated vs True Wave Speed")
hline!(p_c, [c_true]; label = "True c = $c_true", ls = :dash, lc = :black, lw = 2)
if adam_end > 0
    vline!(p_c, [adam_end]; label = "Adam → BFGS", ls = :dot, lc = :gray, lw = 1)
end
savefig(p_c, "inverse_c_convergence.png")

# ============================================================
# GIF 1: PINN vs True line plot over time
# ============================================================
step_line = max(1, Int(cld(length(t_vals), 20)))
anim_line = @animate for k in 1:step_line:length(t_vals)
    t = t_vals[k]
    plot(x_vals, U_true[:, k], label = "True", xlabel = "x", ylabel = "u(x,t)",
         title = "True vs PINN  t = $(round(t, digits=2))  c_est = $(round(c_est, digits=4))",
         ylim = (minimum(U_true), maximum(U_true)), legend = :topright)
    plot!(x_vals, U_pinn[:, k], label = "PINN", ls = :dash)
end
gif(anim_line, "inverse_c_pinn_vs_true.gif"; fps = 5)

# ============================================================
# GIF 2: Heatmap animation (True top, PINN bottom)
# ============================================================
min_u     = minimum([minimum(U_true), minimum(U_pinn)])
max_u     = maximum([maximum(U_true), maximum(U_pinn)])
step_heat = max(1, Int(cld(length(t_vals), 20)))
anim_heat = @animate for i in 1:step_heat:length(t_vals)
    ti  = t_vals[i]
    ph1 = heatmap(t_vals[1:i], x_vals, U_true[:, 1:i], xlabel = "t", ylabel = "x",
                  clim = (min_u, max_u),
                  title = "True solution  (up to t = $(round(ti, digits=2)))", colorbar = false)
    ph2 = heatmap(t_vals[1:i], x_vals, U_pinn[:, 1:i], xlabel = "t", ylabel = "x",
                  clim = (min_u, max_u),
                  title = "PINN solution  (c_est = $(round(c_est, digits=4)))", colorbar = true)
    plot(ph1, ph2, layout = (2, 1), size = (600, 600))
end
gif(anim_heat, "inverse_c_heatmap_animation.gif"; fps = 5)

# ============================================================
# Save all simulation data
# ============================================================
@save "inverse_c_wave_pinn_simulation.jld2" x_vals t_vals U_true U_pinn Err c_true c_est dx_train dx_eval dt_eval Nfourier measure_points u_meas loss_history phase_history c_history

function build_table(xs, ts, U_true, U_pinn, Err)
    data = NamedTuple[]
    for (i, x) in enumerate(xs), (j, t) in enumerate(ts)
        push!(data, (x = x, t = t, u_true = U_true[i, j],
                     u_pinn = U_pinn[i, j], error = Err[i, j]))
    end
    return data
end

data_table = build_table(x_vals, t_vals, U_true, U_pinn, Err)
CSV.write("inverse_c_wave_pinn_simulation.csv", Tables.columntable(data_table))

println("\n==========================================")
println("All output files saved:")
println("  inverse_c_wave_pinn_comparison.png   — 2x2 static comparison")
println("  inverse_c_initial_condition.png      — IC comparison")
println("  inverse_c_finaltime.png              — final time comparison")
println("  inverse_c_true_heatmap.png           — true solution heatmap")
println("  inverse_c_pinn_heatmap.png           — PINN solution heatmap")
println("  inverse_c_loss_history.png           — loss curve with Adam/BFGS split")
println("  inverse_c_convergence.png            — estimated vs true c over iterations")
println("  inverse_c_pinn_vs_true.gif           — animated line comparison")
println("  inverse_c_heatmap_animation.gif      — animated heatmap comparison")
println("  inverse_c_wave_pinn_simulation.jld2  — full data for Julia post-processing")
println("  inverse_c_wave_pinn_simulation.csv   — flat table for any tool")
println("==========================================")
