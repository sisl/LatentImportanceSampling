# packages
using CSV
using DataFrames
using MPOPIS
using Random
using Plots
using ProgressMeter
using LinearAlgebra
using Distributions


#*******************************************************************************
# simulator definition
#*******************************************************************************
function simulate_racing(;
    num_trials = 1,
    num_steps = 200,
    num_cars = 1,
    policy_type = :cemppi,
    laps = 2,
    num_samples = 150, 
    horizon = 50,
    λ = 10.0,
    α = 1.0,
    U₀ = zeros(Float64, num_cars*2),
    cov_mat = block_diagm([0.0625, 0.1], num_cars),
    ais_its = 10,
    λ_ais = 20.0,
    ce_elite_threshold = 0.8,
    ce_Σ_est = :ss,
    cma_σ = 0.75,
    cma_elite_threshold = 0.8,
    state_x_sigma = 0.0,
    state_y_sigma = 0.0,
    state_ψ_sigma = 0.0,
    seed = Int(rand(1:10e10)),
    plot_steps = false,
    pol_log = false,
    plot_traj = false,
    plot_traj_perc = 1.0,
    text_with_plot = true,
    text_on_plot_xy = (80.0, -60.0),
    save_gif = false,
)

    sim_type = :cr
    
    gif_name = "$sim_type-$num_cars-$policy_type-$num_samples-$horizon-$λ-$α-"
    gif_name = gif_name * "$num_trials-$laps.gif"
    anim = Animation()

    pm = Progress(num_trials, 1, "progress: ")

    states = zeros(num_trials, 8)

    for k ∈ 1:num_trials

        env = MPOPIS.CarRacingEnv(rng=MersenneTwister())

        pol = MPOPIS.get_policy(
            policy_type,
            env,num_samples, horizon, λ, α, U₀, cov_mat, pol_log, 
            ais_its, 
            λ_ais, 
            ce_elite_threshold, ce_Σ_est,
            cma_σ, cma_elite_threshold,  
        )

        seed!(env, seed + k)
        seed!(pol, seed + k)

        cnt = 0

        # Main simulation loop
        while !env.done && cnt <= num_steps
            # Get action from policy
            act = pol(env)
            # Apply action to environment
            env(act)
            cnt += 1

            # Plot or collect the plot for the animation
            if plot_steps || save_gif
                if plot_traj
                    p = plot(env, pol, plot_traj_perc, text_output=text_with_plot, text_xy=text_on_plot_xy)
                else 
                    p = plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)
                end
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end

            env.state[1] += state_x_sigma * randn(env.rng)
            env.state[2] += state_y_sigma * randn(env.rng)

            δψ = state_ψ_sigma * randn(env.rng)
            env.state[3] += δψ
            
            # Passive rotation matrix
            rot_mat = [ cos(δψ) sin(δψ) ;
                        -sin(δψ) cos(δψ) ]
            V′ = rot_mat*[env.state[4]; env.state[5]]
            env.state[4:5] = V′

        end

        states[k, :] = env.state[:]
        next!(pm)
    end

    if save_gif
        println("Saving gif...$gif_name")
        gif(anim, gif_name, fps=10)
    end
    return states
end


#*******************************************************************************
# generate data
#*******************************************************************************
targets = simulate_racing(
    num_trials = 50000,
    num_steps = 200, 
    num_samples = 25, 
    policy_type = :mppi, 
    horizon = 50,
    state_x_sigma = 0.05,
    state_y_sigma = 0.05,
    state_ψ_sigma = 0.01,
    seed = 0,
    save_gif=false)


# normalize the data
normalized_targets = (targets .- mean(targets, dims=1))./std(targets, dims=1)

col_names = ["x", "y", "yaw", "vlon", "vlat", "yawrate", "steer", "acc"]
df = DataFrame(normalized_targets, col_names)
CSV.write("racecar-flow.csv", df, writeheader=false)
