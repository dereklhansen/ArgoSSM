module ArgoKalman

using Random: MersenneTwister
using LinearAlgebra: diagm
using Flux
using Zygote
using StatsFuns: logistic, logit
using StaticArrays

# include("./Kalman.jl")

import ..Kalman.kalman_loglik
import ..Kalman.draw_posterior_path
import ..Kalman.kalman_filter_mv
import ..Kalman.kalman_smoother_mv
import ..Kalman.draw_posterior_path

abstract type AbstractArgoKalman end

struct ArgoAR <: AbstractArgoKalman
    step_size::Float64
    iter::Int
    threshold::Float64
    rng::MersenneTwister
    callback::Union{Int,Missing}
end

function ArgoAR(;
    step_size = 1e-3,
    iter = 10000,
    threshold = 1e-6,
    rng = MersenneTwister(),
    callback = missing,
)
    ArgoAR(step_size, iter, threshold, rng, callback)
end

function fit_and_predict(m::AbstractArgoKalman, data)
    μ0, Tau0, X_in = set_initial_values(m, data.X)
    θ = get_initial_theta(m)
    θ_hat = train_argo_kalman(m, data, θ, μ0, Tau0, X_in)
    if !ismissing(m.callback)
        println("Done training model")
    end
    kalman_model = make_kalman_model(m, data.days, θ_hat)
    ll, states, variances, lls = kalman_filter_mv(kalman_model..., μ0, Tau0, X_in)
    states_sm, variances_sm, Covs_sm = kalman_smoother_mv(kalman_model..., μ0, Tau0, X_in)

    kalman_paths =
        map(1:1000) do i
            draw_posterior_path(
                m.rng,
                convert(Matrix{Float64}, (states)),
                variances,
                kalman_model.G0,
                kalman_model.G,
                kalman_model.Σ,
                kalman_model.Tau,
                μ0,
                Tau0,
            )
        end |> ((x) -> cat(x..., dims = 3))

    # For consistency with rest of code
    X_paths = permutedims(kalman_paths, (1, 3, 2))
    paths = X_paths[1:2, :, :]
    states = X_paths
    pred = states_sm[1:2, :]
    return (;
        θ_hat,
        means_sm = states_sm,
        variances_sm,
        Covs_sm,
        paths,
        states,
        pred,
        params_mle = collect(θ_hat),
    )
    #kalman_paths, X_paths)
end

function set_initial_values(::ArgoAR, X_in)
    X_in = deepcopy(X_in)
    μ0 = convert(Vector{Float64}, vcat(X_in[1, :], [0.0, 0.0]))
    Tau0 = diagm([1e-4, 1e-4, 1e-4, 1e-4])
    X_in[1, :] .= missing
    return μ0, Tau0, X_in
end

function get_initial_theta(::ArgoAR)
    θ = (
        σ_x_long = 3e-3,
        σ_x_lat = 3e-3,
        σ_v_long = 3e-3,
        σ_v_lat = 3e-3,
        σ_p = 1e-4,
        α = 0.99,
        v0long = 0.0,
        v0lat = 0.0,
    )
    return θ
end

function train_argo_kalman(
    m::AbstractArgoKalman,
    data,
    # X,
    θ,
    μ0,
    Tau0,
    X_in,
    # days,
    # epochs,
    # thresh_base,
    # lr_base,
    # callback = 10,
)
    θ = to_est(m, θ)
    loss_final = let loss_prev = Inf, loss = Inf, lr = m.step_size, thresh = m.threshold
        for i = 1:m.iter
            loss, ∇ = gradient_with_loss(θ) do θ
                -loglik_argo_kalman(m, X_in, from_est(m, θ), μ0, Tau0, data.days)
            end
            θ = map((p, dp) -> p - dp * lr, θ, ∇[1])
            if !ismissing(m.callback) && i % m.callback == 0
                println(i, ": loss = ", loss)
            end
            if (loss_prev - loss) < 0
                lr /= 2
                thresh *= 2
                if !ismissing(m.callback)
                    println("Alert: loss went up; halving step size to ", lr)
                end
            elseif (loss_prev - loss) / abs(loss_prev) < thresh
                if !ismissing(m.callback)
                    println("Threshold reached; breaking")
                end
                break
            end
            loss_prev = loss
        end
        if (loss_prev - loss) / abs(loss_prev) > thresh
            error_msg = "Kalman filter did not converge in " * string(epochs) * " epochs."
            error(error_msg)
        end
    end
    return from_est(m, θ)
end

function loglik_argo_kalman(m::AbstractArgoKalman, X, θ, μ0, Tau0, days)
    kalman_model = make_kalman_model(m, days, θ)
    loglik = kalman_loglik(kalman_model..., μ0, Tau0, X)
    return loglik
end

function make_kalman_model(::ArgoAR, days, θ)
    deltaf(t) = days[t] - days[t-1]
    F0(t) = [0.0, 0.0]
    F(t) = begin
        X = [
            1.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0
        ]
        return X
    end
    g0_base = [0.0, 0.0, θ.v0long, θ.v0lat]
    G0(t) = (1 - θ.α^(deltaf(t) / 10)) * g0_base
    G(t) = begin
        delta = deltaf(t)
        [
            1.0 0.0 delta 0.0
            0.0 1.0 0.0 delta
            0.0 0.0 θ.α^(delta/10) 0.0
            0.0 0.0 0.0 θ.α^(delta/10)
        ]
    end

    Tau(t) = begin
        delta = deltaf(t)
        delta * [
            θ.σ_x_long^2 0.0 0.0 0.0
            0.0 θ.σ_x_lat^2 0.0 0.0
            0.0 0.0 θ.σ_v_long^2 0.0
            0.0 0.0 0.0 θ.σ_v_lat^2
        ]
    end

    Σ(t) = θ.σ_p * [
        1.0 0.0
        0.0 1.0
    ]
    return (; F0, F, G0, G, Σ, Tau)
end


function to_est(::ArgoAR, θ)
    transforms = (
        σ_x_long = log,
        σ_x_lat = log,
        σ_v_long = log,
        σ_v_lat = log,
        σ_p = log,
        α = logit,
        v0long = identity,
        v0lat = identity,
    )
    return map((f, x) -> f(x), transforms, θ)
end

function from_est(::ArgoAR, θ)
    transforms = (
        σ_x_long = exp,
        σ_x_lat = exp,
        σ_v_long = exp,
        σ_v_lat = exp,
        σ_p = exp,
        α = logistic,
        v0long = identity,
        v0lat = identity,
    )
    return map((f, x) -> f(x), transforms, θ)
end

function from_vec(θ, θ_vec)
    return NamedTuple{keys(θ)}(θ_vec)
end

function gradient_with_loss(f, args...)
    y, back = Zygote.pullback(f, args...)
    return y, back(Zygote.sensitivity(y))
end

## Linear interpolation version
struct ArgoRW <: AbstractArgoKalman
    step_size::Float64
    iter::Int
    threshold::Float64
    rng::MersenneTwister
    callback::Union{Int,Missing}
end


function ArgoRW(;
    step_size = 1e-3,
    iter = 10000,
    threshold = 1e-6,
    rng = MersenneTwister(),
    callback = missing,
)
    ArgoRW(step_size, iter, threshold, rng, callback)
end

function set_initial_values(::ArgoRW, X_in)
    X_in = deepcopy(X_in)
    μ0 = convert(Vector{Float64}, X_in[1, :])
    Tau0 = diagm([1e-4, 1e-4])
    X_in[1, :] .= missing
    return μ0, Tau0, X_in
end

function get_initial_theta(::ArgoRW)
    θ = (σ_x_long = 3e-3, σ_x_lat = 3e-3, σ_p = 1e-4)
    return θ
end

function make_kalman_model(::ArgoRW, days, θ)
    deltaf(t) = days[t] - days[t-1]
    F0(t) = [0.0, 0.0]
    F(t) = begin
        X = [
            1.0 0.0
            0.0 1.0
        ]
        return X
    end
    G0(t) = [0.0, 0.0]
    G(t) = begin
        [
            1.0 0.0
            0.0 1.0
        ]
    end
    Tau(t) = begin
        delta = deltaf(t)
        delta * [
            θ.σ_x_long^2 0.0
            0.0 θ.σ_x_lat^2
        ]
    end

    Σ(t) = θ.σ_p * [
        1.0 0.0
        0.0 1.0
    ]
    return (; F0, F, G0, G, Tau, Σ)
end


function to_est(::ArgoRW, θ)
    transforms = (σ_x_long = log, σ_x_lat = log, σ_p = log)
    return map((f, x) -> f(x), transforms, θ)
end

function from_est(::ArgoRW, θ)
    transforms = (σ_x_long = exp, σ_x_lat = exp, σ_p = exp)
    return map((f, x) -> f(x), transforms, θ)
end

end
