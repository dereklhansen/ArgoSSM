module LinearInterp

import ..ArgoKalman.fit_and_predict, ..ArgoKalman.gradient_with_loss
import ..ArgoModels.faststaticnormallogpdf
import ..PV_interp.get_interp_starts_and_ends
import ..PV_interp.missingtonan
using Random: MersenneTwister
using LinearAlgebra: Symmetric, cholesky, logdet
using Distributions
using DistributionsAD

import StaticArrays.@SMatrix
import StaticArrays.@SVector

struct LinearInterpModel
    step_size::Float64
    iter::Int
    threshold::Float64
    rng::MersenneTwister
    callback::Union{Int,Missing}
end

function LinearInterpModel(;
    step_size = 1e-8,
    iter = 10000,
    threshold = 1e-10,
    rng = MersenneTwister(),
    callback = missing,
)
    LinearInterpModel(step_size, iter, threshold, rng, callback)
end

function fit_and_predict(model::LinearInterpModel, data)
    # Find all missing values of X
    θ_est = fit(model, data.X, data.days)

    interp_starts, interp_ends = get_interp_starts_and_ends(data.X)

    pred = deepcopy(data.X)
    days = data.days
    for i = 1:length(interp_starts)
        X_start = data.X[interp_starts[i], :]
        X_end = data.X[interp_ends[i], :]
        t_start = interp_starts[i]
        t_end = interp_ends[i]

        day_start = days[t_start]
        day_end = days[t_end]
        day_diff = day_end - day_start

        for t = (interp_starts[i]+1):(interp_ends[i]-1)
            pred[t, :] =
                ((day_end - days[t]) / day_diff) * X_start +
                ((days[t] - day_start) / day_diff) * X_end
        end
    end
    pred = permutedims(pred, (2, 1))
    pred = convert(Matrix{Float64}, missingtonan.(pred))

    paths_out = map(1:1000) do i
        sample_path(model, data.X, days, θ_est)
    end |> ((x) -> cat(x..., dims = 3))
    paths = permutedims(paths_out, (2, 3, 1))
    paths = convert(Array{Float64,3}, missingtonan.(paths))

    T = size(pred, 2)
    variances_sm = Array{Float64,3}(undef, 2, 2, T)
    for t = 1:T
        variances_sm[:, :, t] = cov(permutedims(paths[:, :, t], (2, 1)))
    end

    states = paths
    return (;
        pred,
        paths,
        means_sm = pred,
        variances_sm = variances_sm,
        states,
        params_mle = collect(θ_est),
    )

end

function fit(m::LinearInterpModel, X, days)
    θ_initial = (σ_x_long = 1e-1, σ_x_lat = 1e-1)
    θ = to_est(m, θ_initial)
    loss_initial = -loglik(m, X, from_est(m, θ), days)
    loss_final = let loss_prev = Inf, loss = Inf, lr = m.step_size, thresh = m.threshold
        for i = 1:m.iter
            loss, ∇ = gradient_with_loss(θ) do θ
                -loglik(m, X, from_est(m, θ), days)
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
            error_msg =
                "Linear interpolation did not converge in " * string(epochs) * " epochs."
            error(error_msg)
        end
    end
    return from_est(m, θ)
end

function loglik(::LinearInterpModel, X, θ, days)
    T = size(X, 1)
    prev_loc = [X[1, 1], X[1, 2]]
    var_base = [θ.σ_x_long^2 0.0; 0.0 θ.σ_x_lat^2]
    var_zero = [0.0 0.0; 0.0 0.0]
    ll = 0.0
    let prev_loc = prev_loc, var = var_zero
        for t = 2:T
            var = var + var_base * (days[t] - days[t-1])
            if !(ismissing(X[t, 1]) || ismissing(X[t, 2]))
                X_t = view(X, t, :)
                ll += logpdf(MvNormal(Symmetric(var)), X_t - prev_loc)
                prev_loc = X_t
                var = var_zero
            end
        end
    end
    return ll
end

function sample_path(::LinearInterpModel, X, days, θ)
    path = deepcopy(X)
    interp_starts, interp_ends = get_interp_starts_and_ends(X)

    chol_base = @SMatrix[θ.σ_x_long 0.0; 0.0 θ.σ_x_lat]
    for i = 1:length(interp_starts)
        # X_start = data.X[interp_starts[i], :]
        X_end = @SVector [X[interp_ends[i], 1], X[interp_ends[i], 2]]
        # t_start = interp_starts[i]
        t_end = interp_ends[i]
        day_end = days[t_end]
        # t_diff = t_end - t_start

        for t = (interp_starts[i]+1):(interp_ends[i]-1)
            d1 = (days[t] - days[t-1])
            d = day_end - days[t]
            mean_t = (d / (d + d1)) * path[t-1, :] + (d1 / (d + d1)) * X_end
            chol_t = sqrt(d / (d + d1)) * chol_base
            path[t, :] .= mean_t + chol_t * (@SVector randn(2))
            # pred[t] = ((t_end - t)/t_diff) * X_start + ((t - t_start)/t_diff) * X_end
        end
    end
    return path
end

function to_est(::LinearInterpModel, θ)
    transforms = (σ_x_long = log, σ_x_lat = log)
    return map((f, x) -> f(x), transforms, θ)
end

function from_est(::LinearInterpModel, θ)
    transforms = (σ_x_long = exp, σ_x_lat = exp)
    return map((f, x) -> f(x), transforms, θ)
end
end
