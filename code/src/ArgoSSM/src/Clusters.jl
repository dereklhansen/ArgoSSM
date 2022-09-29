#jl https://fredrikekre.github.io/Literate.jl/stable/fileformat/
module Clusters
using Random
using Distributions
using Statistics
using StatsFuns # many pdf functions
# # K - means algorithm

# ## Calculate loss

norm2_squared(x, y) = sum((x .- y) .^ 2)

# ## Main code
function run_kmeans(X, starting_means, iter, norm_squared = norm2_squared)
    @assert size(X)[2] == size(starting_means)[2]
    K = size(starting_means)[1]
    N = size(X)[1]
    Kmean_array = fill(0.0, size(starting_means)..., iter + 1)
    Kmean_array[:, :, 1] .= starting_means

    size_clust_array = fill(0.0, size(Kmean_array)[1], iter + 1)

    assign_array = Array{Int64,2}(undef, size(X)[1], iter + 1)



    for n = 1:iter
        for i = 1:N
            C_i = findmin([norm_squared(X[i, :], Kmean_array[k, :, n]) for k = 1:K])[2]
            assign_array[i, n] = C_i
            size_clust_array[C_i, n] += 1
            Kmean_array[C_i, :, n+1] .=
                ((size_clust_array[C_i, n] - 1) * Kmean_array[C_i, :, n+1] .+ X[i, :]) /
                size_clust_array[C_i, n]
        end
    end

    # Final assignment
    for i = 1:N
        C_i = findmin([norm_squared(X[i, :], Kmean_array[k, :, iter+1]) for k = 1:K])[2]
        assign_array[i, iter+1] = C_i
        size_clust_array[C_i, iter+1] += 1
    end

    return Kmean_array, size_clust_array, assign_array
end


# # EM Algorithm
function run_em_gausscluster(X, starting_means, starting_covs, iters)
    @assert size(X)[2] == size(starting_means)[2]
    @assert size(X)[2] == size(starting_covs)[1]

    K = size(starting_means)[1]
    N = size(X)[1]

    mean_array = fill(0.0, size(starting_means)..., iters + 1)
    mean_array[:, :, 1] .= starting_means

    cov_array = fill(0.0, size(starting_covs)..., iters + 1)
    cov_array[:, :, :, 1] = starting_covs

    pi_array = Array{Float64,2}(undef, K, iters + 1)
    pi_array[:, 1] .= (1 / K)
    # This array will hold the (unnormalized) log-probabilities that an observation is in each cluster
    log_weight_array = Array{Float64,3}(undef, size(X)[1], K, iters + 1)

    prob_array = Array{Float64,3}(undef, size(X)[1], K, iters + 1)

    loglik_array = fill(0.0, iters + 1)
    # This array holds representations of each component distribution
    component_array = Vector{MvNormal}(undef, K)
    for n = 1:iters
        # E-step
        for k = 1:K
            component_array[k] = MvNormal(mean_array[k, :, n], cov_array[:, :, k, n])
            for i = 1:N
                log_weight_array[i, k, n] = logpdf(component_array[k], X[i, :])
            end
        end

        # Normalize probabilities and calculate log-likelihood
        for i = 1:N
            # loglik_array[n]         += log(sum)
            log_normalizing_const =
                log(sum(exp.(log_weight_array[i, :, n] .+ log.(pi_array[:, n]))))
            loglik_array[n] += log_normalizing_const
            prob_array[i, :, n] .=
                exp.(
                    log_weight_array[i, :, n] .+ log.(pi_array[:, n]) .-
                    log_normalizing_const,
                )
        end
        # M-step
        for k = 1:K
            sum_prob = sum(prob_array[i, k, n] for i = 1:N)
            pi_array[k, n+1] = sum_prob / N
            mean_array[k, :, n+1] =
                sum(X[i, :] * prob_array[i, k, n] for i = 1:N) / sum_prob
            cov_array[:, :, k, n+1] =
                sum(X[i, :] * X[i, :]' * prob_array[i, k, n] for i = 1:N) / sum_prob
        end
    end

    # E-step
    for k = 1:K
        component_array[k] =
            MvNormal(mean_array[k, :, iters+1], cov_array[:, :, k, iters+1])
        for i = 1:N
            log_weight_array[i, k, iters+1] = logpdf(component_array[k], X[i, :])
        end
    end

    # Normalize probabilities
    for i = 1:N
        log_normalizing_const =
            log(sum(exp.(log_weight_array[i, :, iters+1] .+ log.(pi_array[:, iters+1]))))
        loglik_array[iters+1] += log_normalizing_const
        prob_array[i, :, iters+1] .=
            exp.(
                log_weight_array[i, :, iters+1] .+ log.(pi_array[:, iters+1]) .-
                log_normalizing_const,
            )
    end

    component_array, mean_array, cov_array, pi_array, prob_array, loglik_array
end

function calc_log_lik_mixture(X, mean_array, cov_array, pi_array)
    N = size(X)[1]
    K = size(cov_array)[1]

    component_array = Array{MvNormal,1}(undef, K)

    for k = 1:K
        component_array[k] = MvNormal(mean_array[k, :], cov_array[:, :, k])
    end

    loglik = 0.0
    for i = 1:N
        lik_i = 0.0
        for k = 1:K
            lik_i += pi_array[k] * pdf(component_array[k], X[i, :])
        end
        loglik += log(lik_i)
    end

    return loglik
end


# # See kmeans++ paper page (3)
using StatsBase
function kmeanspp_initial(
    seed,
    X::Matrix{T},
    K,
    norm_squared = norm2_squared,
) where {T<:Real}
    N = size(X)[1]
    weights = fill(1.0, size(X)[1])
    means = Matrix{T}(undef, K, size(X)[2])
    for k = 1:K
        means[k, :] .= X[wsample(seed, weights), :]

        # Calculate new weights
        if k == 1
            for i = 1:N
                weights[i] = norm_squared(means[k, :], X[i, :])
            end
        else
            for i = 1:N
                D_new = norm_squared(means[k, :], X[i, :])
                if (D_new < weights[i])
                    weights[i] = D_new
                end
            end
        end

    end
    return means
end
end
