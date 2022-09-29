# Kalman filter code.
# Please see Lopes Tsay 2011 "Particle Filters and Bayesian Inference In Financial Econometrics"
module Kalman
using LinearAlgebra
using Distributions
using DistributionsAD

function push_state_forward(G0, G1, Tau, m, V)
    a_t = G0 + G1 * m
    R_t = G1 * V * G1' + Tau
    return a_t, R_t
end

function calc_loglik_t(F0, F1, Σ, a_t, R_t, y_t)
    f_t = F0 + F1 * a_t
    Q_t = F1 * R_t * F1' + Σ
    e_t = (y_t - f_t)
    loglik = logpdf(MvNormal(Symmetric(Q_t)), e_t)[1]
    return e_t, Q_t, loglik
end

function update_state(F1, a_t, R_t, e_t, Q_t)
    # Kalman Gain
    A_t = R_t * F1' / Q_t
    m_t = a_t + A_t * e_t
    V_t = R_t - A_t * Q_t * A_t'
    return m_t, V_t
end




function iterate_kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, m, V, y_t)
    # Push the state forward
    a_t, R_t = push_state_forward(G0, G1, Tau, m, V)
    # Calculate the likelihood
    e_t, Q_t, loglik = calc_loglik_t(F0, F1, Σ, a_t, R_t, y_t)
    # Calculate the Kalman Gain
    m_t, V_t = update_state(F1, a_t, R_t, e_t, Q_t)

    return m_t, V_t, loglik
end

function kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y; mutate = true)
    T = size(Y)[1]
    m = deepcopy(μ0)
    ms = repeat(m, outer = [1, T])
    V = deepcopy(Tau0)
    Vs = repeat(V, outer = [1, 1, T])
    P = inv(V)
    loglik = zero(μ0[1])

    if mutate
        ms = repeat(m, outer = [1, T])
        Vs = repeat(V, outer = [1, 1, T])
        logliks = fill(loglik, T)
    else
        ms = m
        Vs = V
        logliks = missing
    end


    if !(any(ismissing.(Y[1, :])))
        e_t, Q_t, loglik_t = calc_loglik_t(F0(1), F1(1), Σ(1), m, V, @view(Y[1, :]))
        loglik += loglik_t
        m, V = update_state(F1(1), m, V, e_t, Q_t)
        if mutate
            ms[:, 1] = m
            Vs[:, :, 1] = V
            logliks[t] = loglik_t
        else
            ms = m
            Vs = V
            logliks = missing
        end
    else
        loglik += 0.0
    end

    if (T > 1)
        ms, Vs = let ms = ms, Vs = Vs
            for t = 2:T
                # Push system forward
                if !any(ismissing.(Y[t, :]))
                    m, V, loglik_t = iterate_kalman_filter_mv(
                        F0(t),
                        F1(t),
                        G0(t),
                        G1(t),
                        Σ(t),
                        Tau(t),
                        m,
                        V,
                        @view(Y[t, :])
                    )
                    loglik += loglik_t
                else
                    m, V = push_state_forward(G0(t), G1(t), Tau(t), m, V)
                    loglik_t = 0
                end
                if mutate
                    ms[:, t] = m
                    Vs[:, :, t] = V
                    logliks[t] = 0
                else
                    ms = hcat(ms, m)
                    Vs = cat(Vs, V; dims = 3)
                end
            end
            ms, Vs
        end
    end

    return loglik, ms, Vs, logliks
end

# Zygote.@nograd kalman_filter_mv

function kalman_loglik(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    T = size(Y)[1]
    m = deepcopy(μ0)
    V = deepcopy(Tau0)
    P = inv(V)
    loglik = zero(μ0[1])

    if !(any(ismissing.(Y[1, :])))
        e_t, Q_t, loglik_1 = calc_loglik_t(F0(1), F1(1), Σ(1), m, V, @view(Y[1, :]))
        loglik += loglik_1
        m, V = update_state(F1(1), m, V, e_t, Q_t)
        # ms[:, 1]           = m
        # Vs[:, :, 1]         = V
    else
        loglik += 0.0
    end

    if (T > 1)
        for t = 2:T
            # Push system forward
            if !any(ismissing.(Y[t, :]))
                m, V, loglik_t = iterate_kalman_filter_mv(
                    F0(t),
                    F1(t),
                    G0(t),
                    G1(t),
                    Σ(t),
                    Tau(t),
                    m,
                    V,
                    @view(Y[t, :])
                )
                loglik += loglik_t
            else
                m, V = push_state_forward(G0(t), G1(t), Tau(t), m, V)
            end
            # ms[:, t] = m
            # Vs[:, :, t] = V
        end
    end

    return loglik
end
## Smoother

function kalman_smoother_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y; mutate = true)
    T = size(Y)[1]

    loglik, ms, Vs = kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y; mutate = mutate)
    D = size(ms, 1)

    if mutate
        ms_smoothed = deepcopy(ms)
        Vs_smoothed = deepcopy(Vs)

        ms_smoothed[:, 1:(end-1)] .= NaN
        Vs_smoothed[:, :, 1:(end-1)] .= NaN

        Covs_smoothed = zeros(eltype(ms_smoothed), D, D, T - 1)
    else
        ms_smoothed = ms[:, end]
        Vs_smoothed = Vs[:, :, end]
        Covs_smoothed = Array{eltype(ms_smoothed)}(undef, D, D, 0)
    end

    if (T > 1)
        ms_smoothed, Vs_smoothed, Covs_smoothed =
            let ms_smoothed = ms_smoothed,
                Vs_smoothed = Vs_smoothed,
                Covs_smoothed = Covs_smoothed

                for t = (T-1):-1:1
                    if mutate
                        ms_next = view(ms_smoothed, :, t + 1)
                        Vs_next = view(Vs_smoothed, :, :, t + 1)
                    else
                        ms_next = view(ms_smoothed, :, 1)
                        Vs_next = view(Vs_smoothed, :, :, 1)
                    end
                    m_smoothed, V_smoothed, Cov_smoothed = backward_smooth_step_mv(
                        G0(t + 1),
                        G1(t + 1),
                        Tau(t + 1),
                        view(ms, :, t),
                        view(Vs, :, :, t),
                        ms_next,
                        Vs_next,
                    )
                    if mutate
                        ms_smoothed[:, t] = m_smoothed
                        Vs_smoothed[:, :, t] = V_smoothed
                        Covs_smoothed[:, :, t] = Cov_smoothed
                    else
                        ms_smoothed = hcat(m_smoothed, ms_smoothed)
                        Vs_smoothed = cat(V_smoothed, Vs_smoothed; dims = 3)
                        Covs_smoothed = cat(Cov_smoothed, Covs_smoothed; dims = 3)
                    end
                end
                ms_smoothed, Vs_smoothed, Covs_smoothed
            end
    end

    return ms_smoothed, Vs_smoothed, Covs_smoothed
end

# Zygote.@nograd kalman_smoother_mv


function backward_smooth_step_mv(G0, G1, Tau, m_t, V_t, m_tp1_sm, V_tp1_sm)
    # One step ahead predictive
    a_tp1, R_tp1 = push_state_forward(G0, G1, Tau, m_t, V_t)
    B = (V_t * G1') / R_tp1
    m_t_sm = m_t + B * (m_tp1_sm - a_tp1)
    V_t_sm = V_t + B * (V_tp1_sm - R_tp1) * B'
    Cov_tp1_t = V_tp1_sm * B'

    return m_t_sm, V_t_sm, Cov_tp1_t
end

function draw_posterior_path(rng, ms, Vs, G0, G1, Σ, Tau, μ0, Tau0; mutate = true)
    T = size(ms)[2]
    V_fill_in = zero(Vs[:, :, end])
    if mutate
        Xs_smoothed = deepcopy(ms)
        Xs_smoothed[:, end] = rand(rng, MvNormal(ms[:, end], Symmetric(Vs[:, :, end])))
    else
        Xs_smoothed = rand(rng, MvNormal(ms[:, end], Symmetric(Vs[:, :, end])))
    end

    if (T > 1)
        Xs_smoothed = let Xs_smoothed = Xs_smoothed
            for t = (T-1):-1:1
                if mutate
                    Xs_next = view(Xs_smoothed, :, t + 1)
                else
                    Xs_next = view(Xs_smoothed, :, 1)
                end
                m_smoothed, V_smoothed, _ = backward_smooth_step_mv(
                    G0(t + 1),
                    G1(t + 1),
                    Tau(t + 1),
                    view(ms, :, t),
                    view(Vs, :, :, t),
                    Xs_next,
                    V_fill_in,
                )
                if mutate
                    Xs_smoothed[:, t] =
                        rand(rng, MvNormal(m_smoothed, Symmetric(V_smoothed)))
                else
                    Xs_smoothed = hcat(
                        rand(rng, MvNormal(m_smoothed, Symmetric(V_smoothed))),
                        Xs_smoothed,
                    )
                end
            end
            Xs_smoothed
        end
    end

    return Xs_smoothed
end
end
