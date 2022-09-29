# Kalman filter code.
# Please see Lopes Tsay 2011 "Particle Filters and Bayesian Inference In Financial Econometrics"

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

struct KalmanModel{mutate,T<:Any}
    θ::T
end
function KalmanModel(; kwargs...)
    if haskey(kwargs, :mutate)
        mutate = kwargs[:mutate]
    else
        mutate = true
    end
    KalmanModel{mutate,typeof(values(kwargs))}(values(kwargs))
end

function Base.show(io::IO, x::KalmanModel{mutate}) where {mutate}
    print(io, "KalmanModel(θ=", x.θ, ")\n")
    print(io, "mutate=", mutate)
end

function kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    m = KalmanModel(
        F0 = F0,
        F1 = F1,
        G0 = G0,
        G1 = G1,
        Σ = Σ,
        Tau = Tau,
        μ0 = μ0,
        Tau0 = Tau0,
        mutate = true,
    )
    loglik, state = m(Y)
    return loglik, state.ms, state.Vs, state.logliks
end

function init_kalman_state(::KalmanModel{true}, T, m, V, loglik)
    ms = repeat(m, outer = (1, T))
    Vs = repeat(V, outer = (1, 1, T))
    logliks = fill(loglik, T)
    return @NT(ms, Vs, logliks)
end

function init_kalman_state(::KalmanModel{false}, T, m, V, loglik)
    return (ms = m, Vs = reshape(V, size(V, 1), size(V, 2), 1))
end

function update_kalman_state!(::KalmanModel{true}, state, t, m, V, loglik_t)
    state.ms[:, t] = m
    state.Vs[:, :, t] = V
    state.logliks[t] = loglik_t
    return state
end
cat3(args...) = cat(args...; dims = Val(3))
function update_kalman_state!(
    ::KalmanModel{false},
    state,
    t,
    m::Array{Float64,2},
    V::Array{Float64,2},
    loglik_t::Float64,
)
    if t > 1
        ms = hcat(state.ms, m)
        Vs = cat3(state.Vs, reshape(V, size(V, 1), size(V, 2), 1))
    else
        ms = m
        Vs = reshape(V, size(V, 1), size(V, 2), 1)
    end
    return (ms = ms, Vs = Vs)
end

function (model::KalmanModel)(Y)
    θ = model.θ
    T = size(Y)[1]
    m = deepcopy(reshape(θ.μ0, :, 1))
    V = deepcopy(θ.Tau0)
    P = inv(V)
    loglik = zero(θ.μ0[1])

    state = init_kalman_state(model, T, m, V, loglik)
    if !(any(ismissing.(Y[1, :])))
        e_t, Q_t, loglik_t = calc_loglik_t(θ.F0(1), θ.F1(1), θ.Σ(1), m, V, @view(Y[1, :]))
        loglik += loglik_t
        m, V = update_state(θ.F1(1), m, V, e_t, Q_t)
        state = update_kalman_state!(model, state, 1, m, V, loglik_t)
    else
        loglik += zero(loglik)
    end

    if (T > 1)
        for t = 2:T
            # Push system forward
            if !any(ismissing.(Y[t, :]))
                m, V, loglik_t = iterate_kalman_filter_mv(
                    θ.F0(t),
                    θ.F1(t),
                    θ.G0(t),
                    θ.G1(t),
                    θ.Σ(t),
                    θ.Tau(t),
                    m,
                    V,
                    @view(Y[t, :])
                )
                loglik += loglik_t
            else
                m, V = push_state_forward(θ.G0(t), θ.G1(t), θ.Tau(t), m, V)
                loglik_t = zero(loglik)
            end
            state = update_kalman_state!(model, state, t, m, V, loglik_t)
        end
    end

    return loglik, state
end

Zygote.@nograd kalman_filter_mv

function kalman_smoother_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    m = KalmanModel(
        F0 = F0,
        F1 = F1,
        G0 = G0,
        G1 = G1,
        Σ = Σ,
        Tau = Tau,
        μ0 = μ0,
        Tau0 = Tau0,
        mutate = true,
    )
    smooth(m, Y)
end

function init_smooth_state(::KalmanModel{true}, D, T, ms, Vs)
    ms_smoothed = deepcopy(ms)
    Vs_smoothed = deepcopy(Vs)

    ms_smoothed[:, 1:(end-1)] .= NaN
    Vs_smoothed[:, :, 1:(end-1)] .= NaN

    Covs_smoothed = zeros(eltype(ms_smoothed), D, D, T - 1)
    state = @NT(ms_smoothed, Vs_smoothed, Covs_smoothed)
end

function init_smooth_state(::KalmanModel{false}, D, T, ms, Vs)
    ms_smoothed = ms[:, end:end]
    Vs_smoothed = Vs[:, :, end:end]
    Covs_smoothed = Array{eltype(ms_smoothed)}(undef, D, D, 0)
    state = @NT(ms_smoothed, Vs_smoothed, Covs_smoothed)
end

function get_smooth_mean_cov(::KalmanModel{true}, state, t)
    ms_next = view(state.ms_smoothed, :, t + 1)
    Vs_next = view(state.Vs_smoothed, :, :, t + 1)
    return ms_next, Vs_next
end

function get_smooth_mean_cov(::KalmanModel{false}, state, t)
    ms_next = view(state.ms_smoothed, :, 1)
    Vs_next = view(state.Vs_smoothed, :, :, 1)
    return ms_next, Vs_next
end

function update_smooth_state!(
    ::KalmanModel{true},
    state,
    t,
    m_smoothed,
    V_smoothed,
    Cov_smoothed,
)
    state.ms_smoothed[:, t] = m_smoothed
    state.Vs_smoothed[:, :, t] = V_smoothed
    state.Covs_smoothed[:, :, t] = Cov_smoothed
    return state
end

function update_smooth_state!(
    ::KalmanModel{false},
    state,
    t,
    m_smoothed,
    V_smoothed,
    Cov_smoothed,
)
    ms_smoothed = hcat(m_smoothed, state.ms_smoothed)
    Vs_smoothed = cat3(V_smoothed, state.Vs_smoothed)
    Covs_smoothed = cat3(Cov_smoothed, state.Covs_smoothed)
    state = @NT(ms_smoothed, Vs_smoothed, Covs_smoothed)
end

function smooth(model::KalmanModel, Y)
    θ = model.θ
    T = size(Y)[1]

    loglik, filt_state = model(Y)
    ms = filt_state.ms
    Vs = filt_state.Vs
    D = size(ms, 1)

    state = init_smooth_state(model, D, T, ms, Vs)

    if (T > 1)
        for t = (T-1):-1:1
            ms_next, Vs_next = get_smooth_mean_cov(model, state, t)
            m_smoothed, V_smoothed, Cov_smoothed = backward_smooth_step_mv(
                θ.G0(t + 1),
                θ.G1(t + 1),
                θ.Tau(t + 1),
                view(ms, :, t),
                view(Vs, :, :, t),
                ms_next,
                Vs_next,
            )
            state =
                update_smooth_state!(model, state, t, m_smoothed, V_smoothed, Cov_smoothed)

        end
    end

    return state
end

Zygote.@nograd kalman_smoother_mv


function backward_smooth_step_mv(G0, G1, Tau, m_t, V_t, m_tp1_sm, V_tp1_sm)
    # One step ahead predictive
    a_tp1, R_tp1 = push_state_forward(G0, G1, Tau, m_t, V_t)
    B = (V_t * G1') / R_tp1
    m_t_sm = m_t + B * (m_tp1_sm - a_tp1)
    V_t_sm = V_t + B * (V_tp1_sm - R_tp1) * B'
    Cov_tp1_t = V_tp1_sm * B'

    return m_t_sm, V_t_sm, Cov_tp1_t
end

function can_mutate(::KalmanModel{mutate}) where {mutate}
    return mutate
end

function sample_smooth(m::KalmanModel, rng, ms, Vs)
    T = size(ms)[2]
    V_fill_in = zero(Vs[:, :, end])
    if can_mutate(m)
        Xs_smoothed = deepcopy(ms)
        Xs_smoothed[:, end] = rand(rng, MvNormal(ms[:, end], Symmetric(Vs[:, :, end])))
    else
        Xs_smoothed = rand(rng, MvNormal(ms[:, end], Symmetric(Vs[:, :, end])), 1)
    end

    if (T > 1)
        for t = (T-1):-1:1
            if can_mutate(m)
                Xs_next = view(Xs_smoothed, :, t + 1)
            else
                Xs_next = view(Xs_smoothed, :, 1)
            end
            m_smoothed, V_smoothed, _ = backward_smooth_step_mv(
                m.θ.G0(t + 1),
                m.θ.G1(t + 1),
                m.θ.Tau(t + 1),
                view(ms, :, t),
                view(Vs, :, :, t),
                Xs_next,
                V_fill_in,
            )
            if can_mutate(m)
                Xs_smoothed[:, t] = rand(rng, MvNormal(m_smoothed, Symmetric(V_smoothed)))
            else
                Xs_smoothed = hcat(
                    rand(rng, MvNormal(m_smoothed, Symmetric(V_smoothed)), 1),
                    Xs_smoothed,
                )
            end
        end
    end

    return Xs_smoothed
end

# Old interface
function draw_posterior_path(rng, ms, Vs, G0, G, Σ, Tau, μ0, Tau0)
    m = KalmanModel(G0 = G0, G1 = G, Σ = Σ, Tau = Tau, μ0 = μ0, Tau0 = Tau0, mutate = true)
    sample_smooth(m, rng, ms, Vs)
end
