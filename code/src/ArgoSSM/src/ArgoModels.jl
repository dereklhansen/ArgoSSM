#TODO: Rename to ArgoSMC
module ArgoModels
using Distributions
using DistributionsAD
using DataFrames

using ..Kalman: kalman_filter_mv, kalman_smoother_mv
using SMC: smc, simulate_backward
using SMC: smc_pmcmc_proposal, smc_pmcmc_proposal_logdens, dt_smc2_estimation
using SMC: SMCModel
using LinearAlgebra: diagm, isposdef, Symmetric, I, logdet, cholesky
using Random: MersenneTwister, GLOBAL_RNG
using Base.Iterators: cycle
using Distributed
using Distributed: pmap, CachingPool, workers, clear!
import Distributed.@everywhere
using StatsBase: mean
using Statistics: cov, median
using StaticArrays:
    SizedVector, SizedMatrix, SVector, SMatrix, reinterpret, SOneTo, SUnitRange
import StaticArrays.@SMatrix
import StaticArrays.@SVector
import ..ArgoKalman.set_initial_values
import ..ArgoKalman.fit_and_predict
import ..ArgoKalman.ArgoAR
import ..PV_source.norm2
using ..PV_source: dist_earth
using ..IceIndex: read_concentration_for_float, lookup_ice_grid_on_day_nn
import SMC.@NT
abstract type ArgoModel end

struct ArgoSSM{T} <: ArgoModel where {T}
    K_infer::Int
    K_smc::Int
    K_theta::Int
    K_ffbs::Int
    particle_schedule::T
    use_ice::Bool
    use_pv::Bool
end


function ArgoSSM(;
    K_infer = 100,
    K_smc = 100,
    K_theta = 100,
    K_ffbs = 100,
    particle_schedule = (
        xis = (0.0, 0.01, 0.1, 0.3, 0.5),
        n_particles = (100, 500, 2500, 10_000, 20_000),
    ),
    use_ice = true,
    use_pv = true,
)
    return ArgoSSM(K_infer, K_smc, K_theta, K_ffbs, particle_schedule, use_ice, use_pv)
end


function fit_and_predict(m::ArgoSSM, float)
    ice_grid = read_concentration_for_float(float)
    if :holdout_idx in fieldnames(typeof(float))
        holdout_idxs = float.holdout_idx
    else
        holdout_idxs = ()
    end
    res = ArgoModels.fit_and_predict_smc2(
        m,
        float,
        ice_grid;
        K_smc_infer = m.K_infer,
        K_smc = m.K_smc,
        K_theta = m.K_theta,
        K_ffbs = m.K_ffbs,
        particle_schedule = m.particle_schedule,
        holdout_idxs = holdout_idxs,
    )
    return res
end

export fit_and_predict

function fit_and_predict_smc2(
    m::ArgoModel,
    float,
    ice_grid;
    K_smc_infer = 100,
    K_smc = 100,
    K_theta = 100,
    K_ffbs = 100,
    C_multipler = 1.0,
    particle_schedule = missing,
    holdout_idxs = (),
    kwargs...,
)

    T = size(float.X, 1)
    loglik_fn = let float = float, ice_grid = ice_grid
        (θ, n_particles) -> begin
            smc_model = make_argo_smc_model(
                m,
                float,
                fromest(m, θ),
                n_particles,
                ice_grid;
                record_history = false,
                holdout_idxs = holdout_idxs,
            )
            return smc_model(MersenneTwister(), T).loglik
        end
    end

    thetas_prior, prior_fn, rprop_pmh, dprop_pmh = smc2_model(m, K_theta, C_multipler)

    print("Running SMC2...")
    @everywhere GC.gc(true)

    if ismissing(particle_schedule)
        particle_schedule = K_smc_infer
    end

    thetas, logliks, acceptances = dt_smc2_estimation(
        thetas_prior,
        loglik_fn,
        prior_fn;
        rprop_pmh = rprop_pmh,
        dprop_pmh = dprop_pmh,
        parallelize = true,
        particle_schedule = particle_schedule,
        kwargs...,
    )
    @everywhere GC.gc(true)

    @show typeof(thetas)
    @show typeof(thetas_prior)
    println("Running FFBS...")
    wp = CachingPool(workers())
    ffbs_results = pmap(wp, thetas) do θ
        rng = MersenneTwister()
        smc_model = make_argo_smc_model(
            m,
            float,
            fromest(m, θ),
            K_smc,
            ice_grid,
            record_history = true,
            holdout_idxs = holdout_idxs,
        )
        smc_out = smc_model(rng, T)
        return simulate_backward(
            rng,
            smc_out.particle_history,
            smc_out.logweight_history,
            smc_model.θ.dtransition,
            K_ffbs,
        )
    end
    Distributed.clear!(wp)
    X_paths = cat(ffbs_results..., dims = 2)
    paths = X_paths[1:2, :, :]
    states = X_paths
    means_sm = mean(X_paths; dims = 2)[:, 1, :]
    variances_sm = cat([cov(states[:, :, t]') for t = 1:size(states, 3)]..., dims = 3)
    pred = means_sm[1:2, :]
    param_mat, param_names = make_param_mat(thetas)
    @everywhere GC.gc(true)
    return (;
        thetas,
        logliks,
        acceptances,
        paths,
        states,
        means_sm,
        variances_sm,
        pred,
        param_mat,
        param_names,
    )
end



function make_argo_smc_model(
    m::ArgoModel, float, θ, K, ice_grid; holdout_idxs = (), record_history = false,
)
    # kalman_proposal = make_kalman_proposal(m, float, θ; σ_v_mult = (1.0 / θ.γ))
    if m.use_pv
        tau_pv = 1 + 1.0 / (θ.γ)
        σ_v_mult = tau_pv
    else
        σ_v_mult = 1.0
    end
    kalman_proposal = make_kalman_proposal(m, float, θ; σ_v_mult = σ_v_mult)

    _, _, G0_model, G_model, Σ, Tau_model = make_kalman_model(m, float.days, θ)
    G_X = t -> getindex(G_model(t), SOneTo(2), :)
    G0_V = t -> getindex(G0_model(t), SUnitRange(3, 4))
    G_V = t -> getindex(G_model(t), SUnitRange(3, 4), SUnitRange(3, 4))
    Tau_X = t -> Symmetric(getindex(Tau_model(t), SOneTo(2), SOneTo(2)))
    Tau_PV = t -> Symmetric(getindex(Tau_model(t), SUnitRange(3, 4), SUnitRange(3, 4)))


    rinit(rng) = rinit_base(m, rng, K, kalman_proposal)
    dinit(Zs) = dinit_base(m, Zs, kalman_proposal)

    m0 = kalman_proposal.states[1]
    Tau0 = Symmetric(kalman_proposal.variances[1])
    dpr(Zs) = dpr_base(m, Zs, m0, Tau0)



    γ = (t) -> θ.γ^(0.1 * (float.days[t] - float.days[t-1]))
    med_pv = median(float.PV_grid)
    obs_missing = (t) -> t > 1 && any(ismissing.(view(kalman_proposal.X_in, t, :)))
    missing_at_random = (t) -> (t in holdout_idxs)

    model = (; G_X, G0_V, G_V, Σ, Tau_X, Tau_PV, PV_grid=float.PV_grid, γ, med_pv, ice_grid, obs_missing, missing_at_random, ice_tpr = θ.ice_tpr, ice_tnr = θ.ice_tnr, ice_mar = θ.ice_mar)

    rp(rng, Zs, t) = rp_base(m, rng, Zs, t, model, kalman_proposal)
    dp(Zs, Zs_new, t) = dp_base(m, Zs, Zs_new, t, model, kalman_proposal)
    dt(Zs, Zs_new, t) = dt_base(m, Zs, Zs_new, t, model)
    dm(Zs, t) = dm_base(m, Zs, t, kalman_proposal.X_in, model)

    dpre(Zs, t) = dpre_base(m, Zs, t, kalman_proposal)

    smc_filter = SMCModel(
        record_history = record_history,
        rinit = rinit,
        rproposal = rp,
        dinit = dinit,
        dprior = dpr,
        dproposal = dp,
        dtransition = dt,
        dmeasure = dm,
        dpre = dpre,
        threshold = 0.5,
    )
    return smc_filter
end

function make_kalman_model(::ArgoModel, days, θ)
    deltaf(t) = days[t] - days[t-1]
    F0(t) = [0.0, 0.0]
    F(t) = begin
        X = @SMatrix [
            1.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0
        ]
        return X
    end
    g0_base = @SVector [0.0, 0.0, θ.v0long, θ.v0lat]
    G0(t) = (1 - θ.α^(deltaf(t) / 10)) * g0_base
    G(t) = begin
        delta = deltaf(t)
        @SMatrix [
            1.0 0.0 delta 0.0
            0.0 1.0 0.0 delta
            0.0 0.0 θ.α^(delta/10) 0.0
            0.0 0.0 0.0 θ.α^(delta/10)
        ]
    end

    Tau(t) = begin
        delta = deltaf(t)
        delta * @SMatrix[
            θ.σ_x_long^2 0.0 0.0 0.0
            0.0 θ.σ_x_lat^2 0.0 0.0
            0.0 0.0 θ.σ_v_long^2 0.0
            0.0 0.0 0.0 θ.σ_v_lat^2
        ]
    end

    Σ(t) = θ.σ_p * @SMatrix [
        1.0 0.0
        0.0 1.0
    ]
    return (; F0, F, G0, G, Σ, Tau)
end

function make_kalman_proposal(m::ArgoModel, float, θ; σ_v_mult = 1.0)
    μ0, Tau0, X_in = set_initial_values(m, float.X)
    θ_proposal = map(keys(θ), θ) do key, value
        if key == :σ_v_long || key == :σ_v_lat
            value = value * σ_v_mult
        end
        return value
    end
    θ_proposal = NamedTuple{keys(θ)}(θ_proposal)
    kalman_model_proposal = make_kalman_model(m, float.days, θ_proposal)
    ll, states, variances, lls = kalman_filter_mv(kalman_model_proposal..., μ0, Tau0, X_in)
    states_sm, variances_sm, Covs_sm =
        kalman_smoother_mv(kalman_model_proposal..., μ0, Tau0, X_in)

    states = copy(reinterpret(SVector{4,eltype(states)}, vec(states)))
    variances = copy(reinterpret(SMatrix{4,4,eltype(variances),16}, vec(variances)))

    states_sm = copy(reinterpret(SVector{4,eltype(states_sm)}, vec(states_sm)))
    variances_sm =
        copy(reinterpret(SMatrix{4,4,eltype(variances_sm),16}, vec(variances_sm)))
    Covs_sm = copy(reinterpret(SMatrix{4,4,eltype(Covs_sm),16}, vec(Covs_sm)))

    return (; μ0, Tau0, X_in, states, variances, lls, states_sm, variances_sm, Covs_sm)
end

function set_initial_values(::ArgoModel, X_in)
    return set_initial_values(ArgoAR(), X_in)
end

## Initial state

function rinit_base(::ArgoModel, rng, K, kalman_proposal)
    XVs =
        kalman_proposal.states_sm[1] .+
        (sqrt(Symmetric(kalman_proposal.variances_sm[1])) * randn(rng, 4, K))
    pv = fill(NaN, 2, K)
    ice_detections = fill(0.0, 5, K)
    ice_detections[4, :] .= 1.0
    vcat(XVs, pv, ice_detections)
end

function dinit_base(::ArgoModel, Zs, kalman_proposal)
    XVs = view(Zs, 1:4, :)
    logpdf(
        MvNormal(kalman_proposal.states_sm[1], Symmetric(kalman_proposal.variances_sm[1])),
        XVs,
    )
end

function dpr_base(::ArgoModel, Zs, m0, Tau0)
    XVs = view(Zs, 1:4, :)
    logpdf(MvNormal(m0, Tau0), XVs)
end

## One-step proposal

function rp_base(m::ArgoModel, rng, Zs, t, model, kalman_proposal)
    K = size(Zs, 2)
    Zs_new = Matrix{Float64}(undef, 11, K)
    # A = prep_kalman_proposal(kalman_proposal)
    for k = 1:K
        Zs_k = SVector{11}(@view Zs[:, k])
        Zs_new[:, k] = sample_proposal(m, rng, Zs_k, t, model, kalman_proposal)
    end
    return Zs_new
end

function dp_base(m::ArgoModel, Zs, Zs_new, t, model, kalman_proposal)
    K = size(Zs, 2)
    d = Vector{Float64}(undef, K)
    for k = 1:K
        Zs_k = SVector{11}(@view Zs[:, k])
        Zs_new_k = SVector{11}(@view Zs_new[:, k])
        d[k] = density_proposal(m, Zs_k, Zs_new_k, t, model, kalman_proposal)
    end
    return d
end


function sample_proposal(m, rng, Z_k, t, model, kalman_proposal)
    X_k = @SVector [Z_k[1], Z_k[2]]
    V_k = @SVector [Z_k[3], Z_k[4]]
    XV_k = @SVector [Z_k[1], Z_k[2], Z_k[3], Z_k[4]]
    ice_k = @SVector [Z_k[7], Z_k[8], Z_k[9], Z_k[10], Z_k[11]]

    prop = calc_prop_Xk(m, XV_k, t, kalman_proposal)
    chol_X_k = cholesky(prop.Σ_X_k).L
    X_k_new = prop.μ_X_k + chol_X_k * SVector{2}((randn(rng), randn(rng)))

    if m.use_ice
        ice_new_k = ice_detection_proposal_k(m, X_k_new, ice_k, t, model)
    else
        ice_new_k = @SVector zeros(5)
    end

    if m.use_pv
        Σ_X = model.Tau_X(t)
        σ_long = sqrt(Σ_X[1, 1])
        σ_lat = sqrt(Σ_X[2, 2])
        pv_in_km = model.PV_grid(X_k_new[1], X_k_new[2])
        rlong = dist_earth(X_k_new[1], X_k_new[2], X_k_new[1] + σ_long, X_k_new[2])
        rlat = dist_earth(X_k_new[1], X_k_new[2], X_k_new[1], X_k_new[2] + σ_lat)
        g_long = (pv_in_km[1] / model.med_pv) * σ_long / rlong
        g_lat = (pv_in_km[2] / model.med_pv) * σ_lat / rlat
        g_k = @SVector [g_long, g_lat]
    else
        g_k = @SVector zeros(2)
    end

    μ_V_k, Σ_V_k = calc_prop_Vk_cond_Xk(m, X_k_new, V_k, t, g_k, model, prop)
    chol_V_k = cholesky(Σ_V_k).L
    V_k_new = μ_V_k + chol_V_k * SVector{2}((randn(rng), randn(rng)))

    return vcat(X_k_new, V_k_new, g_k, ice_new_k)
end

function density_proposal(m, Z_k, Z_new_k, t, model, kalman_proposal)
    X_k = @SVector [Z_k[1], Z_k[2]]
    V_k = @SVector [Z_k[3], Z_k[4]]
    XV_k = @SVector [Z_k[1], Z_k[2], Z_k[3], Z_k[4]]

    X_k_new = @SVector [Z_new_k[1], Z_new_k[2]]
    V_k_new = @SVector [Z_new_k[3], Z_new_k[4]]
    g_k = @SVector [Z_new_k[5], Z_new_k[6]]


    prop = calc_prop_Xk(m, XV_k, t, kalman_proposal)
    d_X_k_new = faststaticnormallogpdf(X_k_new, prop.μ_X_k, SMatrix{2,2}(prop.Σ_X_k))

    μ_V_k, Σ_V_k = calc_prop_Vk_cond_Xk(m, X_k_new, V_k, t, g_k, model, prop)
    d_V_k_new = faststaticnormallogpdf(V_k_new, μ_V_k, SMatrix{2,2}(Σ_V_k))

    return d_X_k_new + d_V_k_new
end

function calc_prop_Xk(::ArgoModel, Zs_k, t, kp)
    #kp short for kalman_proposal
    A = (kp.Covs_sm[t-1] / kp.variances_sm[t-1])
    μ_k = kp.states_sm[t] + A * (Zs_k - kp.states_sm[t-1])
    Σ_k = Symmetric(kp.variances_sm[t] - A * kp.Covs_sm[t-1]')

    @assert !any(isnan.(μ_k))
    @assert !any(isnan.(Σ_k))

    μ_X_k = μ_k[SOneTo(2)]
    Σ_X_k = SMatrix{2,2}(Symmetric(Σ_k[SOneTo(2), SOneTo(2)]))
    return (; μ_k, Σ_k, μ_X_k, Σ_X_k)
end

function ice_detection_proposal_k(m, X_k_new, ice_k, t, model)
    # ice_detections_new = zero(ice_detections)
    if ice_k[5] > 0.0
        return @SVector [0.0, 0.0, 0.0, 0.0, 1.0]
    end

    # ## Step 1: Update based on previous observation
    if model.obs_missing(t - 1) && !model.missing_at_random(t - 1)
        p_missing_at_random = model.ice_mar
        one_minus_p = 1.0 - p_missing_at_random
        ice_k_gps = @SVector [
            ice_k[1] * one_minus_p,
            ice_k[2] * one_minus_p,
            ice_k[3] * one_minus_p,
            ice_k[4] * p_missing_at_random,
            0.0,
        ]
        # p = ice_k[1] + ice_k[2] + ice_k[3] + ice_k[4]
        ice_k_gps = ice_k_gps / sum(ice_k_gps)
    else
        ice_k_gps = @SVector [0.0, 0.0, 0.0, 1.0, 0.0]
    end

    ## Step 2: propogate forward based on ice at location
    extent = get_extent_k(X_k_new, t, model)
    if extent < 0.0 # Invalid state (either coastline or land)
        ice_k_new = @SVector [0.0, 0.0, 0.0, 0.0, 1.0]
    else
        one_minus_extent = 1.0 - extent
        ice_k_new = @SVector [
            extent,
            ice_k_gps[1] * one_minus_extent,
            ice_k_gps[2] * one_minus_extent,
            (ice_k_gps[3] + ice_k_gps[4]) * one_minus_extent,
            0.0,
        ]
    end

    if !all(ice_k_new .>= 0)
        @show ice_k
        @show ice_k_gps
        @show ice_k_new
        @show extent
        @assert false
    end
    @assert sum(ice_k_new) ≈ 1.0
    return ice_k_new
end

function get_extent_k(X_k_new, t, model)
    extent_raw = lookup_ice_grid_on_day_nn(model.ice_grid, X_k_new[1], X_k_new[2], t)
    if extent_raw < 0.0
        return extent_raw
    end
    @assert extent_raw >= 0.0
    @assert extent_raw <= 1.0
    prob_tp = model.ice_tpr
    prob_fp = (1 - model.ice_tnr)
    extent = extent_raw * prob_tp + (1.0 - extent_raw) * prob_fp
    @assert extent >= 0.0
    @assert extent <= 1.0
    return extent
end

function calc_prop_Vk_cond_Xk(m::ArgoModel, X_k_new, V_k, t, g_k, model, prop)
    A = prop.Σ_k[SUnitRange(3, 4), SOneTo(2)] / prop.Σ_X_k
    B = A * prop.Σ_k[SOneTo(2), SUnitRange(3, 4)]
    Σ_V_future = prop.Σ_k[SUnitRange(3, 4), SUnitRange(3, 4)] - B

    Tau = model.Tau_PV(t)
    tau_pv = 2.0 / (model.γ(t) * (Tau[1, 1] + Tau[2, 2]))
    b = A * (@SVector [X_k_new[1] - prop.μ_X_k[1], X_k_new[2] - prop.μ_X_k[2]])
    μ_V_future_k = (@SVector [prop.μ_k[3], prop.μ_k[4]]) + b
    if m.use_pv
        μ_V_k, _ = apply_pv_gain(μ_V_future_k, Σ_V_future, g_k, tau_pv)
    else
        μ_V_k = μ_V_future_k
    end
    Σ_V_k = Σ_V_future + I * 1e-5
    return (; μ_V_k, Σ_V_k)
end

function dt_base(m::ArgoModel, Zs, Zs_new, t, model)
    K = size(Zs, 2)
    Xs = view(Zs, 1:2, :)
    Xs_new = view(Zs_new, 1:2, :)
    Vs = view(Zs, 3:4, :)
    Vs_new = view(Zs_new, 3:4, :)
    XVs = view(Zs, 1:4, :)
    g = view(Zs_new, 5:6, :)

    μX = model.G_X(t) * XVs
    Σ_X = model.Tau_X(t)
    d_pos = fast2dnormallogpdf_mat(Xs_new - μX, (0.0, 0.0), Σ_X)

    @assert !any(isnan.(g))

    μ_V = Vector{SVector{2,Float64}}(undef, K)
    Σ_V = Vector{SMatrix{2,2,Float64,4}}(undef, K)
    Tau_V = SMatrix{2,2}(model.Tau_PV(t))
    tau_pv = 2.0 / (model.γ(t) * (Tau_V[1, 1] + Tau_V[2, 2]))
    for k = 1:K
        Vs_k = @SVector [Vs[1, k], Vs[2, k]]
        μ_V_prior_k = model.G0_V(t) + model.G_V(t) * Vs_k
        g_k = @SVector [g[1, k], g[2, k]]
        if m.use_pv
            μ_V_k, Σ_V_k = apply_pv_gain(μ_V_prior_k, Tau_V, g_k, tau_pv)
        else
            μ_V_k, Σ_V_k = μ_V_prior_k, Tau_V
        end
        μ_V[k] = μ_V_k
        Σ_V[k] = Σ_V_k
    end
    d_vel = Vector{Float64}(undef, K)
    for k = 1:K
        if isposdef(Σ_V[k]) && !any(isnan.(μ_V[k]))
            Vs_new_k = @SVector [Vs_new[1, k], Vs_new[2, k]]
            d_vel[k] = faststaticnormallogpdf(Vs_new_k, μ_V[k], Σ_V[k])
        else
            d_vel[k] = -Inf
        end

    end
    @assert !any(isnan.(d_pos))
    @assert !any(isnan.(d_vel))
    @assert !any(d_vel .== Inf)
    return d_pos + d_vel
end

function dm_base(m::ArgoModel, Zs, t, Y, model)
    K = size(Zs, 2)
    d = Vector{Float64}(undef, K)
    yt = @SVector [Y[t, 1], Y[t, 2]]
    Σ = SMatrix{2,2}(model.Σ(t))
    missing_obs = any(ismissing.(yt))
    for k = 1:K
        ice_k = SVector{5}(@view Zs[7:11, k])
        if missing_obs
            log_gps_prob = 0.0
        else
            X_k = @SVector [Zs[1, k], Zs[2, k]]
            log_gps_prob = faststaticnormallogpdf(X_k, yt, Σ)
        end
        log_ice_prob = dmeasure_ice(m, t, model, missing_obs, ice_k)
        d[k] = log_gps_prob + log_ice_prob
        @assert !isnan(d[k])
    end
    return d
end

function dmeasure_ice(m::ArgoModel, t, model, missing_obs, ice_k::SVector)
    if m.use_ice 
        if missing_obs && (t > 1) && !model.missing_at_random(t)
            if ice_k[5] > 0.0
                log_ice_prob = -Inf
            else
                ice_prob = ice_k[1] + ice_k[2] + ice_k[3] + model.ice_mar * ice_k[4]
                log_ice_prob = log(ice_prob)
            end
        else
            ice_prob = (1 - model.ice_mar) * ice_k[4]
            log_ice_prob = log(ice_prob)
        end
    else
        log_ice_prob = 0.0
    end
    return log_ice_prob
end



function fast2dnormallogpdf_mat(xs, Mu, Sigma)
    mu = (Mu[1], Mu[2])
    sigma = (Sigma[1], Sigma[2], Sigma[3], Sigma[4])
    vals = zeros(eltype(xs), (size(xs, 2),))
    for i = 1:size(xs, 2)
        x = (xs[1, i], xs[2, i])
        vals[i] = fast2dnormallogpdf(x, mu, sigma)
    end
    return vals
end

function fast2dnormallogpdf(x, mu, Sigma)
    x_centered = (x[1] - mu[1], x[2] - mu[2])
    Sigma_inv, det = fast2by2inv(Sigma)
    logdet = log(det)
    if isinf(logdet)
        return -Inf
    end
    s = fast2dnorm(x_centered, Sigma_inv)
    return -(s + logdet) / 2 - log(2 * pi)
end

function faststaticnormallogpdf(x::SVector, mu::SVector, Sigma::SMatrix)
    k = length(x)
    x_centered = x - mu
    chol_Sigma = cholesky(Symmetric(Sigma)).L
    logdet_Sigma = 2 * logdet(chol_Sigma)
    if isinf(logdet_Sigma)
        return -Inf
    end
    z = (chol_Sigma \ x_centered)

    # If zi is NaN, that means that we're in a 0/0 situation.
    # We say at that point the residual is "zero"
    s = zero(z[1])
    for zi in z
        if !isnan(zi)
            s += zi^2
        end
    end
    return -(s + logdet_Sigma) / 2 - (k / 2) * log(2 * pi)
end

function apply_pv_gain(mean_prior, Tau, pv, tau_pv)
    precis_pv = (pv * pv') * tau_pv
    pv_gain = Tau * precis_pv + I
    mean = pv_gain \ mean_prior
    cov = pv_gain \ Tau
    return mean, Symmetric(cov)
end



function fast2by2matmultvec(x, y)
    return (x[1] * y[1] + x[3] * y[2], x[2] * y[1] + x[4] * y[2])
end

function fast2by2inv(x)
    det = x[1] * x[4] - x[2] * x[3]
    return (x[4] / det, -x[2] / det, -x[3] / det, x[1] / det), det
end

function fast2by2scalar(x, c)
    (x[1] * c, x[2] * c, x[3] * c, x[4] * c)
end

function fast2dnorm(x, A)
    y = fast2by2matmultvec(A, x)
    return x[1] * y[1] + x[2] * y[2]
end

function dpre_base(::ArgoModel, Zs, t, kp)
    K = size(Zs, 2)
    XVs = view(Zs, 1:4, :)
    d = zeros(eltype(XVs), K)
    ll_sum = sum(@view(kp.lls[t:end]))
    for k in K
        xv_k = SVector{4}((XVs[1, k], XVs[2, k], XVs[3, k], XVs[4, k]))
        d_smooth_k = faststaticnormallogpdf(xv_k, kp.states_sm[t-1], kp.variances_sm[t-1])
        d_filt_k = faststaticnormallogpdf(xv_k, kp.states[t-1], kp.variances[t-1])
        if d_smooth_k == -Inf || d_filt_k == -Inf
            d[k] == -Inf
        else
            d[k] = d_smooth_k - d_filt_k + ll_sum
        end
        @assert !isnan(d[k])
        @assert d[k] < Inf
    end
    return d
end



function norm_innerprod(pv_long, pv_lat, d_long, d_lat)
    sqrt((pv_long / d_long)^2 + (pv_lat / d_lat)^2)
end

function grad_orthonormal_mat(PV_grid, Xs_new, Vs, d_long, d_lat)
    K = size(Vs, 2)
    g = zero(Vs)
    for k = 1:K
        xk1 = getindex(Xs_new, 1, k)
        xk2 = getindex(Xs_new, 2, k)
        vk1 = getindex(Vs, 1, k)
        vk2 = getindex(Vs, 2, k)
        grad_orthonormal!(
            PV_grid,
            view(g, :, k),
            xk1,
            xk2,
            vk1,
            vk2;
            d_long = d_long,
            d_lat = d_lat,
        )
    end
    return g
end

function grad_orthonormal!(
    PV_grid,
    pv_orth,
    x_t1,
    x_t2,
    v_t1,
    v_t2;
    normalize = true,
    d_long = 1,
    d_lat = 1,
)
    pv = PV_grid(x_t1, x_t2)
    pv_orth[1] = pv[2]
    pv_orth[2] = -pv[1]
    if normalize && !((pvnorm = norm_innerprod(pv_orth[1], pv_orth[2], d_long, d_lat)) ≈ 0)
        pv_orth ./= pvnorm
    end
    return pv_orth
end





## SMC2 code
fromest_maps = (
    σ_x_long = exp,
    σ_x_lat = exp,
    σ_v_long = exp,
    σ_v_lat = exp,
    σ_p = exp,
    γ = exp,
    α = exp,
    ice_tpr = exp,
    ice_tnr = exp,
    ice_mar = exp,
    v0long = identity,
    v0lat = identity,
)
toest_maps = (
    σ_x_long = log,
    σ_x_lat = log,
    σ_v_long = log,
    σ_v_lat = log,
    σ_p = log,
    γ = log,
    α = log,
    ice_tpr = log,
    ice_tnr = log,
    ice_mar = log,
    v0long = identity,
    v0lat = identity,
)
fromest_logderiv_maps = Dict(
    exp => identity,
    identity => zero,
)

function fromest(::ArgoModel, θ)
    map((f, x) -> f(x), fromest_maps, θ)
end

function toest(::ArgoModel, θ)
    map((f, x) -> f(x), toest_maps, θ)
end
parameter_priors = (
    σ_x_long = Gamma{Float64}(0.9054227151424368, 0.017787558270812857),
    σ_x_lat = Gamma{Float64}(1.7938999928177333, 0.00435704634323515),
    σ_v_long = Gamma{Float64}(2.8220263559146757, 0.0008027464793253406),
    σ_v_lat = Gamma{Float64}(1.3930180871749989, 0.00041853104431169606),
    σ_p = Gamma{Float64}(2.533490553523033, 4.6938333743396956e-5),
    γ = LogNormal(0.0, 3.0),
    α = Beta(8.895517265265315, 0.9887058307717875),
    ice_tpr = Beta(9.0, 1.0),
    ice_tnr = Beta(9.0, 1.0),
    ice_mar = Beta(1.0, 9.0),
    v0long = Normal(0.0, 0.01),
    v0lat = Normal(0.0, 0.01),
)

function smc2_dprior(m::ArgoModel, θ)
    theta = fromest(m, θ)
    dens = 0.0
    for key in keys(theta)
        param_val = getproperty(theta, key)
        prior = getproperty(parameter_priors, key)
        log_dens = logpdf(prior, param_val)
        trans_dens = fromest_logderiv_maps[getproperty(fromest_maps, key)](param_val)
        dens += log_dens
        dens += trans_dens
    end
    return dens
end

function smc2_rprior_notrans(::ArgoModel)
    return map(rand, parameter_priors)
end

smc2_rprior(m::ArgoModel) = toest(m, smc2_rprior_notrans(m))

smc2_lower_bound(::ArgoModel, x, n) = -Inf
function smc2_upper_bound(::ArgoModel, x, n)
    if n in 7:10
        return 0.0
    else
        return Inf
    end
end
function smc2_model(m::ArgoModel, K_theta, C_multiplier)
    prior_fn = (θ) -> ArgoModels.smc2_dprior(m, θ)
    thetas_prior = [ArgoModels.smc2_rprior(m) for i = 1:K_theta]
    C = (2.38^2) / length(thetas_prior) * C_multiplier

    lower_bound = (x, n) -> smc2_lower_bound(m, x, n)
    upper_bound = (x, n) -> smc2_upper_bound(m, x, n)

    rprop_pmh =
        (theta, mean, cov) -> begin
            thetavec = smc_pmcmc_proposal(
                vcat(theta...),
                mean,
                cov;
                lower_bound = lower_bound,
                upper_bound = upper_bound,
                C = C,
            )
            NamedTuple{keys(theta)}(Tuple(thetavec))
        end
    dprop_pmh =
        (theta, theta_new, mean, cov) -> smc_pmcmc_proposal_logdens(
            vcat(theta...),
            vcat(theta_new...),
            mean,
            cov;
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            C = C,
        )

    return @NT(thetas_prior, prior_fn, rprop_pmh, dprop_pmh)
end



function benchmark_smc(
    m::ArgoModel,
    theta,
    X_in,
    days,
    PV_grid;
    K_smc_infer = 100,
    K_theta = 1,
    C_multipler = 1.0,
    kwargs...,
)
    T = size(X_in, 1)
    data = @NT(X_in, days)

    smc_fns = make_argo_smc_model(m, fromest(m, theta), data, PV_grid, K_smc_infer)
    smc(MersenneTwister(), T, smc_fns...; threshold = 0.5)
end

function benchmark_smc2(
    m::ArgoModel,
    X_in,
    days,
    PV_grid;
    K_smc_infer = 100,
    K_theta = 1,
    C_multipler = 1.0,
    kwargs...,
)
    T = size(X_in, 1)

    data = @NT(X_in, days)

    loglik_fn =
        (θ) -> begin
            smc_model = make_argo_smc_model(m, fromest(m, θ), data, PV_grid, K_smc_infer)
            return smc_model(GLOBAL_RNG, T).loglik
        end

    thetas_prior, prior_fn, rprop_pmh, dprop_pmh = smc2_model(m, K_theta, C_multipler)
    println(thetas_prior)
    logliks = map(loglik_fn, thetas_prior)
    println("Time spent on running particle filters:")
    @time logliks = map(loglik_fn, thetas_prior)

    return logliks
end

smooth_mean(ffbs_out) = mean(ffbs_out; dims = 2)[:, 1, :]
smooth_var(ffbs_out) = var(ffbs_out; dims = 2)[:, 1, :]

function make_param_mat(thetas)
    thetamat = hcat([collect(p) for p in thetas]...)
    thetanames = collect(string.(keys(thetas[1])))
    return thetamat, thetanames
end

end
