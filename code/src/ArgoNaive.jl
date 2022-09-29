#TODO: Rename to ArgoSMC
module ArgoNaive
using Distributions
using DistributionsAD
using DataFrames

using SMC: kalman_filter_mv, kalman_smoother_mv
using SMC: smc, simulate_backward
using SMC: smc_pmcmc_proposal, smc_pmcmc_proposal_logdens, dt_smc2_estimation
using LinearAlgebra: diagm, isposdef, Symmetric
using Random: MersenneTwister
using Base.Iterators: cycle
using Distributed: pmap
include("PV_source.jl")
include("ArgoKalman.jl")
import .ArgoKalman.set_initial_values, .ArgoKalman.make_kalman_model
import .PV_source.norm2
import SMC.@NT
abstract type ArgoModel end

struct ArgoBaseline <: ArgoModel end

## Initial state

function rpr_base(::ArgoModel, rng, K, states, variances)
    XVs = states[:, 1] .+ (sqrt(Symmetric(variances[:, :, 1])) * randn(rng, 4, K))
    vcat(XVs, fill(NaN, 2, K))
end

function dpr_base(::ArgoModel, Zs, states, variances)
    XVs = view(Zs, 1:4, :)
    logpdf(MvNormal(states[:, 1], Symmetric(variances[:, :, 1])), XVs)
end

## One-step proposal

calc_norm_of_V(Vs) = sqrt.(sum(Vs .^ 2; dims = 1))
function rt_base(::ArgoModel, rng, Zs, t, h, G_X, G0_V, G_V, Tau_X, γ, Tau_PV)
    K = size(Zs, 2)
    Xs = view(Zs, 1:2, :)
    # Xs_new = view(Zs_new, 1:2, :)
    Vs = view(Zs, 3:4, :)
    # Vs_new = view(Zs_new, 3:4, :)
    XVs = view(Zs, 1:4, :)
    μX = G_X(t) * XVs
    Σ_X = Tau_X(t)
    Xs_new = μX + rand(rng, MvNormal(Σ_X), size(Xs, 2))

    g = h(Xs_new, Vs)
    @assert !any(isnan.(g))

    V_norm = calc_norm_of_V(Vs)
    g_scaled = g .* V_norm

    μV = γ(t) .* (G0_V(t) .+ G_V(t) * Vs) + (1 .- γ(t)) .* g_scaled
    Vs_new = μV + rand(MvNormal(Tau_PV(t)), size(μV, 2))

    Zs_new = vcat(Xs_new, Vs_new, g)
    return Zs_new
end

function dt_base(::ArgoModel, Zs, Zs_new, t, G_X, G0_V, G_V, Tau_X, γ, Tau_PV)
    K = size(Zs, 2)
    Xs = view(Zs, 1:2, :)
    Xs_new = view(Zs_new, 1:2, :)
    Vs = view(Zs, 3:4, :)
    Vs_new = view(Zs_new, 3:4, :)
    XVs = view(Zs, 1:4, :)
    g = view(Zs_new, 5:6, :)

    μX = G_X(t) * XVs
    Σ_X = Tau_X(t)
    d_pos = logpdf(MvNormal(Σ_X), Xs_new - μX)

    @assert !any(isnan.(g))
    # if any(isnan.(g))
    #     return fill(-Inf, size(Zs, 2))
    # end

    V_norm = calc_norm_of_V(Vs)
    g_scaled = g .* V_norm

    μV = γ(t) .* (G0_V(t) .+ G_V(t) * Vs) + (1 .- γ(t)) .* g_scaled
    @assert !any(isnan.(Vs_new))
    @assert !any(isnan.(μV))
    d_vel = logpdf(MvNormal(Tau_PV(t)), Vs_new - μV)
    @assert !any(isnan.(d_pos))
    @assert !any(isnan.(d_vel))
    return d_pos + d_vel
end

function dm_base(::ArgoModel, Zs, t, Y, Σ)
    yt = view(Y, t, :)
    if any(ismissing.(yt))
        return zeros(eltype(Zs), size(Zs, 2))
    else
        Xs = view(Zs, 1:2, :)
        d = logpdf(MvNormal(Σ(t)), Xs .- yt)
        @assert !any(isnan.(d))
        return d
    end
end

function make_kalman_proposal(::ArgoModel, X_in, θ, days; σ_v_mult = 1.0)
    μ0, Tau0, X_in = set_initial_values(X_in)
    F0, F, G0, G, Tau, Σ = make_kalman_model(
        days,
        θ.σ_x_long,
        θ.σ_x_lat,
        θ.σ_v_long * σ_v_mult,
        θ.σ_v_lat * σ_v_mult,
        θ.σ_p,
        θ.α,
        [θ.v0long, θ.v0lat],
    )
    ll, states, variances, lls = kalman_filter_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, X_in)
    states_sm, variances_sm, Covs_sm =
        kalman_smoother_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, X_in)
    _, _, G0_model, G_model, Tau_model, _ = make_kalman_model(
        days,
        θ.σ_x_long,
        θ.σ_x_lat,
        θ.σ_v_long,
        θ.σ_v_lat,
        θ.σ_p,
        θ.α,
        [θ.v0long, θ.v0lat],
    )
    G_X = t -> getindex(G_model(t), 1:2, :)
    G0_V = t -> getindex(G0_model(t), 3:4)
    G_V = t -> getindex(G_model(t), 3:4, 3:4)
    Tau_X = t -> Symmetric(getindex(Tau_model(t), 1:2, 1:2))
    Tau_PV = t -> Symmetric(getindex(Tau_model(t), 3:4, 3:4))

    return μ0,
    Tau0,
    X_in,
    G_X,
    G0_V,
    G_V,
    Tau_X,
    Tau_PV,
    Σ,
    states,
    variances,
    lls,
    states_sm,
    variances_sm,
    Covs_sm
end

function norm_innerprod(pv_long, pv_lat, d_long, d_lat)
    sqrt((pv_long / d_long)^2 + (pv_lat / d_lat)^2)
end

function grad_orthonormal!(
    pv_orth,
    x_t1,
    x_t2,
    v_t1,
    v_t2,
    long_grid,
    lat_grid,
    offsets,
    long_grad,
    lat_grad;
    normalize = true,
    d_long = 1,
    d_lat = 1,
)
    pv = bilin_interp(x_t1, x_t2, long_grid, lat_grid, offsets, long_grad, lat_grad)
    pv_orth[1] = pv[2]
    pv_orth[2] = -pv[1]
    if normalize && !((pvnorm = norm_innerprod(pv_orth[1], pv_orth[2], d_long, d_lat)) ≈ 0)
        pv_orth ./= pvnorm
    end
    return pv_orth
end

function grad_orthonormal_mat(
    Xs_new,
    Vs,
    d_long,
    d_lat,
    long_grid,
    lat_grid,
    offsets,
    long_grad,
    lat_grad,
)
    K = size(Vs, 2)
    g = zero(Vs)
    for k = 1:K
        xk1 = getindex(Xs_new, 1, k)
        xk2 = getindex(Xs_new, 2, k)
        vk1 = getindex(Vs, 1, k)
        vk2 = getindex(Vs, 2, k)
        grad_orthonormal!(
            view(g, :, k),
            xk1,
            xk2,
            vk1,
            vk2,
            long_grid,
            lat_grid,
            offsets,
            long_grad,
            lat_grad;
            d_long = d_long,
            d_lat = d_lat,
        )
    end
    return g
end

function smc_model(m::ArgoModel, θ, data, K)
    μ0,
    Tau0,
    X_in,
    G_X,
    G0_V,
    G_V,
    Tau_X,
    Tau_PV,
    Σ,
    states,
    variances,
    lls,
    states_sm,
    variances_sm,
    Covs_sm = make_kalman_proposal(m, data.X_in, θ, data.days; σ_v_mult = (1.0 / θ.γ))
    rinit(rng) = rpr_base(m, rng, K, states, variances)
    dpr(Zs) = dpr_base(m, Zs, states, variances)
    dinit = dpr

    sort!(data.PV_grad, [:long, :lat])
    long_grid = unique(data.PV_grad[:, :long])
    lat_grid = unique(data.PV_grad[:, :lat])
    long_num = groupby(data.PV_grad, :long) |> x -> combine(x, :lat => length)
    offsets = vcat(0, cumsum(long_num.lat_length)[1:(end-1)])

    h =
        (Xs_new, Vs) -> grad_orthonormal_mat(
            Xs_new,
            Vs,
            1.0,
            1.0,
            long_grid,
            lat_grid,
            offsets,
            data.PV_grad.long_grad,
            data.PV_grad.lat_grad,
        )

    γ = (t) -> θ.γ^(0.1 * (data.days[t] - data.days[t-1]))

    rp(rng, Zs, t) = rt_base(m, rng, Zs, t, h, G_X, G0_V, G_V, Tau_X, γ, Tau_PV)
    dt(Zs, Zs_new, t) = dt_base(m, Zs, Zs_new, t, G_X, G0_V, G_V, Tau_X, γ, Tau_PV)
    dp = dt
    dm(Zs, t) = dm_base(m, Zs, t, X_in, Σ)

    return @NT(rinit, rp, dinit, dpr, dp, dt, dm)
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
    v0long = identity,
    v0lat = identity,
)

function fromest(::ArgoModel, θ)
    map((f, x) -> f(x), fromest_maps, θ)
end

function toest(::ArgoModel, θ)
    map((f, x) -> f(x), toest_maps, θ)
end
# default_xv  = truncated(LogNormal(log(3e-3), 5), 0.0, 0.1)
# default_p   = truncated(LogNormal(log(1e-4), 5), 0.0, 0.1)
# default_xv  = Uniform(1e-4, 1e-1)
# default_p   = Uniform(1e-6, 1e-3)
default_γ = Beta(1.0, 1.0)
default_xlong = Gamma{Float64}(0.9054227151424368, 0.017787558270812857)
default_xlat = Gamma{Float64}(1.7938999928177333, 0.00435704634323515)
default_vlong = Gamma{Float64}(2.8220263559146757, 0.0008027464793253406)
default_vlat = Gamma{Float64}(1.3930180871749989, 0.00041853104431169606)
default_p = Gamma{Float64}(2.533490553523033, 4.6938333743396956e-5)
default_α = Beta(8.895517265265315, 0.9887058307717875)
default_v0long = Normal(0.0, 0.01)
default_v0lat = Normal(0.0, 0.01)


function dprior_smc(
    ::ArgoModel,
    σ_x_long,
    σ_x_lat,
    σ_v_long,
    σ_v_lat,
    σ_p,
    γ,
    α,
    v0long,
    v0lat;
    F_σ_x_long = default_xlong,
    F_σ_x_lat = default_xlat,
    F_σ_v_long = default_vlong,
    F_σ_v_lat = default_vlat,
    F_σ_p = default_p,
    F_γ = default_γ,
    F_α = default_α,
    F_v0long = default_v0long,
    F_v0lat = default_v0lat,
)
    logpdf(F_σ_x_long, σ_x_long) +
    logpdf(F_σ_x_lat, σ_x_lat) +
    logpdf(F_σ_v_long, σ_v_long) +
    logpdf(F_σ_v_lat, σ_v_lat) +
    logpdf(F_σ_p, σ_p) +
    logpdf(F_γ, γ) +
    logpdf(F_α, α) +
    logpdf(F_v0long, v0long) +
    logpdf(F_v0lat, v0lat)
end

function smc2_dprior(m::ArgoModel, θ)
    dprior_smc(m, fromest(m, θ)...) + sum(Tuple(θ)[1:7])
end

function smc2_rprior_notrans(
    ::ArgoModel;
    F_σ_x_long = default_xlong,
    F_σ_x_lat = default_xlat,
    F_σ_v_long = default_vlong,
    F_σ_v_lat = default_vlat,
    F_σ_p = default_p,
    F_γ = default_γ,
    F_α = default_α,
    F_v0long = default_v0long,
    F_v0lat = default_v0lat,
)

    (
        σ_x_long = rand(F_σ_x_long),
        σ_x_lat = rand(F_σ_x_lat),
        σ_v_long = rand(F_σ_v_long),
        σ_v_lat = rand(F_σ_v_lat),
        σ_p = rand(F_σ_p),
        γ = rand(F_γ),
        α = rand(F_α),
        v0long = rand(F_v0long),
        v0lat = rand(F_v0lat),
    )
end

smc2_rprior(m::ArgoModel) = toest(m, smc2_rprior_notrans(m))

smc2_lower_bound(::ArgoModel, x, n) = -Inf
function smc2_upper_bound(::ArgoModel, x, n)
    if (n == 6) || (n == 7)
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

function fit_and_predict_smc2(
    m::ArgoModel,
    X_in,
    days,
    PV_grad;
    K_smc_infer = 100,
    K_smc = 100,
    K_theta = 100,
    K_ffbs = 100,
    C_multipler = 1.0,
    kwargs...,
)
    T = size(X_in, 1)
    loglik_fn =
        (θ) -> begin
            smc_fns = smc_model(m, fromest(m, θ), @NT(X_in, days, PV_grad), K_smc_infer)
            smc(MersenneTwister(), T, smc_fns...; threshold = 0.5).loglik
        end

    thetas_prior, prior_fn, rprop_pmh, dprop_pmh = smc2_model(m, K_theta, C_multipler)

    thetas, logliks, acceptances = dt_smc2_estimation(
        thetas_prior,
        loglik_fn,
        prior_fn;
        rprop_pmh = rprop_pmh,
        dprop_pmh = dprop_pmh,
        kwargs...,
    )

    results_fn =
        (θ, K) -> begin
            rng = MersenneTwister()
            smc_fns = smc_model(m, fromest(m, θ), @NT(X_in, days, PV_grad), K_smc_infer)
            smc_out = smc(rng, T, smc_fns...; record_history = true, threshold = 0.5)
            ffbs_out = simulate_backward(
                rng,
                smc_out.particle_history,
                smc_out.logweight_history,
                smc_fns.dt,
                K,
            )
        end

    ffbs_results = pmap(results_fn, thetas, cycle([K_ffbs]))
    X_paths = cat(ffbs_results..., dims = 2)

    return @NT(thetas, logliks, acceptances, X_paths)
end

function benchmark_smc(
    m::ArgoModel,
    theta,
    X_in,
    days,
    PV_grad;
    K_smc_infer = 100,
    K_theta = 1,
    C_multipler = 1.0,
    kwargs...,
)
    T = size(X_in, 1)
    smc_fns = smc_model(m, fromest(m, theta), @NT(X_in, days, PV_grad), K_smc_infer)
    smc(MersenneTwister(), T, smc_fns...; threshold = 0.5)
end

function benchmark_smc2(
    m::ArgoModel,
    X_in,
    days,
    PV_grad;
    K_smc_infer = 100,
    K_theta = 1,
    C_multipler = 1.0,
    kwargs...,
)
    T = size(X_in, 1)
    loglik_fn =
        (θ) -> begin
            smc_fns = smc_model(m, fromest(m, θ), @NT(X_in, days, PV_grad), K_smc_infer)
            smc(MersenneTwister(), T, smc_fns...; threshold = 0.5).loglik
        end

    thetas_prior, prior_fn, rprop_pmh, dprop_pmh = smc2_model(m, K_theta, C_multipler)
    println(thetas_prior)
    logliks = map(loglik_fn, thetas_prior)

    return logliks
end

smooth_mean(ffbs_out) = mean(ffbs_out; dims = 2)[:, 1, :]
smooth_var(ffbs_out) = var(ffbs_out; dims = 2)[:, 1, :]
end
