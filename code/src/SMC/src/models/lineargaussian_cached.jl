using LinearAlgebra: Symmetric, isposdef
using Distributions: MvNormal, logpdf
using Distributions

import ..kalman_filter_mv
import ..kalman_smoother_mv

struct LinearGaussianCached{T<:Any} <: AbstractLinearGaussian
    θ::T
end

function LinearGaussianCached(; kwargs...)
    LinearGaussianCached{typeof(kwargs.data)}(kwargs.data)
end

function Base.show(io::IO, ::LinearGaussianCached)
    print(io, "LinearGaussianCached()")
end

# Generative Model
function dpr_base(m::LinearGaussianCached, Zs)
    logpdf!(m.θ._dpr, MvNormal(m.θ.μ0, m.θ.Tau0), Zs)
end

function rpr_base(model::LinearGaussianCached, rng)
    rand(
        rng,
        MvNormal(model.θ.states[:, 1], Symmetric(model.θ.variances[:, :, 1])),
        model.θ.K,
    )
end

function dt_base(m::LinearGaussianCached, Zs, Zs_new, t)
    μ = m.θ.G(t) * Zs
    Σ = m.θ.Tau(t)
    if size(m.θ._dt, 1) != size(μ, 2)
        resize!(m.θ._dt, size(μ, 2))
    end
    d = logpdf!(m.θ._dt, MvNormal(Σ), Zs_new - μ)
    @assert !any(isnan.(d))
    return d
end

function dm_base(m::LinearGaussianCached, Zs, t)
    yt = view(m.θ.Y, t, :)
    if any(ismissing.(yt))
        return zeros(eltype(Zs), size(Zs, 2))
    else
        d = logpdf!(m.θ._dm, MvNormal(m.θ.Σ(t)), m.θ.F(t) * Zs .- yt)
        @assert !any(isnan.(d))
        return d
    end
end

# Inference Model
function rinit_base(m::LinearGaussianCached, rng)
    rand(rng, MvNormal(m.θ.states_sm[:, 1], Symmetric(m.θ.variances_sm[:, :, 1])), m.θ.K)
end

function dinit_base(m::LinearGaussianCached, Zs)
    logpdf!(
        m.θ._dinit,
        MvNormal(m.θ.states_sm[:, 1], Symmetric(m.θ.variances_sm[:, :, 1])),
        Zs,
    )
end

function calc_prop(m::LinearGaussianCached, Zs, t)
    A = @view(m.θ.Covs_sm[:, :, t-1]) / @view(m.θ.variances_sm[:, :, t-1])
    eps = m.θ._eps_prop .= (Zs .- @view(m.θ.states_sm[:, t-1]))
    μ = (A * eps) .+= @view(m.θ.states_sm[:, t])

    Σ = Symmetric(
        m.θ.variances_sm[:, :, t] -
        (m.θ.Covs_sm[:, :, t-1] / m.θ.variances_sm[:, :, t-1]) * m.θ.Covs_sm[:, :, t-1]',
    )

    @assert !any(isnan.(μ))
    @assert !any(isnan.(Σ))

    return μ, Σ
end

function rp_base(m::LinearGaussianCached, rng, Zs, t)
    μ, Σ = calc_prop(m, Zs, t)
    @assert isposdef(Σ)
    Zs_new = (Distributions.rand!(rng, MvNormal(Σ), m.θ._eps[1]) .+= μ)
    m.θ._eps[1] = m.θ._eps[2]
    m.θ._eps[2] = Zs_new
    return Zs_new
end

function dp_base(m::LinearGaussianCached, Zs, Zs_new, t)
    μ, Σ = calc_prop(m, Zs, t)
    @assert isposdef(Σ)
    d = logpdf!(m.θ._dp, MvNormal(Σ), Zs_new - μ)
    return d
end

function dpre_base(m::LinearGaussianCached, Zs, t)
    d_smooth = logpdf!(
        m.θ._dsmooth,
        MvNormal(m.θ.states_sm[:, t-1], Symmetric(m.θ.variances_sm[:, :, t-1])),
        Zs,
    )
    d_filt = logpdf!(
        m.θ._dfilt,
        MvNormal(m.θ.states[:, t-1], Symmetric(m.θ.variances[:, :, t-1])),
        Zs,
    )

    d = (d_smooth .-= d_filt)
    d .+= sum(@view(m.θ.lls[t:end]))
    @assert !any(isnan.(d))
    return d
end

# function make_kalman_proposal(F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)
#     ll, states, variances, lls = kalman_filter_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, Y)
#     states_sm, variances_sm, Covs_sm = kalman_smoother_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, Y)
#     return @NT(ll, states, variances, lls, states_sm, variances_sm, Covs_sm)
# end

function lgc_smc_model(K, F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)
    ll, states, variances, lls, states_sm, variances_sm, Covs_sm =
        make_kalman_proposal(F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)

    m = LinearGaussianCached(
        F = F,
        G = G,
        Tau = Tau,
        Σ = Σ,
        μ0 = μ0,
        Tau0 = Tau0,
        K = K,
        states = states,
        variances = variances,
        lls = lls,
        states_sm = states_sm,
        variances_sm = variances_sm,
        Covs_sm = Covs_sm,
        Y = Y,
        _eps = [
            Matrix{Float64}(undef, size(μ0, 1), K),
            Matrix{Float64}(undef, size(μ0, 1), K),
        ],
        _eps_prop = Matrix{Float64}(undef, size(μ0, 1), K),
        _dpr = Vector{Float64}(undef, K),
        _dt = Vector{Float64}(undef, K),
        _dm = Vector{Float64}(undef, K),
        _dinit = Vector{Float64}(undef, K),
        _dp = Vector{Float64}(undef, K),
        _dsmooth = Vector{Float64}(undef, K),
        _dfilt = Vector{Float64}(undef, K),
    )

    dpr(Zs) = dpr_base(m, Zs)
    dt(Zs, Zs_new, t) = dt_base(m, Zs, Zs_new, t)
    dm(Zs, t) = dm_base(m, Zs, t)

    rinit(rng) = rinit_base(m, rng)
    dinit(Zs) = dinit_base(m, Zs)
    rp(rng, Zs, t) = rp_base(m, rng, Zs, t)
    dp(Zs, Zs_new, t) = dp_base(m, Zs, Zs_new, t)

    dpre(Zs, t) = dpre_base(m, Zs, t)

    return @NT(rinit, rp, dinit, dpr, dp, dt, dm, dpre)
end
