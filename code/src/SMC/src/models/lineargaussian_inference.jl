using Distributions
using Random
using LinearAlgebra: diagm
using SMC

struct LinearGaussianInfer{T}
    α::T
    K::Int
    Y::Matrix{Float64}
    end_t::Int
end

function LinearGaussianInfer(K, Y, end_t)
    sigma_dist = LogNormal(log(9.0), 0.1)
    tau_dist = LogNormal(log(1.0), 0.1)
    return LinearGaussianInfer((σ = sigma_dist, τ = tau_dist), K, Y, end_t)
end

function rprior(m::LinearGaussianInfer, K)
    [map(rand, m.α) for _ = 1:K]
end

function dprior(m::LinearGaussianInfer, p)
    sum(map(logpdf, m.α, p))
end

function dprior(m::LinearGaussianInfer, ps::Array)
    [dprior(m, p) for p in ps]
end

function loglik_fn(m::LinearGaussianInfer, p)
    F0(t) = [0.0, 0.0, 0.0]
    G0(t) = [0.0, 0.0, 0.0]
    F(t) = diagm([1.0, 1.0, 1.0])
    G(t) = diagm(([1.0, 1.0, 1.0]))
    Σ(t) = p.σ * diagm(([1.0, 1.0, 1.0]))
    Tau(t) = diagm([p.τ, p.τ, p.τ])
    μ0 = vcat(0.0, 0.0, 0.0)
    Tau0 = 100.0 * diagm([1.0, 1.0, 1.0])

    smc_fns = lgc_smc_model(m.K, F0, F, G0, G, Tau, Σ, μ0, Tau0, m.Y)

    smc_filter = SMC.SMCModel(
        record_history = true,
        rinit = smc_fns.rinit,
        rproposal = smc_fns.rp,
        dinit = smc_fns.dinit,
        dprior = smc_fns.dpr,
        dproposal = smc_fns.dp,
        dtransition = smc_fns.dt,
        dmeasure = smc_fns.dm,
        dpre = smc_fns.dpre,
        threshold = 1.0,
    )

    return smc_filter(Random.GLOBAL_RNG, m.end_t).loglik
end

function (m::LinearGaussianInfer)(p)
    loglik_fn(m, p)
end
