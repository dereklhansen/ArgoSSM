using LinearAlgebra
using Random: MersenneTwister
using Distributions: MvNormal, logpdf
using StatsFuns: logsumexp

using SMC.Models: lg_smc_model, lgc_smc_model, LinearGaussian
using SMC: smc

F0(t) = [0.0, 0.0, 0.0]
G0(t) = [0.0, 0.0, 0.0]
F(t) = diagm([1.0, 1.0, 1.0])
G(t) = diagm(([1.0, 1.0, 1.0]))
Σ(t) = 9.0 * diagm(([1.0, 1.0, 1.0]))
Tau(t) = diagm([1.0, 1.0, 1.0])
μ0 = vcat(0.0, 0.0, 0.0)
Tau0 = 100.0 * diagm([1.0, 1.0, 1.0])

y = [
    5.22896384735402,
    2.059848221254017,
    11.77696919584194,
    5.721925811242492,
    2.647894413351219,
    8.833040488128724,
    10.112474414603064,
    6.208953089526956,
    7.483218024927158,
    1.1450746725213778,
    2.3827281539488214,
    2.5154588929528154,
    7.6938119335857795,
    11.267170019825972,
    11.203900809640977,
    4.641402249798912,
    6.393560809006704,
    5.434277785253692,
    7.754107613915341,
    2.8018519282190573,
    8.844915959029999,
    6.179788030963477,
    8.986422350276268,
    8.129873884597485,
    10.464689898594969,
    7.43846896751153,
    6.195006607054435,
    7.684383628629019,
    6.432819127902091,
    9.784040586681458,
    3.9386281953008275,
    1.0524045623884053,
    5.2287370014479535,
    3.3793567409786296,
    9.109449328047496,
    2.4107833351064665,
    3.1442786924940815,
    8.694980615700167,
    9.34957815303634,
    1.9899759308134595,
    1.7676330607869204,
    6.164419273710474,
    7.378003270378701,
    3.730353213883039,
    8.15997706282842,
    3.912665621137947,
    2.8222087161027987,
    7.89108559137954,
    2.016432756238303,
    -0.7283326731242079,
    5.34191426139306,
    3.7422889485477566,
    6.879539434261266,
    1.697715484841611,
    5.7492808736295835,
    3.4683312150848638,
    7.897465524526299,
    9.358577329317843,
    9.01915967340802,
    5.098410508423983,
    10.161486865442141,
    12.905026331072838,
    11.193444598591357,
    5.284455585283422,
    5.403785641006497,
    2.984985087125228,
    -0.1108114821742765,
    3.8443055205297396,
    1.1335781598255243,
    7.313228382179326,
    1.1291030218984508,
    0.8188795644543472,
    13.044240581728545,
    4.551403331575773,
    5.182374381380807,
    4.064021944452551,
    13.385904690258663,
    2.7383489638189253,
    4.272321073129483,
    1.936449829003382,
    7.579915913135302,
    6.891186573209264,
    5.696714256363617,
    2.9401339639559136,
    4.045792031292042,
    -2.69331630340043,
    4.253305831682493,
    -6.440609695250053,
    1.2224659946193035,
    3.6232261847393623,
    5.212788618742227,
    -0.5932923334934714,
    6.036279124314067,
    6.862343503023183,
    -2.9726591794689665,
    4.290653945351619,
    -2.420184182764869,
    0.6612926367123761,
    0.6312700637171165,
    -0.041609972691161345,
    -0.24839219509891972,
];

Y = hcat(y, y, y)
ll_true = -275.019326809115 * 3
K = 20
T = size(Y, 1)
rng = MersenneTwister(34)
for md in [lg_smc_model, lgc_smc_model]
    smc_fns = md(K, F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)

    Xs = @inferred smc_fns.rinit(rng)
    dprs = @inferred smc_fns.dpr(Xs)
    dm1 = @inferred smc_fns.dm(Xs, 1)
    dinits = @inferred smc_fns.dinit(Xs)
    dpres = @inferred smc_fns.dpre(Xs, 2)

    @test size(Xs) == (3, K)

    dpr_actual = logpdf(MvNormal(μ0, Tau0), Xs)
    @test dprs ≈ dpr_actual

    dm_actual = logpdf(MvNormal(Y[1, :], Σ(1)), Xs)
    @test dm1 ≈ dm_actual

    ## The adapted filter should have all weights equal to the total-data likelihood
    logweights = dprs + dm1 - dinits + dpres
    @test all(logweights .≈ ll_true)

    Xs_new = @inferred smc_fns.rp(rng, Xs, 2)
    dts = @inferred smc_fns.dt(Xs, Xs_new, 2)
    dm2 = @inferred smc_fns.dm(Xs_new, 2)
    dps = @inferred smc_fns.dp(Xs, Xs_new, 2)

    #= using Debugger =#

    #= @enter smc_fns.rp(rng, Xs, 2) =#

    #= @enter smc_fns.dt(Xs, Xs_new, 2) =#

    @test size(Xs_new) == (3, K)

    dt_actual = logpdf(MvNormal(Tau(2)), Xs_new - G(2) * Xs)
    @test dts ≈ dt_actual

    dm_actual = logpdf(MvNormal(Y[2, :], Σ(2)), Xs_new)
    @test dm2 ≈ dm_actual

    @time smc_out = @inferred smc(rng, T, smc_fns...; threshold = 0.5)

    ## There will still be some slight noise in the particle filter, so this approx may fail
    @test smc_out.loglik ≈ ll_true

    # Test out new interface
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
    smc_out = @inferred smc_filter(rng, T)
    smc_out = @time smc_filter(rng, T)
    ## There will still be some slight noise in the particle filter, so this approx may fail
    @test smc_out.loglik ≈ ll_true

    fwd_weights =
        smc_out.logweight_history[:, 27] +
        smc_fns.dpre(smc_out.particle_history[:, :, 27], 28)
    fwd_w = fwd_weights .- logsumexp(fwd_weights)
    @test all(isapprox.(fwd_w, -log(K)))

    @inferred SMC.calc_filtered_mean(smc_out)

    @inferred SMC.simulate_backward(
        rng,
        smc_out.particle_history,
        smc_out.logweight_history,
        smc_fns.dt,
        5,
    )
    @time SMC.simulate_backward(
        rng,
        smc_out.particle_history,
        smc_out.logweight_history,
        smc_fns.dt,
        5,
    )
end
