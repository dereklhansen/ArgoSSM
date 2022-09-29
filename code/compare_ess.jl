# using Distributed
# using Statistics
# import DrWatson.@tagsave
# import Printf.@sprintf
# using ProfileView
using BSON
using Printf
using HDF5: h5write
using Distributed: pmap
# # using HDF5
# using Revise
# using Distributed
# using Pkg
# Pkg.activate(string(@__DIR__))
# push!(LOAD_PATH, string(@__DIR__) * "/src")
#using ArgoSSM
@everywhere using Random: MersenneTwister

using ArgoSSM.ArgoModels: fromest, make_argo_smc_model
using ArgoSSM.Floats: get_floats, get_model, args
using ArgoSSM.IceIndex: read_concentration_for_float

function compare_ess(cfg)
    cfg_args = args()
    cfg = merge(cfg_args, cfg)
    floats = get_floats(cfg)
    # models = ["smc2", "smc2_ice"]
    models = ["smc2", "smc2_pv"]
    write_to_h5(float, h_id, model, name, value) =
        h5write(cfg["outfolder"] * "/compare_ess.h5", @sprintf("%s/%d/%s/%s", float, h_id, model, name), value)
    for float in floats
        for model_name in models
        res_file = cfg["outfolder"] * "/" * model_name * "/" * float.id * "/" * string(float.holdout_id) * ".bson"
            if isfile(res_file) && (float.holdout_id == 1)
                res = BSON.load(res_file)
                thetas = res[:results].thetas
                logliks = res[:results].logliks
                theta = thetas[findmax(logliks)[2]]
                if !haskey(theta, :ice_tpr)
                    theta = (; 
                        theta[(:σ_x_long, :σ_x_lat, :σ_v_long, :σ_v_lat, :σ_p, :γ, :α)]...,
                        ice_tpr= log(0.5), ice_tnr= log(0.5), ice_mar = log(0.1),
                        theta[(:v0long, :v0lat)]...,
                    )
                end
                model = get_model(cfg, model_name)
                ice_grid = read_concentration_for_float(float)
                smc_model = make_argo_smc_model(model, float, fromest(model, theta), cfg["K-smc"], ice_grid)
                T = size(float.X, 1)
                wp = CachingPool(workers())
                results = pmap(wp, 1:100) do _
                    smc_model(MersenneTwister(), T)
                end
                clear!(wp)
                ess = hcat((res.ess for res in results)...)
                logliks = vcat((res.loglik for res in results)...)

                write_to_h5(float.id, float.holdout_id, model_name, "ess", ess)
                write_to_h5(float.id, float.holdout_id, model_name, "logliks", logliks)
            end
        end
    end
end

# function compare_ess_on_float(float, model, theta)
#     T = size(X_in, 1)
#     smc_model = make_argo_smc_model(model, float, fromest(model, theta))
#     return smc_model(MersenneTwister(), T)
# end


# map(1:210) do h
#     holdout_file = "../output/holdouts/holdouts_06_2021.h5"
#     PV_grad = PV_source.read_pv(b = 300)
#     X, days, pos_qc, float, holdout_idxs = Holdout.read_holdout_data(holdout_file, h)
#     res = BSON.load(@sprintf("../output/holdouts_scratch/holdouts_fv3_smc2_%03d.bson", h))

#     thetas = res[:res].res.thetas
#     logliks = res[:res].res.logliks

#     theta = thetas[findmax(logliks)[2]]

#     smc_argossm = ArgoModels.benchmark_smc(
#         ArgoModels.ArgoBaseline(),
#         theta,
#         X,
#         days,
#         PV_grad;
#         K_smc_infer = 1000,
#     )
#     smc_argonaive = ArgoNaive.benchmark_smc(
#         ArgoNaive.ArgoBaseline(),
#         theta,
#         X,
#         days,
#         PV_grad;
#         K_smc_infer = 1000,
#     )
#     write_to_h5(h, "argossm", "filtering_ess", smc_argossm.ess)
#     write_to_h5(h, "argonaive", "filtering_ess", smc_argonaive.ess)

#     logliks_argossm = pmap(1:1000) do i
#         sum(
#             ArgoModels.benchmark_smc(
#                 ArgoModels.ArgoBaseline(),
#                 theta,
#                 X,
#                 days,
#                 PV_grad;
#                 K_smc_infer = 1000,
#             ).logliks,
#         )
#     end

#     logliks_argonaive = pmap(1:1000) do i
#         sum(
#             ArgoNaive.benchmark_smc(
#                 ArgoNaive.ArgoBaseline(),
#                 theta,
#                 X,
#                 days,
#                 PV_grad;
#                 K_smc_infer = 1000,
#             ).loglik,
#         )
#     end
#     write_to_h5(h, "theta", "theta", collect(theta))
#     write_to_h5(h, "argossm", "logliks", logliks_argossm)
#     write_to_h5(h, "argonaive", "logliks", logliks_argonaive)

#     @show h
#     @show var(logliks_argossm)
#     @show var(logliks_argonaive)
#     @show smc_argossm.ess[end]
#     @show smc_argonaive.ess[end]
# end
