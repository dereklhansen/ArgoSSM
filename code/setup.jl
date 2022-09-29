using Revise
using Distributed
using Pkg
Pkg.activate(string(@__DIR__))
push!(LOAD_PATH, string(@__DIR__) * "/src")
using ArgoSSM
using ArgoSSM.Floats: main

cfg_override = Dict(
    "K-infer" => 2500,
    "K-ffbs" => 5,
    "K-theta" => 200,
    "K-smc" => 5000,
    "chamberlain" => true,
    "use-particle-schedule" => true,
)

cfg_override_chamberlain_holdout = merge(
    cfg_override,
    Dict(
        "float-mode" => "southern_ocean_holdouts",
        "infile" => ArgoSSM.PROJECT_ROOT * "output/holdouts_under_ice.h5",
        "chamberlain-outfolder" =>
            ArgoSSM.PROJECT_ROOT * "./output/float_chamberlain_holdouts",
        "skip-completed" => true,
    ),
)

cfg_override_holdouts = merge(
    cfg_override,
    Dict(
        "chamberlain" => false,
        "float-mode" => "southern_ocean_holdouts",
        "infile" => ArgoSSM.PROJECT_ROOT * "output/holdouts_under_ice.h5",
        "skip-completed" => true,
    ),
)
cfg_override_all = merge(
    cfg_override,
    Dict(
        "chamberlain" => false,
        "skip-completed" => true,
        "models" => ["smc2_ice", "smc2"],
        "outfolder" => ArgoSSM.PROJECT_ROOT * "output/floats_results_predict",
    ),
)
