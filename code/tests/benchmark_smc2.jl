using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(string(@__DIR__) * "/..")

using Statistics
import DrWatson.@tagsave
import Printf.@sprintf

include("../src/Holdout.jl")
include("../src/PV_source.jl")
include("../src/ArgoKalman.jl")
include("../src/ArgoModels.jl")
import Main.Holdout.read_holdout_data, Main.Holdout.run_holdout_prediction

h = 177
holdout_file = "../output/holdouts/holdouts_06_2021.h5"
PV_grid = PV_source.PVGrid()
X, days, pos_qc, float, holdout_idxs = Holdout.read_holdout_data(holdout_file, h)
ll = ArgoModels.benchmark_smc2(
    ArgoModels.ArgoSSM(),
    X,
    days,
    PV_grid;
    K_smc_infer = 50,
    K_theta = 1,
)
@time ll = ArgoModels.benchmark_smc2(
    ArgoModels.ArgoSSM(),
    X,
    days,
    PV_grid;
    K_smc_infer = 500,
    K_theta = 1,
)
