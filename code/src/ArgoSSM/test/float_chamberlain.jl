using Distributed
using HDF5: h5read
using Test

# @everywhere using Pkg
# @everywhere Pkg.activate(string(@__DIR__) * "/..")
# @everywhere using ArgoSSM
using ArgoSSM.Floats: main

tempdir = ArgoSSM.PROJECT_ROOT * "temp/testfloats"
cfg_override = Dict(
    "K-infer" => 10,
    "K-ffbs" => 5,
    "K-theta" => 10,
    "K-smc" => 10,
    "kalman-iter" => 500,
    "chamberlain" => true,
    "chamberlain-outfolder" => ArgoSSM.PROJECT_ROOT * "temp/chamberlain_test_results",
    "clust-max-paths" => 50,
    "temp-output-prefix" => tempdir * "/testfloat",
)
rm(cfg_override["chamberlain-outfolder"], force = true, recursive = true)
rm(tempdir, recursive = true, force = true)
mkdir(tempdir)

cfg = main(cfg_override)

## Test resulting h5 file
of = cfg_override["chamberlain-outfolder"]
T = 174
for model in cfg["models"]
    read_outfile = (var) -> h5read(of * "/" * model * "/5901717.h5", var)
    if model in ("smc2", "smc2_ice", "smc2_pv")
        @test size(read_outfile("distance_mean_t")) == (1, T)
        @test size(read_outfile("distance_qs_t")) == (11, T)
        @test size(read_outfile("distance_qs_t_quantiles")) == (11,)
        n_paths = cfg_override["K-theta"] * cfg_override["K-ffbs"]
    elseif model == "kalman"
        @test size(read_outfile("params_mle")) == (8,)
        n_paths = 1000
    elseif model == "linearinterp"
        @test size(read_outfile("params_mle")) == (2,)
        n_paths = 1000
    elseif occursin(r"pvinterp[1-2]", model)
        n_paths = 1
    end
    @test size(read_outfile("paths")) == (2, n_paths, T)
    #27 clusters by 2 dimensions by 174 time points
    if !occursin(r"pvinterp[1-2]", model)
        @test size(read_outfile("clusters")) == (27, 2, T)
        #size of each cluster at each iteration of k-means
        @test size(read_outfile("cluster_sizes")) == (27,)
    end


end
rm(cfg_override["chamberlain-outfolder"], recursive = true)
rm(tempdir, recursive = true)
