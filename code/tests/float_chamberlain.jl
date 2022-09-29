using Distributed
using HDF5: h5read

@everywhere using Pkg
@everywhere Pkg.activate(string(@__DIR__) * "/..")
@everywhere using ArgoSSM
using ArgoSSM.Floats: main

tempdir = string(@__DIR__) * "/../../temp/testfloats"
cfg_override = Dict(
    "K-infer" => 10,
    "K-ffbs" => 5,
    "K-theta" => 10,
    "K-smc" => 10,
    "kalman-iter" => 500,
    "chamberlain" => true,
    "chamberlain-outfolder" =>
        string(@__DIR__) * "/../../temp/chamberlain_test_results",
    "clust-max-paths" => 50,
    "temp-output-prefix" => tempdir * "/testfloat",
)
rm(cfg_override["chamberlain-outfolder"], force = true, recursive = true)
rm(tempdir, recursive = true, force = true)
mkdir(tempdir)

main(cfg_override)

## Test resulting h5 file
of = cfg_override["chamberlain-outfolder"]
for model in ["pvinterp1", "pvinterp2", "smc2", "kalman", "kalman_lininterp"]
    read_outfile = (var) -> h5read(of * "/" * model * "/5901717.h5", var)
    if model == "smc2"
        @test size(read_outfile("distance_mean_t")) == (1, 174)
        @test size(read_outfile("distance_qs_t")) == (11, 174)
        @test size(read_outfile("distance_qs_t_quantiles")) == (11,)
        n_paths = cfg_override["K-theta"] * cfg_override["K-ffbs"]
    elseif model == "kalman"
        @test size(read_outfile("params_mle")) == (8,)
        n_paths = 1000
    elseif model == "kalman_lininterp"
        @test size(read_outfile("params_mle")) == (3,)
        n_paths = 1000
    elseif occursin(r"pvinterp[1-2]", model)
        n_paths = 1
    end
    @test size(read_outfile("paths")) == (2, n_paths, 174)
    #27 clusters by 2 dimensions by 174 time points
    if !occursin(r"pvinterp[1-2]", model)
        @test size(read_outfile("clusters")) == (27, 2, 174)
        #size of each cluster at each iteration of k-means
        @test size(read_outfile("cluster_sizes")) == (27,)
    end


end
rm(cfg_override["chamberlain-outfolder"], recursive = true)
rm(tempdir, recursive = true)
