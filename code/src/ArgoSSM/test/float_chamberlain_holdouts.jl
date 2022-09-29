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
    "float-mode" => "southern_ocean_holdouts",
    "infile" => ArgoSSM.PROJECT_ROOT * "output/holdouts_under_ice.h5",
    "max-holdouts" => 1,
)
rm(cfg_override["chamberlain-outfolder"], force = true, recursive = true)
rm(tempdir, recursive = true, force = true)
mkdir(tempdir)

cfg = main(cfg_override)

## Test resulting h5 file
of = cfg["chamberlain-outfolder"]
n_holdouts = cfg["max-holdouts"]
for model in cfg["models"]
    for holdout_id = 1:n_holdouts
        read_outfile =
            (var) ->
                h5read(of * "/" * model * "/5901717/" * string(holdout_id) * ".h5", var)
        if model == "smc2"
            n_paths = cfg["K-theta"] * cfg["K-ffbs"]
            n_states = 10
        elseif model == "kalman"
            @test size(read_outfile("params_mle")) == (8,)
            n_paths = 1000
            n_states = 4
        elseif model == "linearinterp"
            n_paths = 1000
            @test size(read_outfile("params_mle")) == (2,)
            n_states = 2
        end
        if !occursin(r"pvinterp[1-2]", model)
            @test size(read_outfile("paths")) == (2, n_paths, 174)
            #27 clusters by 2 dimensions by 174 time points
            @test size(read_outfile("clusters")) == (27, 2, 174)
            #size of each cluster at each iteration of k-means
            @test size(read_outfile("cluster_sizes")) == (27,)

            @test size(read_outfile("distance_mean_t")) == (1, 174)
            @test size(read_outfile("distance_qs_t")) == (11, 174)
            @test size(read_outfile("distance_qs_t_quantiles")) == (11,)

            @test size(read_outfile("means_sm")) == (n_states, 174)
            @test size(read_outfile("variances_sm")) == (n_states, n_states, 174)
        end
        @test size(read_outfile("pred")) == (2, 174)
        @test size(read_outfile("error_km")) == (174,)
        @test size(read_outfile("holdout_mse_km")) == ()
    end
end

rm(cfg["chamberlain-outfolder"], recursive = true)
rm(tempdir, recursive = true)
