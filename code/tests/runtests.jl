using Test

using Distributed
addprocs(1)

@testset "benchmark_smc" begin
    include("benchmark_smc2.jl")
end

@testset "float_chamberlain" begin
    include("float_chamberlain.jl")
end

@testset "holdout" begin
    include("holdouts.jl")
end
