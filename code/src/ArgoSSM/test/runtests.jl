using ArgoSSM
using Test
using Distributed
addprocs(2)
@everywhere using ArgoSSM

@testset "float_chamberlain" begin
    include("./float_chamberlain.jl")
end

@testset "float_chamberlain_holdouts" begin
    include("./float_chamberlain_holdouts.jl")
end
