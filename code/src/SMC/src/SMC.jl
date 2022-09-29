module SMC

using Random
using Zygote
using StatsFuns
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Printf
using Distributed
using Crayons
using Base.Iterators
using Dates

include("resampler.jl")
include("smc.jl")
include("kalman.jl")
include("smc2_inference.jl")

export @NT

module Models
import ..@NT
include("models/lineargaussian.jl")
include("models/lineargaussian_cached.jl")
include("models/lineargaussian_inference.jl")
end

end
