module ArgoSSM

## Ugly globals
PROJECT_ROOT = string(@__DIR__) * "/../../../../"

include("./PV_source.jl")
include("./IceIndex.jl")
include("./Kalman.jl")
include("./ArgoKalman.jl")
using .ArgoKalman: fit_and_predict
include("./ArgoModels.jl")
using .ArgoModels: fit_and_predict
include("./PV_interp.jl")
using .PV_interp: fit_and_predict
include("LinearInterp.jl")
using .LinearInterp: fit_and_predict
include("./Clusters.jl")
include("./Floats.jl")
end
