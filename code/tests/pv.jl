using HDF5
using Test
using Interpolations
@everywhere using Pkg
@everywhere Pkg.activate(string(@__DIR__, "/.."))
@everywhere dirinclude(x) = include(string(@__DIR__, "/../", x))
@everywhere dirinclude("src/PV_source.jl")

#= unique_longs, unique_lats, PV_grad = Main.PV_source.PV_precalculate_grid(300) =#
h5file = string(@__DIR__, "/../../temp/PV_grad_300.h5")
unique_longs = h5read(h5file, "unique_longs")
unique_lats = h5read(h5file, "unique_lats")
PV_grad = h5read(h5file, "PV_grad")

## set up pv grid
pv_grid = PV_source.PVGrid(unique_longs, unique_lats, PV_grad)
@inferred pv_grid(0.0, -60.0)
@inferred pv_grid(10.0, -52.0)

pv_linear_long =
    interpolate((unique_longs, unique_lats), PV_grad[:, :, 1], Gridded(Linear()))
pv_linear_lat =
    interpolate((unique_longs, unique_lats), PV_grad[:, :, 2], Gridded(Linear()))

interp = @inferred(pv_linear_long(0.0, -60.0)), pv_linear_lat(0.0, -60.0)
close = PV_grad[
    searchsortedfirst(unique_longs, 0.0),
    searchsortedfirst(unique_lats, -60.0),
    1:2,
]

pv_grid.lookup_lat(-60.0)
pv_grid.lookup_long(0.0)

pv_grid(0.0, -60.0)
