module PV_source
using SparseArrays
using NetCDF
using DataFrames
using Distances
using DelimitedFiles: readdlm
using HDF5
using Distributed
using Interpolations
import Statistics.median
dist2(x, y) = sqrt(sum((x .- y) .^ 2))
norm2(x) = dist2(x, zero(x))
epan_kern(u) = (3 / 4) * (1 - u^2)
epan_kern(x, b) = (1 / b) * epan_kern(x / b)

import ..PROJECT_ROOT

function read_depth_data(grid_file; lat_max = -60.0, long_max = 20.0)
    depth = ncread(grid_file, "Depth")
    lat = ncread(grid_file, "YC")
    long = ncread(grid_file, "XC")
    @show quantile(long[:], [0.0, 1.0])
    @show quantile(lat[:], [0.0, 1.0])
    omega = 7.292115e-5

    depth_df = DataFrame(long = long[:], lat = lat[:], depth_val = depth[:])

    depth_df = filter(
        row -> row.lat < lat_max && (row.long > 300 || row.long < long_max),
        depth_df,
    )
    depth_df.long .= ifelse.(depth_df.long .> 180, depth_df.long .- 360, depth_df.long)

    return depth_df
end

function PV_precalculate(b)
    depth = ncread("../temp/data/grid.nc", "Depth")
    lat = ncread("../temp/data/grid.nc", "YC")
    long = ncread("../temp/data/grid.nc", "XC")
    omega = 7.292115e-5

    depth_df = DataFrame(long = long[:], lat = lat[:], depth_val = depth[:])

    depth_df = filter(row -> row.lat < -50 && (row.long > 290 || row.long < 30), depth_df)
    depth_df.long .= ifelse.(depth_df.long .> 180, depth_df.long .- 360, depth_df.long)

    grid_vals = convert(Matrix, depth_df[!, [:long, :lat]])

    PV_grad = zeros(nrow(depth_df), 4)

    for i = 1:nrow(depth_df)
        PV_grad[i, :] = gradient_est(
            grid_vals[i, 1],
            grid_vals[i, 2],
            b,
            depth_df.long,
            depth_df.lat,
            depth_df.depth_val,
        )
        if (i % 5000 == 0)
            println(i, "/", nrow(depth_df))
        end
    end

    PV_grad = hcat(grid_vals, PV_grad)
    writedlm(string("../temp/data/PV", b, "_w_est.csv"), PV_grad, ',')
    PV_grad = PV_grad[:, 1:4]
    writedlm(string("../temp/data/PV", b, ".csv"), PV_grad, ',')
end

function PV_precalculate_grid(b)
    h5file = string(PROJECT_ROOT, "temp/PV_grad_", b, ".h5")

    depth = ncread(string(PROJECT_ROOT, "temp/data/grid.nc"), "Depth")
    lat = ncread(string(PROJECT_ROOT, "temp/data/grid.nc"), "YC")
    long = ncread(string(PROJECT_ROOT, "temp/data/grid.nc"), "XC")
    omega = 7.292115e-5

    depth_df = DataFrame(long = long[:], lat = lat[:], depth_val = depth[:])

    depth_df.long .= ifelse.(depth_df.long .> 180, depth_df.long .- 360, depth_df.long)
    # depth_df = filter(row -> row.lat < -50 && row.long > -70 && row.long < 40, depth_df)

    unique_longs = sort!(unique(depth_df.long))
    unique_lats = sort!(unique(depth_df.lat))
    PV_grad = Array{Float64}(undef, length(unique_longs), length(unique_lats), 4)

    grid = [(lng, lt) for lng in unique_longs, lt in unique_lats]

    grid_res = map(grid) do (lng, lt)
        gradient_est(lng, lt, b, depth_df.long, depth_df.lat, depth_df.depth_val)
    end

    for (i, _) in enumerate(unique_longs)
        for (j, _) in enumerate(unique_lats)
            PV_grad[i, j, :] = grid_res[i, j]
        end
    end

    rm(h5file; force = true)
    h5write(h5file, "unique_longs", unique_longs)
    h5write(h5file, "unique_lats", unique_lats)
    h5write(h5file, "PV_grad", PV_grad)

    return unique_longs, unique_lats, PV_grad
end

struct PVGrid{G,LLNG,LLAT,NLNG,NLAT,BLNG,BLAT}
    grid::G
    lookup_long::LLNG
    lookup_lat::LLAT
    node_long::NLNG
    node_lat::NLAT
    bounds_long::BLNG
    bounds_lat::BLAT
    median::Float64
end

function PVGrid(unique_longs::Vector, unique_lats::Vector, PV_grad::Array)
    lookup_long = GridLookupTree(unique_longs)
    lookup_lat = GridLookupTree(unique_lats)
    pv_median = calc_pv_median(PV_grad)
    return PVGrid(
        PV_grad,
        lookup_long,
        lookup_lat,
        unique_longs,
        unique_lats,
        extrema(unique_longs),
        extrema(unique_lats),
        pv_median,
    )
end

function PVGrid(; b = 300, long_min = -70.0, long_max = 40.0, lat_max = 50.0)
    # depth_df = filter(row -> row.lat < -50 && row.long > -70 && row.long < 40, depth_df)
    h5file = string(PROJECT_ROOT, "temp/PV_grad_", b, ".h5")

    unique_longs = h5read(h5file, "unique_longs")
    unique_lats = h5read(h5file, "unique_lats")
    PV_grad = h5read(h5file, "PV_grad")

    long_bounds = (
        searchsortedfirst(unique_longs, long_min),
        searchsortedlast(unique_longs, long_max),
    )
    lat_bounds = (1, searchsortedlast(unique_lats, lat_max))
    return PVGrid(
        unique_longs[long_bounds[1]:long_bounds[2]],
        unique_lats[lat_bounds[1]:lat_bounds[2]],
        PV_grad[long_bounds[1]:long_bounds[2], lat_bounds[1]:lat_bounds[2], :],
    )

end

function (pv::PVGrid)(k_long, k_lat)
    if (
        (k_long < pv.bounds_long[1]) ||
        (k_long > pv.bounds_long[2]) ||
        (k_lat < pv.bounds_lat[1]) ||
        (k_lat > pv.bounds_lat[2])
    )
        return (zero(k_long), zero(k_lat))
    end

    long_idx = pv.lookup_long(k_long) - 1
    lat_idx = pv.lookup_lat(k_lat) - 1

    ## Get the right values
    pv_grad_at_point = view(pv.grid, long_idx:(long_idx+1), lat_idx:(lat_idx+1), 1:2)

    p1_long =
        (pv.node_long[long_idx+1] - k_long) /
        (pv.node_long[long_idx+1] - pv.node_long[long_idx])
    p1_lat =
        (pv.node_lat[lat_idx+1] - k_lat) / (pv.node_lat[lat_idx+1] - pv.node_lat[lat_idx])

    p11 = p1_long * p1_lat
    p21 = (1 - p1_long) * p1_lat
    p12 = p1_long * (1 - p1_lat)
    p22 = (1 - p1_long) * (1 - p1_lat)

    grad_interp_long =
        pv_grad_at_point[1, 1, 1] * p11 +
        pv_grad_at_point[2, 1, 1] * p21 +
        pv_grad_at_point[1, 2, 1] * p12 +
        pv_grad_at_point[2, 2, 1] * p22

    grad_interp_lat =
        pv_grad_at_point[1, 1, 2] * p11 +
        pv_grad_at_point[2, 1, 2] * p21 +
        pv_grad_at_point[1, 2, 2] * p12 +
        pv_grad_at_point[2, 2, 2] * p22

    if isnan(grad_interp_long) || isnan(grad_interp_lat)
        return (zero(k_long), zero(k_lat))
    end

    return (grad_interp_long, grad_interp_lat)
end

struct GridLookupTree{G,Q<:Real,FG}
    grid::G
    min::Q
    max::Q
    step::Q
    fine_grid::FG
    grid_mapping::Array{Int}
end

function GridLookupTree(grid)
    min = minimum(grid)
    max = maximum(grid)
    step = minimum(diff(grid))
    fine_grid = range(min, max; step = step)
    grid_mapping = vcat([searchsortedfirst(grid, x) for x in fine_grid], [length(grid)])
    return GridLookupTree(grid, min, max, step, fine_grid, grid_mapping)
end

function (g::GridLookupTree)(x)
    if x > g.max || x < g.min
        return 0
    end
    i = ceil(Int, (x - g.min) / g.step) + 1
    i_grid = g.grid_mapping[i]
    l = g.grid[i_grid-1]
    if x <= l
        return i_grid - 1
    else
        return i_grid
    end
end

function nodes(pv::PVGrid)
    return (long = pv.node_long, lat = pv.node_lat)
end

function calc_pv_median(grid)
    lens = Matrix{Union{Missing,eltype(grid)}}(undef, size(grid, 1), size(grid, 2))
    for i = 1:size(grid, 1)
        for j = 1:size(grid, 2)
            if !any(isnan.(grid[i, j, :]))
                lens[i, j] = sqrt(grid[i, j, 1]^2 + grid[i, j, 2]^2)
            end
        end
    end
    return median(filter(x -> !ismissing(x), lens))
end

function median(pv::PVGrid)
    return pv.median
end

function read_pv(; b = 300, filedir = "../temp/data/")
    PV_grad = readdlm(string(filedir, "PV", b, ".csv"), ',')
    PV_grad = DataFrame(
        long = PV_grad[:, 1],
        lat = PV_grad[:, 2],
        long_grad = PV_grad[:, 3],
        lat_grad = PV_grad[:, 4],
    )
    return PV_grad
end

function dist_earth(x_long, x_lat, y_long, y_lat, r = 6378.388)
    if ismissing(x_long) || ismissing(x_lat) || ismissing(y_long) || ismissing(y_lat)
        return missing
    end
    haversine((x_long, x_lat), (y_long, y_lat), r)
end

function max_dist_long(x_long, x_lat, b, r = 6378.388)
    diff_rad = b / r
    rad2deg(diff_rad)
end

function kernel_est(k_long, k_lat, b, grid_longs, grid_lats, grid_depths)
    thresh_lat = b / 112 * 1.1
    thresh_lon = abs(b / (112 * cos((k_lat - b / 112) * pi / 180))) * 1.1
    grid_small =
        (abs.(grid_longs .- k_long) .< thresh_lon) .&
        (abs.(grid_lats .- k_lat) .< thresh_lat)

    grid_longs_small = grid_longs[grid_small]
    grid_lats_small = grid_lats[grid_small]
    grid_depths_small = grid_depths[grid_small]

    dist = broadcast(dist_earth, k_long, k_lat, grid_longs_small, grid_lats_small)
    pos_wt = dist .< b
    W = spdiagm(0 => epan_kern.(view(dist, pos_wt), b))

    diff_long = view(grid_longs_small, pos_wt) .- k_long
    diff_lat = view(grid_lats_small, pos_wt) .- k_lat
    X = hcat(
        fill(1.0, length(diff_long)),
        diff_long,
        diff_lat,
        diff_long .^ 2,
        diff_lat .^ 2,
    )
    txW = X' * W
    return (txW * X) \ (txW * grid_depths_small[pos_wt])
end

function gradient_est(
    k_long,
    k_lat,
    b,
    grid_longs,
    grid_lats,
    grid_depths,
    omega = 7.29215e-5,
)
    depth_est = kernel_est(k_long, k_lat, b, grid_longs, grid_lats, grid_depths)
    PV_grad = vcat(
        vcat(0.0, 2 * omega * cos(k_lat * pi / 180) * pi / 180 / depth_est[1]) -
        (2 * omega * sin(k_lat * pi / 180)) / (depth_est[1]^2) * depth_est[2:3],
        2 * omega * sin(k_lat * pi / 180) / depth_est[1],
        depth_est[1],
    )

    return PV_grad
end
end
