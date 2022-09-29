module IceIndex
using Base.Iterators: cycle
using HDF5: h5read, h5open, create_group
using BSON
using CodecZlib: GzipCompressorStream, GzipDecompressorStream
using ..PV_source: GridLookupTree
import ..PROJECT_ROOT
using Dates: Date, Day
using SharedArrays: SharedArray

function find_bounds(v::StepRangeLen, x)
    step_size = step(v)
    if x < minimum(v)
        ret = (0, 0)
    elseif x > maximum(v)
        ret = (length(v) + 1, length(v) + 1)
    else
        ret_float = 1.0 + (x - first(v)) / step_size
        ub = ceil(Int, ret_float)
        lb = ub - 1
        ret = step_size > 0 ? (lb, ub) : (ub, lb)
    end
    return ret
end


function get_ice_for_X(float)
    ice = Vector{Union{Float64,Missing}}(undef, size(float.X, 1))
    grid = read_concentration_for_float(float)
    for t = 1:length(ice)
        if any(ismissing.(float.X[t, :]))
            ice[t] = missing
        else
            ice[t] = lookup_ice_grid_on_day(grid, float.X[t, 1], float.X[t, 2], t)
        end
    end
    return ice
end

function read_long_lat_for_day(day, h5_folder)
    h5_file = h5_folder * "all.h5"
    date_str = string(Date(1970, 1, 1) + Day(day))

    long_grid = map(("start", "end", "delta")) do pos
        h5read(h5_file, date_str * "/" * "long_" * pos)
    end
    long = long_grid[1]:long_grid[3]:long_grid[2]

    lat_grid = map(("start", "end", "delta")) do pos
        h5read(h5_file, date_str * "/" * "lat_" * pos)
    end
    lat = lat_grid[1]:lat_grid[3]:lat_grid[2]

    return long, lat

end

function read_concentration_for_day(day, h5_folder)
    h5_file = h5_folder * "all.h5"
    concentration = h5read(h5_file, string(Date(1970, 1, 1) + Day(day)) * "/concentration")
    # return negtomissing.(concentration)
    return concentration
end

nantomissing(x) = isnan(x) ? missing : x
negtomissing(x) = x < 0 ? missing : x

function float_to_int16(x)
    if isnan(x) || ismissing(x)
        return Int16(-1)
    else
        return Int16(x)
    end
end

function read_concentration_for_float(
    float;
    h5_folder = PROJECT_ROOT * "temp/data/ice_data_concentration_h5/",
    shared = true,
)
    long, lat = read_long_lat_for_day(float.days[1], h5_folder)
    if shared
        concentration = SharedArray{Int16,3}(length(long), length(lat), length(float.days))
    else
        concentration = Array{Union{Int16,Missing},3}(
            undef,
            length(long),
            length(lat),
            length(float.days),
        )
    end
    for (k, day) in enumerate(float.days)
        concentration[:, :, k] = read_concentration_for_day(day, h5_folder)
    end
    (; long, lat, concentration)
end

function find_first_equal(l, x)
    i = searchsortedfirst(l, x)
    @assert l[i] == x
    return i
end

function lookup_ice_grid_on_day_bilinear(ice_grid, long, lat, day_idx)
    long_idxs = find_bounds(ice_grid.long, long)
    lat_idxs = find_bounds(ice_grid.lat, lat)
    if long_idxs[1] == 0 ||
       lat_idxs[1] == 0 ||
       long_idxs[1] > length(ice_grid.long) ||
       lat_idxs[1] > length(ice_grid.lat)
        return 0.0
    end

    w_longs =
        (ice_grid.long[long_idxs[2]] - long, long - ice_grid.long[long_idxs[1]]) ./
        abs(step(ice_grid.long))
    w_lats =
        (ice_grid.lat[lat_idxs[2]] - lat, lat - ice_grid.lat[lat_idxs[1]]) ./
        abs(step(ice_grid.lat))

    w11 = w_longs[1] * w_lats[1]
    w12 = w_longs[1] * w_lats[2]
    w21 = w_longs[2] * w_lats[1]
    w22 = w_longs[2] * w_lats[2]

    g11 = ice_grid.concentration[long_idxs[1], lat_idxs[1], day_idx] / 1000
    g12 = ice_grid.concentration[long_idxs[1], lat_idxs[2], day_idx] / 1000
    g21 = ice_grid.concentration[long_idxs[2], lat_idxs[1], day_idx] / 1000
    g22 = ice_grid.concentration[long_idxs[2], lat_idxs[2], day_idx] / 1000

    return w11 * g11 + w12 * g12 + w21 * g21 + w22 * g22
end


function lookup_ice_grid_on_day_nn(ice_grid, long, lat, day_idx)
    long_idxs = find_bounds(ice_grid.long, long)
    lat_idxs = find_bounds(ice_grid.lat, lat)
    if long_idxs[1] == 0 ||
       lat_idxs[1] == 0 ||
       long_idxs[1] > length(ice_grid.long) ||
       lat_idxs[1] > length(ice_grid.lat)
        return 0.0
    end
    long_bounds = (ice_grid.long[long_idxs[1]], ice_grid.long[long_idxs[2]])
    long_idx = long_idxs[argmin(abs.(long_bounds .- long))]

    lat_bounds = (ice_grid.lat[lat_idxs[1]], ice_grid.lat[lat_idxs[2]])
    lat_idx = lat_idxs[argmin(abs.(lat_bounds .- lat))]

    val = ice_grid.concentration[long_idx, lat_idx, day_idx]
    # -1 : NaN converted to negative
    # 2530: Coast line
    # 2540: Land
    # 2550: Missing

    if val < 0 || (val == 2550)
        return 0.0
    elseif val == 2530 || val == 2540
        return -1.0
    else
        if val > 1000
            @show val
        end
        return val / 1000.0
    end
end

function aggregate_h5_into_one(h5_folder)
    h5open(h5_folder * "/all.h5", "w") do file
        for f in readdir(h5_folder)
            nm = f[1:(end-3)]
            h5open(h5_folder * "/" * f, "r") do infile
                if haskey(infile, "concentration")
                    concentration = read(infile["concentration"])
                    concentration = float_to_int16.(concentration)
                    g = create_group(file, nm)
                    g["concentration"] = concentration

                    long = read(infile["long"])
                    lat = read(infile["lat"])
                    unique_long = unique(long)
                    unique_lat = unique(lat)
                    dlong = diff(unique_long)[1]
                    dlat = diff(unique_lat)[1]
                    @assert all(diff(unique_long) .≈ dlong)
                    @assert all(diff(unique_lat) .≈ dlat)

                    g["long_start"] = first(unique_long)
                    g["long_end"] = last(unique_long)
                    g["long_delta"] = dlong

                    g["lat_start"] = first(unique_lat)
                    g["lat_end"] = last(unique_lat)
                    g["lat_delta"] = dlat

                end
            end
        end
    end
end

end
