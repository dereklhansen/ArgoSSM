module Floats
using Distributed
using Base.Iterators: cycle
using Statistics
import DrWatson.@tagsave
import Printf.@sprintf
using BSON: bson
using Printf
using HDF5
using Plots
using Random
using ArgParse

using ..PV_source: PVGrid, dist_earth
using ..ArgoModels: ArgoSSM, fit_and_predict
using ..ArgoKalman: ArgoAR, ArgoRW, fit_and_predict
using ..Clusters: kmeanspp_initial, run_kmeans
using ..PV_interp: PVInterp, fit_and_predict
using ..LinearInterp: LinearInterpModel, fit_and_predict

import ..PROJECT_ROOT


function main(cfg_override = missing)
    cfg = args()
    if !ismissing(cfg_override)
        cfg = merge(cfg, cfg_override)
    end
    models = cfg["models"]
    floats = get_floats(cfg)
    for model_name in models
        model = get_model(cfg, model_name)
        if typeof(model) in [ArgoAR, ArgoRW, LinearInterpModel]
            pmap(run_float, cycle([cfg]), cycle([model]), floats)
        else
            map(run_float, cycle([cfg]), cycle([model]), floats)
        end
    end
    return cfg
end

function get_floats(cfg)
    floats = h5open(cfg["infile"], "r") do h5f
        if cfg["chamberlain"]
            println("Running Chamberlain float")
            cfg["outfolder"] = cfg["chamberlain-outfolder"]
            groups = [h5f["5901717"]]
        else
            groups = h5f
        end

        if cfg["float-mode"] == "southern_ocean"
            floats = map(ArgoFloat, groups)
        elseif cfg["float-mode"] == "southern_ocean_holdouts"
            floats = vcat(
                map(get_holdouts_for_float_group, groups, cycle((cfg["holdouts"],)))...,
            )
        else
            error("float-mode=", cfg["float-mode"], " not implemented.")
        end
        floats
    end
    return floats
end

struct ArgoFloat{D,P,PVG}
    id::String
    X::Matrix{Union{Missing,Float64}}
    days::D
    pos_qc::P
    PV_grid::PVG
end

function ArgoFloat(ingroup)
    read_from_h5(name) = read(ingroup[name])
    floatid = read_from_h5("id")[1]
    X = Matrix{Union{Missing,Float64}}(read_from_h5("X"))
    days = read_from_h5("days")
    pos_qc = read_from_h5("pos_qc")
    interp = .!((pos_qc .== 1) .| (pos_qc .== 2))
    X[interp, :] .= missing
    PV_grid = PVGrid()
    return ArgoFloat(floatid, X, days, pos_qc, PV_grid)
end

function get_holdouts_for_float_group(ingroup, holdouts)
    holdout_floats = map(ArgoFloatHoldout, ingroup)
    if !ismissing(holdouts)
        holdout_floats = holdout_floats[holdouts]
    end
    return holdout_floats
end

struct ArgoFloatHoldout{D,P,PVG}
    id::String
    X_orig::Matrix{Union{Missing,Float64}}
    days::D
    pos_qc::P
    PV_grid::PVG
    X::Matrix{Union{Missing,Float64}}
    holdout_idx::Vector{Int}
    holdout_type::String
    holdout_id::Int
end

function ArgoFloatHoldout(ingroup)
    float = ArgoFloat(ingroup)
    read_from_h5(name) = read(ingroup[name])
    holdout_idx = convert(Vector{Int}, read_from_h5("holdout_idx"))
    holdout_type = read_from_h5("holdout_type")[1]
    holdout_id = convert(Vector{Int}, read_from_h5("holdout_id"))[1]
    X = deepcopy(float.X)
    X[holdout_idx, :] .= missing
    return ArgoFloatHoldout(
        float.id,
        float.X,
        float.days,
        float.pos_qc,
        float.PV_grid,
        X,
        holdout_idx,
        holdout_type,
        holdout_id,
    )
end

function get_model(cfg, model_name)
    if model_name in ("smc2", "smc2_pv", "smc2_ice")
        if cfg["use-particle-schedule"]
            particle_schedule = (
                xis = cfg["particle-schedule-xis"],
                n_particles = cfg["particle-schedule-n-particles"],
            )
        else
            particle_schedule = missing
        end
        model = ArgoSSM(;
            K_infer = cfg["K-infer"],
            K_smc = cfg["K-smc"],
            K_theta = cfg["K-theta"],
            K_ffbs = cfg["K-ffbs"],
            particle_schedule = particle_schedule,
            use_ice = model_name in ("smc2", "smc2_ice"),
            use_pv = model_name in ("smc2", "smc2_pv"),
        )
    elseif model_name == "kalman"
        model = ArgoAR(;
            step_size = cfg["kalman-learn-rate"],
            iter = cfg["kalman-iter"],
            callback = cfg["kalman-callback"],
        )
    elseif model_name == "kalman_lininterp"
        model = ArgoRW(;
            step_size = cfg["kalman-learn-rate"],
            iter = cfg["kalman-iter"],
            callback = cfg["kalman-callback"],
        )
    elseif occursin(r"pvinterp[1-2]", model_name)
        type = parse(Int, model_name[end])
        model = PVInterp(type)
    elseif model_name == "linearinterp"
        model = LinearInterpModel(
            step_size = cfg["kalman-learn-rate"],
            iter = cfg["kalman-iter"],
            callback = cfg["kalman-callback"],
        )
    end
    return model
end

function run_float(cfg, model, float)
    try
        save_prefix = get_save_prefix(model, float, cfg["outfolder"])
        bson_file = save_prefix * ".bson"
        h5_file = save_prefix * ".h5"
        if cfg["skip-completed"] && (isfile(bson_file)) && (isfile(h5_file))
            println("INFO: Skipping completed at ", bson_file)
            return nothing
        end
        results = fit_and_predict(model, float)
        # @everywhere GC.gc(true)
        metrics = get_metrics(cfg, model, float, results)
        results = (; results..., metrics...)
        save_results_to_bson(model, results, bson_file)
        save_results_to_h5(model, results, h5_file)
        return nothing
    catch e
        println("Exception caught for float ", float.id, " on model ", name(model))
        println(e)
    end
end

function get_metrics(cfg, model, float, results)
    if is_probabilistic(model)
        if any(isnan.(results.paths))
            @warn "NaNs in prediction for model " * name(model) " on float " * float.id
            clusters = (;)
            path_metrics = (;)
        else
            clusters = cluster_paths(results.paths, cfg["clust-max-paths"], cfg["nclust"])
            path_metrics = calc_path_metrics(results.paths, get_true_X(float))
        end
    else
        clusters = (;)
        path_metrics = (;)
    end
    if float isa ArgoFloatHoldout
        holdout_metrics = get_holdout_metrics(results, float)
    else
        holdout_metrics = (;)
    end
    return (; clusters..., path_metrics..., holdout_metrics...)
end

is_probabilistic(::Any) = false
is_probabilistic(::ArgoSSM) = true
is_probabilistic(::ArgoAR) = true
is_probabilistic(::ArgoRW) = true
is_probabilistic(::LinearInterpModel) = true



function cluster_paths(paths, clust_max_paths, nclust)
    n_save_particles = min(clust_max_paths, size(paths, 2))
    paths = paths[1:2, 1:n_save_particles, :]
    paths_long =
        paths[1:2, :, :] |> x -> permutedims(x, [2, 1, 3]) |> x -> reshape(x, size(x, 1), :)

    dist_earth_squared(x, y) =
        sum(dist_earth.(x[1:2:end], x[2:2:end], y[1:2:end], y[2:2:end]) .^ 2)

    start = kmeanspp_initial(MersenneTwister(231), paths_long, nclust, dist_earth_squared)
    (centers_long_array, size_clust_array, assign_array) =
        run_kmeans(paths_long, start, 1000)

    centers_long = centers_long_array[:, :, end]
    clusters = reshape(centers_long, nclust, 2, size(paths, 3))
    cluster_sizes = size_clust_array[:, end]

    return (; clusters, cluster_sizes)
end

get_true_X(float::ArgoFloat) = float.X
get_true_X(float::ArgoFloatHoldout) = float.X_orig

function get_holdout_metrics(results, float::ArgoFloatHoldout)
    y_true = permutedims(float.X_orig, (2, 1))
    y_pred = results.pred
    error_km = dist_earth.(y_true[1, :], y_true[2, :], y_pred[1, :], y_pred[2, :])
    holdout_mse_km = mean(error_km[float.holdout_idx] .^ 2)
    return (; error_km, holdout_mse_km)
end

function calc_path_metrics(paths, truth)
    qs = [0.005, 0.01, 0.05, 0.10, 0.25]
    quantiles = vcat(qs, [0.5], reverse(1 .- qs))
    means = mean(paths; dims = 2)
    dist_pred_from_mean =
        dist_earth.(paths[1, :, :], paths[2, :, :], means[1, :, :], means[2, :, :])
    dist_mean_from_truth =
        dist_earth.(truth[1, 1, :], truth[2, 1, :], means[1, 1, :], means[2, 1, :])
    ecdf = calc_empirical_dist_cdf(dist_pred_from_mean, dist_mean_from_truth)
    distance_mean_t = mean(dist_pred_from_mean; dims = 1)
    distance_qs_t = mapslices(x -> quantile(x, quantiles), dist_pred_from_mean; dims = 1)
    return (;
        dist_mean_from_truth,
        distance_mean_t,
        distance_qs_t,
        distance_qs_t_quantiles = quantiles,
        ecdf,
    )
end

function calc_empirical_dist_cdf(dist_pred_from_mean, dist_mean_from_truth)
    ecdf = zeros(eltype(dist_mean_from_truth), size(dist_pred_from_mean, 2))
    for t = 1:size(dist_pred_from_mean, 2)
        ecdf[t] = mean(dist_pred_from_mean[:, t] .<= dist_mean_from_truth[t])
    end
    return ecdf
end

function get_save_prefix(model, float::ArgoFloat, outfolder)
    save_path = outfolder * "/" * name(model) * "/"
    if !isdir(save_path)
        mkpath(save_path)
    end
    save_prefix = save_path * float.id
    return save_prefix
end

function get_save_prefix(model, float::ArgoFloatHoldout, outfolder)
    save_path = outfolder * "/" * name(model) * "/" * float.id * "/"
    if !isdir(save_path)
        mkpath(save_path)
    end
    save_prefix = save_path * string(float.holdout_id)
    return save_prefix
end

function save_results_to_bson(model, results, outfile)
    output = Dict(:model => name(model), :results => results)
    bson(outfile, output)
    return nothing
end

function save_results_to_h5(model, results, outfile)
    println("Saving results to outfile: ", outfile)
    for key in keys(results)
        value = make_h5_friendly(getproperty(results, key))
        if applicable(HDF5.datatype, value)
            h5write(outfile, string(key), value)
        end
    end
end

function make_h5_friendly(x)
    return x
end

function make_h5_friendly(x::AbstractArray{Missing})
    missing_to_other.(x, NaN)
end

function make_h5_friendly(x::AbstractArray{Union{Missing,Q}}) where {Q}
    nan_val = convert(Q, NaN)
    missing_to_other.(x, nan_val)
end

function missing_to_other(x, other)
    if ismissing(x)
        y = other
    else
        y = x
    end
    return y
end

function name(m::ArgoSSM)
    if m.use_ice && m.use_pv
        nm = "smc2"
    elseif m.use_ice
        nm = "smc2_ice"
    elseif m.use_pv
        nm = "smc2_pv"
    else
        nm = "smc2_ar"
    end
    return nm
end

name(::ArgoAR) = "kalman"
name(::ArgoRW) = "kalman_lininterp"
name(m::PVInterp) = @sprintf "pvinterp%d" m.type
name(m::LinearInterpModel) = "linearinterp"

function args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--float-mode"
        help = "Mode for type of experiment to run"
        arg_type = String
        default = "southern_ocean"
        "--models"
        help = "String indicating which model(s) to run"
        arg_type = Vector{String}
        default = [
            "pvinterp1",
            "pvinterp2",
            "linearinterp",
            "smc2",
            "smc2_ice",
            "smc2_pv",
            "kalman",
        ]
        "--infile"
        help = "H5 File from which to load float data"
        arg_type = String
        default = PROJECT_ROOT * "output/floats.h5"
        "--outfolder"
        help = "Folder in which to store results"
        arg_type = String
        default = PROJECT_ROOT * "./output/floats_results"
        "--chamberlain"
        help = "Run the example float from Chamberlain et al 2018"
        action = :store_true
        "--chamberlain-outfolder"
        help = "Folder to store the Chamberlain results"
        arg_type = String
        default = PROJECT_ROOT * "./output/float_chamberlain"
        "--nclust"
        help = "Number of clusters of paths to form after (for visualization)"
        arg_type = Int
        default = 27
        "--clust-max-paths"
        help = "Max number of paths to enter into clustering"
        arg_type = Int
        default = 1000
        "--skip-completed"
        help = "Skip runs that already have files present (instead of running and erroring out)"
        action = :store_true
    end
    add_arg_group!(s, "Kalman")
    @add_arg_table! s begin
        "--kalman-learn-rate"
        help = "Learning rate for gradient descent"
        arg_type = Float64
        default = 1e-3
        "--kalman-iter"
        help = "Maximum number of iterations for gradient descent"
        arg_type = Int
        default = 10_000_000
        "--kalman-callback"
        help = "Number of steps to report the current training loss"
        arg_type = Int
        default = 100
    end

    add_arg_group!(s, "SMC2")
    @add_arg_table! s begin
        "--K-infer"
        help = "Number of particles for inference"
        arg_type = Int
        default = 1000
        "--K-smc"
        help = "Number of particles for forward filtering prediction"
        arg_type = Int
        default = 100
        "--K-theta"
        help = "Number of parameter particles"
        arg_type = Int
        default = 200
        "--K-ffbs"
        help = "Number of particles for backward smoothing prediction"
        arg_type = Int
        default = 5
        "--use-particle-schedule"
        help = "Use MCMC particle schedule within SMC2?"
        action = :store_true
        "--particle-schedule-xis"
        help = "Increasing sequence of tempering coefficients to trigger schedule"
        arg_type = Vector{Float64}
        default = [0.0, 0.01, 0.1, 0.3, 0.5]
        "--particle-schedule-n-particles"
        help = "Increasing sequence of number of particles to schedule"
        arg_type = Vector{Int}
        default = [100, 500, 2500, 10_000, 20_000]
    end
    add_arg_group!(s, "Holdouts")
    @add_arg_table! s begin
        "--holdouts"
        help = "Indicies of holdouts to run"
        arg_type = Union{Vector{Int},Missing}
        default = missing
    end
    parsed_args = parse_args(s)
    return parsed_args
end

end
