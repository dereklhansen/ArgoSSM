module PV_interp

using Distances: Haversine, pairwise

import ..ArgoKalman.fit_and_predict
import ..PV_source: PVGrid, nodes

struct PVInterp
    type::Int
    n_steps_along::Int
    n_steps_desc::Int
    step_size::Float64
end

function PVInterp(type; n_steps_along = 400, n_steps_desc = 300, step_size = 0.03)
    PVInterp(type, n_steps_along, n_steps_desc, step_size)
end

function fit_and_predict(model::PVInterp, data)
    # Get all intervals of X which are missing
    interp_starts, interp_ends = get_interp_starts_and_ends(data.X)

    X_interp = deepcopy(data.X)
    for i = 1:length(interp_starts)
        pos_initial = data.X[interp_starts[i], :]
        pos_recover = data.X[interp_ends[i], :]
        t_miss = data.days[interp_starts[i]:interp_ends[i]]

        @assert !any(ismissing, pos_initial)
        @assert !any(ismissing, pos_recover)


        PV_results = pv_interp(
            pos_initial,
            pos_recover,
            t_miss,
            model.n_steps_desc,
            model.n_steps_along,
            model.step_size,
            data.PV_grid,
        )

        if model.type == 1
            X_interp[interp_starts[i]:interp_ends[i], :] = PV_results[:, 1:2]
        elseif model.type == 2
            X_interp[interp_starts[i]:interp_ends[i], :] = PV_results[:, 4:5]
        end
    end

    X_interp = convert(Matrix{Float64}, missingtonan.(X_interp))
    pred = permutedims(X_interp, (2, 1))
    paths = reshape(pred, (size(pred, 1), 1, size(pred, 2)))
    states = paths
    return (; pred, paths, states)

end

missingtonan(x) = ismissing(x) ? NaN : x

function get_interp_starts_and_ends(X)
    # Find all missing values of X
    missing_X = any(ismissing, X; dims = 2)

    # Get all intervals of X which are missing
    interp_starts, interp_ends, missing_streak =
        let missing_streak = false,
            interp_starts = Vector{Int}(),
            interp_ends = Vector{Int}()

            for t = 1:length(missing_X)
                if missing_X[t] && !missing_streak
                    missing_streak = true
                    push!(interp_starts, t - 1)
                elseif !missing_X[t] && missing_streak
                    missing_streak = false
                    push!(interp_ends, t)
                end
            end
            if interp_starts[1] == 0
                deleteat!(interp_starts, 1)
                deleteat!(interp_ends, 1)
            end
            interp_starts, interp_ends, missing_streak
        end
    if missing_streak
        pop!(interp_starts)
    end
    @assert length(interp_starts) == length(interp_ends)
    return interp_starts, interp_ends
end


function hold_out_grad_interp(
    X_in,
    holdout_idxs,
    days,
    grad_df,
    n_steps_along,
    n_steps_desc,
    step_size,
)
    X_interp1 = deepcopy(X_in)
    X_interp2 = deepcopy(X_in)
    n_removed = length(holdout_idxs)
    nonmissX = any(x -> !ismissing(x), X_in; dims = 2)[:]
    idx_init = findlast(nonmissX[1:(holdout_idxs[1]-1)])
    idx_rec_shift = findfirst(nonmissX[(holdout_idxs[end]+1):end])
    if isnothing(idx_init) || isnothing(idx_rec_shift)
        X_interp1[holdout_idxs, :] .= missing
        X_interp2[holdout_idxs, :] .= missing
        return [X_interp1, X_interp2]
    else
        idx_rec = holdout_idxs[end] + idx_rec_shift
        holdout_idxs = (idx_init+1):(idx_rec-1)
        pos_initial = X_in[idx_init, :]
        pos_recover = X_in[idx_rec, :]
        t_miss = days[idx_init:idx_rec]
        # t_miss = seq(0,1,200) would give finer grain path of results,
        # instead of only times when a profile was collected under ice

        @assert !any(ismissing, pos_initial)
        @assert !any(ismissing, pos_recover)
        PV_results = pv_interp(
            pos_initial,
            pos_recover,
            t_miss,
            n_steps_desc,
            n_steps_along,
            step_size,
            grad_df,
        )
        PV_results = PV_results[2:(size(PV_results)[1]-1), :]
        X_interp1[holdout_idxs, :] = PV_results[:, 1:2]
        X_interp2[holdout_idxs, :] = PV_results[:, 4:5]
        return [X_interp1, X_interp2]
    end
end

function pv_interp(
    pos_initial,
    pos_recover,
    t_miss,
    n_steps_desc,
    n_steps_along,
    step_size,
    PV_grid,
)
    t_miss_scale = (t_miss .- t_miss[1]) ./ (t_miss[size(t_miss)[1]] .- t_miss[1])

    # take axes out, with descent at the initial position
    grad_desc_t_init = gradient_algo(
        pos_initial[1],
        pos_initial[2],
        n_steps_desc,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        true,
        false,
    )
    grad_desc_t_init[isnan.(grad_desc_t_init)] .= 50

    grad_desc_f_init = gradient_algo(
        pos_initial[1],
        pos_initial[2],
        n_steps_desc,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        false,
        false,
    )
    grad_desc_f_init[isnan.(grad_desc_f_init)] .= 20

    # and along gradient at the recovered position
    grad_along_t_reco = gradient_algo(
        pos_recover[1],
        pos_recover[2],
        n_steps_along,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        true,
        true,
    )
    grad_along_t_reco[isnan.(grad_along_t_reco)] .= -10
    grad_along_f_reco = gradient_algo(
        pos_recover[1],
        pos_recover[2],
        n_steps_along,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        false,
        true,
    )
    grad_along_f_reco[isnan.(grad_along_f_reco)] .= 80 # ensure far enough apart

    # find closest points between the axes
    PV_interp1 = PV_axes_interp(grad_desc_t_init, grad_along_t_reco, t_miss_scale)
    PV_interp2 = PV_axes_interp(grad_desc_f_init, grad_along_t_reco, t_miss_scale)
    PV_interp3 = PV_axes_interp(grad_desc_t_init, grad_along_f_reco, t_miss_scale)
    PV_interp4 = PV_axes_interp(grad_desc_f_init, grad_along_f_reco, t_miss_scale)
    # choose the PV interpolation for which the axes are closest/intersect
    PV_dists = [PV_interp1[2]; PV_interp2[2]; PV_interp3[2]; PV_interp4[2]]
    PV_coords = [[PV_interp1[1]] [PV_interp2[1]] [PV_interp3[1]] [PV_interp4[1]]]
    PV_final_1 = PV_coords[.!isnan.(PV_dists)][argmin(PV_dists[.!isnan.(PV_dists)])]
    PV_final_1 = PV_final_1[end:-1:1, :]

    # Repeat, but switching the two positions
    grad_desc_t_reco = gradient_algo(
        pos_recover[1],
        pos_recover[2],
        n_steps_desc,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        true,
        false,
    )
    grad_desc_t_reco[isnan.(grad_desc_t_reco)] .= -10

    grad_desc_f_reco = gradient_algo(
        pos_recover[1],
        pos_recover[2],
        n_steps_desc,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        false,
        false,
    )
    grad_desc_f_reco[isnan.(grad_desc_f_reco)] .= -10

    grad_along_t_init = gradient_algo(
        pos_initial[1],
        pos_initial[2],
        n_steps_along,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        true,
        true,
    )
    grad_along_t_init[isnan.(grad_along_t_init)] .= 80
    grad_along_f_init = gradient_algo(
        pos_initial[1],
        pos_initial[2],
        n_steps_along,
        step_size,
        nodes(PV_grid).long,
        nodes(PV_grid).lat,
        PV_grid,
        false,
        true,
    )
    grad_along_f_init[isnan.(grad_along_f_init)] .= 80 # ensure far enough apart

    PV_interp21 = PV_axes_interp(grad_desc_t_reco, grad_along_t_init, t_miss_scale)
    PV_interp22 = PV_axes_interp(grad_desc_f_reco, grad_along_t_init, t_miss_scale)
    PV_interp23 = PV_axes_interp(grad_desc_t_reco, grad_along_f_init, t_miss_scale)
    PV_interp24 = PV_axes_interp(grad_desc_f_reco, grad_along_f_init, t_miss_scale)
    PV_dists = [PV_interp21[2]; PV_interp22[2]; PV_interp23[2]; PV_interp24[2]]
    PV_coords = [[PV_interp21[1]] [PV_interp22[1]] [PV_interp23[1]] [PV_interp24[1]]]
    PV_final_2 = PV_coords[.!isnan.(PV_dists)][argmin(PV_dists[.!isnan.(PV_dists)])]
    return [[PV_final_1 t_miss] [PV_final_2 t_miss]]
end

function gradient_algo(
    long_init,
    lat_init,
    iter,
    step_size,
    unique_long,
    unique_lat,
    pv_grid,
    opp,
    along,
)


    # long_seq = fill!(Array(Float64,iter+1,1),NaN)
    # long_seq = nans(iter+1, 1)
    long_seq = fill(NaN, iter + 1, 1)
    lat_seq = fill(NaN, iter + 1, 1)
    # long_seq    = Array{Union{NaN, Float64}}(missing, iter+1, 1)
    # lat_seq    = Array{Union{Missing, Float64}}(missing, iter+1, 1)
    long_seq[1] = long_init
    lat_seq[1] = lat_init

    for k = 1:iter
        if (
            (lat_seq[k] < (minimum(unique_lat) + 1 / 6)) |
            (lat_seq[k] > (maximum(unique_lat) - 1 / 6)) |
            (long_seq[k] < (minimum(unique_long) + 1 / 6)) |
            (long_seq[k] > (maximum(unique_long) - 1 / 6)) |
            isnan(long_seq[k]) |
            isnan(lat_seq[k])
        )
            break
        end

        gradient = collect(pv_grid(long_seq[k], lat_seq[k]))
        gradient = gradient ./ sqrt(gradient[1]^2 + gradient[2]^2)
        if (along)
            gradient = [gradient[2] -gradient[1]]
        end
        if (opp)
            gradient[1:2] = -gradient[1:2]
        end
        long_seq[k+1] = long_seq[k] - gradient[1] * step_size
        lat_seq[k+1] = lat_seq[k] - gradient[2] * step_size
    end
    return Matrix([long_seq lat_seq])
end

function PV_axes_interp(grad_desc, grad_along, t)
    distances = pairwise(Haversine(6378.388), grad_desc', grad_along', dims = 2)
    min_dist = argmin(distances)
    desc_seq = collect(0:1/(min_dist[1]-1):1)
    along_seq = collect(0:1/(min_dist[2]-1):1)

    if (min_dist[1] == 1)
        desc_interp_long = 0
        desc_interp_lat = 0
    else
        desc_interp_long = lin_interp_all(1 .- t, desc_seq, grad_desc[1:min_dist[1], 1])
        desc_interp_long = desc_interp_long .- desc_interp_long[1]

        desc_interp_lat = lin_interp_all(1 .- t, desc_seq, grad_desc[1:min_dist[1], 2])
        desc_interp_lat = desc_interp_lat .- desc_interp_lat[1]
    end

    if (min_dist[2] == 1)
        along_interp_long = grad_along[1, 1]
        along_interp_lat = grad_along[1, 2]
    else
        along_interp_long = lin_interp_all(t, along_seq, grad_along[1:min_dist[2], 1])
        along_interp_lat = lin_interp_all(t, along_seq, grad_along[1:min_dist[2], 2])
    end
    PV_interp_long = desc_interp_long .+ along_interp_long
    PV_interp_lat = desc_interp_lat .+ along_interp_lat
    return [[PV_interp_long PV_interp_lat], minimum(distances), min_dist]
end

# could be replaced with interpolations.jl
function lin_interp_all(xout, xin, yin)
    n = size(xout)[1]
    yout = zeros(n)
    for i = 1:n
        yout[i] = lin_interp(xout[i], xin, yin)
    end
    return yout
end

function lin_interp(xout, xin, yin)
    i = 0
    j = size(xin)[1] - 1
    while (i < j - 1)# /* x[i] <= v <= x[j] */
        ij = Integer(round((i + j + 2) / 2)) #; /* i+1 <= ij <= j-1 */
        if (xout < xin[ij])
            j = ij - 1
        else
            i = ij - 1
        end
    end
    yin[i+1] + (yin[j+1] - yin[i+1]) * ((xout - xin[i+1]) / (xin[j+1] - xin[i+1]))
end
end
