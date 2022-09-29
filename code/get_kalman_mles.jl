using BSON
using Printf
using Statistics
using Distributions
mles = map(1:210) do h
    x = BSON.load(@sprintf("../output/holdouts_scratch/holdouts_kalman_%03d.bson", h))
    x[:res].res.params
end

mle_mat = hcat(mles...)

dists = map(1:5) do i
    d = fit_mle(Gamma, mle_mat[i, :])
    println(d)
    return d
end

names = ["σ_x_long", "σ_x_lat", "σ_v_long", "σ_v_lat", "σ_p"]

for (i, dist) in enumerate(dists)
    println(names[i], ":", dist)
end
