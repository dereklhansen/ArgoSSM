using Random
rng = MersenneTwister(92)
n = 100
lw = randn(rng, n)
lw_max = maximum(lw)
W = exp.(lw .- lw_max)

s = @inferred SMC.residual_resample(rng, W, div(n, 2))
@test size(s) == (div(n, 2),)
s = @inferred SMC.residual_resample(rng, W, n)
@test size(s) == (n,)

src = 1:n;
dest = Vector{Int}(undef, n);
s2 = @inferred SMC.residual_resample!(rng, src, W, dest, SMC.AliasTable(one(Float64), n))
@test size(s2) == (n,)

## Alias table functions
atable = SMC.AliasTable(one(Float64), n)
dest = Vector{Int}(undef, n);
@inferred SMC.make_alias_table!(W, atable);
s3 = @inferred SMC.sample_alias_table!(rng, atable, dest)
@test size(s3) == (n,)

## Test that correct dist is targeted
src = randn(rng, n)
m1 = sum(src .* W) / sum(W)
m2 = sum((src .^ 2) .* W) / sum(W)

s_large = @inferred SMC.residual_resample(rng, W, 10_000_000);
@test size(s_large) == (10_000_000,)
m1_s = sum(src[s_large]) / 10_000_000
m2_s = sum(src[s_large] .^ 2) / 10_000_000

@test isapprox(m1, m1_s, atol = 0.0001)
@test isapprox(m2, m2_s, atol = 0.0001)
