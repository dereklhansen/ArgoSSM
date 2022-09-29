struct AliasTable{Q<:Real}
    ap::Vector{Q}
    alias::Vector{Int}
    larges::Vector{Int}
    smalls::Vector{Int}
end

AliasTable(x, size) = AliasTable(fill(x, size), fill(0, size), fill(0, size), fill(0, size))

function make_alias_table!(w, t::AliasTable)
    n = length(w)
    probsum = sum(w)
    probmean = probsum / n


    let n_small = 0, n_large = 0
        for i = 1:n
            if (w[i] < probmean)
                t.smalls[n_small+1] = i
                n_small += 1
            else
                t.larges[n_large+1] = i
                n_large += 1
            end
        end

        while (n_small > 0 && n_large > 0)
            n_small -= 1
            n_large -= 1

            small_current = t.smalls[n_small+1]
            large_current = t.larges[n_large+1]

            t.ap[small_current] = w[small_current] / probmean
            t.alias[small_current] = large_current

            w[large_current] = w[small_current] + w[large_current] - probmean

            if (w[large_current] < probmean)
                t.smalls[n_small+1] = large_current
                n_small += 1
            else
                t.larges[n_large+1] = large_current
                n_large += 1
            end
        end

        while (n_large > 0)
            n_large -= 1
            t.ap[t.larges[n_large+1]] = 1.0
        end

        while (n_small > 0)
            n_small -= 1
            t.ap[t.smalls[n_small+1]] = 1.0
        end
    end

    return t
end

function sample_alias_table!(rng, t::AliasTable, out)
    n = length(t.ap)
    for i = 1:length(out)
        N = sample(rng, 1:n)
        if (rand(rng) <= t.ap[N])
            out[i] = N
        else
            out[i] = t.alias[N]
        end
    end
    return out
end

function residual_resample!(rng, src, weights, dest, t::AliasTable)
    n_dest = length(dest)
    n_src = length(src)
    mean_w = mean(weights) * (n_src) / (n_dest)
    let m = 1
        for n = 1:n_src
            n_fixed = floor(Int64, weights[n] / mean_w)
            if n_fixed > 0
                dest[m:(m+n_fixed-1)] .= n
                m = m + n_fixed
                weights[n] = weights[n] - n_fixed * (mean_w)
            end
        end
        # Resample the rest
        @assert all(weights .<= mean_w)
        make_alias_table!(weights, t)
        sample_alias_table!(rng, t, view(dest, m:n_dest))
    end
    return dest
end

function residual_resample!(
    rng,
    src::AbstractArray{T},
    weights::AbstractArray{Q},
    dest::AbstractArray{T},
) where {T<:Real,Q<:Real}
    n = length(src)
    residual_resample!(rng, src, weights, dest, AliasTable(one(Q), n))
end

function residual_resample(rng, weights::AbstractArray{Q}, N) where {Q<:Real}
    src = 1:length(weights)
    dest = Vector{Int}(undef, N)

    residual_resample!(rng, src, weights, dest)
end

## For now, mark these as having no gradient
## Versions could be added later which have a gradient defined via adjoint
Zygote.@nograd residual_resample, residual_resample!, make_alias_table!, sample_alias_table!
