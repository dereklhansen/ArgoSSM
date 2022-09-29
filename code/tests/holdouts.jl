using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere dirinclude(x) = include(string(@__DIR__) * "/../" * x)
@everywhere dirinclude("src/Holdout.jl")
dirinclude("src/PV_source.jl")

import Main.Holdout.main
import Main.Holdout.process_holdout

hstart = 1
hend = 2
parallel = true
if parallel
    map = Distributed.pmap
else
    map = Base.map
end

for model in ["pvinterp1"]
    parsed_args = Dict(
        "model" => model,
        "holdout_file" => "../output/holdouts/holdouts_06_2021.h5",
        "output_prefix" => missing,
        "h5file" => missing,
        "run_holdout" => true,
        "process_holdout" => true,
    )

    outputs = map(hstart:hend) do h
        Main.Holdout.main(parsed_args, h)
    end

    Base.map(hstart:hend, outputs) do h, output_bson
        Main.Holdout.process_holdout(parsed_args, h, output_bson)
    end
end
