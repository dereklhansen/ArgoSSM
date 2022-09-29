#!/bin/bash
cd code/
## Set up julia for first time
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. precalculate_pv_grid.jl
