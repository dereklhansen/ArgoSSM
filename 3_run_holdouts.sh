#!/bin/bash
cd code/
## Set number of parallel processes here
export NPROCS=20

# Only needs to be run once
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -p 20 -L setup.jl -e "main(cfg_override_holdouts)"

## Run analysis in R
cd ..
Rscript code/src_R/plot_holdouts.R
