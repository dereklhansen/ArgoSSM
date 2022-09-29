#!/bin/bash
export NPROCS=20

#Only need to run if not running the under-ice holdouts
Rscript code/src_R/floats_to_h5.R
cd code

julia --project=. -p 20 -L setup.jl -e "main(cfg_override_all)"

# Plot floats
Rscript code/src_R/plot_floats.R

# Downstream tasks
Rscript code/cov_est.R
Rscript code/cov_est_vel.R
