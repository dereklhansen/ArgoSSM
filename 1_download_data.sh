#!/bin/bash

Rscript code/load_trajectory_data.R
Rscript code/load_profile_data.R
Rscript code/load_trajectory_pressure_data.R

# http://sose.ucsd.edu/BSOSE6_iter122_solution.html
wget -O temp/data/grid.nc http://sose.ucsd.edu/SO6/SETUP/grid.nc

# Ice data
pushd temp/data/ice_data_tiffs
./download_tiffs.sh
popd
Rscript code/src_R/ice_data.R
Rscript code/src_R/ice_data_to_h5.R
