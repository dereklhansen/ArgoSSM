# ArgoSSM: A probabilistic model of ocean floats under ice 
This repo implements the experiments in
> Hansen, D. and Yarger, D. (2022). A probabilistic model of ocean floats under ice.

## Datasets

## Installation
Make sure the following software is available:
- R
- Julia 1.7.3

The dependencies for the Julia part of the codebase are located in `code/Project.toml` and `code/Manifest.toml`.

For plots and tables, additional R packages will need to be installed:

- `tidyverse`
- `ggplot2`
- `ncdf4`
- `rhdf5` (Bioconductor)
- `rgdal`
- `doParallel`
- `latex2exp`
- `mapproj`

## Running
The four bash scripts starting with numbers in the main folder recreate steps of downloading the data, running the float trajectory model in Julia, and then performing downstream analysis in R.
