using PackageCompiler
create_sysimage(
    [
        :ArgParse,
        :DrWatson,
        :Flux,
        :Zygote,
        :HDF5,
        :SparseArrays,
        :NetCDF,
        :DataFrames,
        :Distances,
        :DelimitedFiles,
        :DataFrames,
        :Distributions,
        :DistributionsAD,
        :Plots,
    ],
    sysimage_path = "sysimg.so",
)
