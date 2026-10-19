# Thin glue module mirroring the repository's extension pattern:
# the real optional I/O implementation lives in src/HDF5Ext/ rather than
# inside the extension file itself. save_hdf5/load_hdf5! are declared as
# stub functions in src/HDF5IO.jl and given concrete methods here (via
# `HaloGrids.save_hdf5(...) = ...`) — see that file for why, and
# backend.jl's pack_target for the same pattern used elsewhere.
module HaloGridsHDF5Ext
    include(joinpath(@__DIR__, "..", "src", "HDF5Ext", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "HDF5Ext", "metadata.jl"))
    include(joinpath(@__DIR__, "..", "src", "HDF5Ext", "io.jl"))
end
