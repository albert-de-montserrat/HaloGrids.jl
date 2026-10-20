# HDF5 I/O for the MPI/HDF5 extension.
#
# No parallel-HDF5 build is assumed (`HDF5.has_parallel()` is false for the
# ordinary HDF5_jll-backed install this package's weakdep resolves to), so
# there is no meaningful true-collective (simultaneous multi-rank) file
# access available to build on. Instead, every rank that calls save_hdf5/
# load_hdf5! together (this is still a collective operation in the sense
# that every rank in the grid's topology must call it, or the internal
# MPI.Barrier calls below deadlock) takes a turn, in rank order, opening the
# same file, writing only its own hyperslab of each field's shared global
# dataset, and closing it before the next rank's turn — so at most one rank
# ever has the file open at a time. This keeps a single shared file with a
# genuine `nglobal`-shaped dataset per field (not one file per rank), at the
# cost of a strictly serialized (not simultaneous) write schedule. Reads
# don't need this: every rank opens the file independently in "r" mode and
# reads concurrently, since concurrent read-only access to the same file from
# multiple processes is ordinary, safe filesystem/HDF5 behavior even without
# a parallel-HDF5 build — only concurrent *writes* need serializing.
#
# Halo cells that duplicate a neighbor's data are never part of the payload
# (see local_interior_range in shared.jl) — load_hdf5! leaves them exactly as
# they were in the caller's array; call update_halo! afterward to repopulate
# them from the freshly loaded neighbor data.

using HaloGrids
using HDF5
using MPI

function resolve_group(h5, group::AbstractString, create::Bool)
    group == HDF5_IO_DEFAULT_GROUP && return h5
    if haskey(h5, group)
        return h5[group]
    elseif create
        return create_group(h5, group)
    else
        throw(HDF5IOError("group \"$group\" does not exist in $(HDF5.filename(h5))."))
    end
end

function check_field_batch(grid::HaloGrids.Grid{N}, fields::NTuple{M,HaloGrids.Field}, names::NTuple{M,AbstractString}) where {N,M}
    M >= 1 || throw(HDF5IOError("at least one field is required."))
    allunique(names) || throw(HDF5IOError("field names must be unique within a file; got $(names)."))
    for (i, f) in enumerate(fields)
        ndims(f) == N || throw(HDF5IOError(
            "field at position $i has $(ndims(f)) dimensions, but the grid has $N; " *
            "save_hdf5/load_hdf5! require every field to match the grid's dimensionality exactly."))
        require_cpu_array(f, i)
    end
    HaloGrids.check_fields(grid, fields)
    return nothing
end

function write_field_dataset!(parent, group::AbstractString, grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N},
                               meta::HDF5Metadata, create::Bool) where {T,N}
    if create
        haskey(parent, meta.field_name) && throw(HDF5IOError(
            "a dataset named \"$(meta.field_name)\" already exists in group \"$group\"; " *
            "use a distinct field name, a distinct group, or start from a fresh file (mode=\"w\")."))
        dset = create_dataset(parent, meta.field_name, datatype(T), dataspace(meta.nglobal))
        write_metadata_attrs!(dset, meta)
    end
    dset = parent[meta.field_name]
    local_ranges, global_ranges = local_and_global_ranges(grid, field)
    dset[global_ranges...] = view(field.data, local_ranges...)
    return nothing
end

function read_field_dataset!(parent, group::AbstractString, grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N},
                              field_name::AbstractString; repartition::Bool = false) where {T,N}
    haskey(parent, field_name) || throw(HDF5IOError(
        "no dataset named \"$field_name\" in group \"$group\"."))
    dset = parent[field_name]
    meta = metadata_from_dataset(dset)
    validate_metadata!(meta, grid, field; repartition = repartition)
    size(dset) == meta.nglobal || error(
        "Dataset shape mismatch for field \"$field_name\": metadata has $(meta.nglobal), dataset has $(size(dset)).")
    local_ranges, global_ranges = local_and_global_ranges(grid, field)
    field.data[local_ranges...] = dset[global_ranges...]
    return nothing
end

function HaloGrids.save_hdf5(grid::HaloGrids.Grid{N}, fields::NTuple{M,HaloGrids.Field}, path::AbstractString;
                               names::NTuple{M,AbstractString} = default_field_names(M),
                               group::AbstractString = HDF5_IO_DEFAULT_GROUP,
                               mode::AbstractString = "w") where {N,M}
    check_field_batch(grid, fields, names)
    metas = ntuple(i -> metadata_from_grid(grid, fields[i], names[i]), M)

    comm = grid.topology.comm
    rank = grid.topology.rank
    nprocs = MPI.Comm_size(comm)

    # Each rank's turn is wrapped in try/catch and its outcome shared via
    # MPI.Allreduce (itself already collective/blocking, so it doubles as the
    # inter-turn synchronization a plain MPI.Barrier would otherwise provide)
    # rather than a bare MPI.Barrier: an uncaught error on rank r's turn would
    # otherwise make rank r skip straight past that barrier while every other
    # rank blocks on it forever, waiting for a rank that has already errored
    # out. Every rank raises together instead.
    # A fresh file ("w") is written under a temporary name and renamed only after
    # every rank has written, so an interrupted save never truncates the
    # previous checkpoint or leaves a complete-looking file full of fill values.
    target = mode == "w" ? path * ".tmp" : path
    for r in 0:(nprocs - 1)
        local_err = nothing
        if rank == r
            try
                h5open(target, r == 0 ? mode : "r+") do h5
                    parent = resolve_group(h5, group, r == 0)
                    for i in 1:M
                        write_field_dataset!(parent, group, grid, fields[i], metas[i], r == 0)
                    end
                end
            catch e
                local_err = e
            end
        end
        nfail = MPI.Allreduce(local_err === nothing ? 0 : 1, +, comm)
        if nfail > 0
            local_err === nothing || rethrow(local_err)
            error("save_hdf5 aborted: rank $r failed while writing its turn " *
                  "(see that rank's error output for the underlying cause).")
        end
    end
    if target != path
        rename_err = nothing
        if rank == 0
            try
                mv(target, path; force = true)
            catch e
                rename_err = e
            end
        end
        # Also keeps other ranks from returning (and maybe reading `path`) before the rename.
        MPI.Allreduce(rename_err === nothing ? 0 : 1, +, comm) == 0 ||
            (rename_err === nothing ? error("save_hdf5: rank 0 failed to rename $target to $path.") : throw(rename_err))
    end
    return path
end

function HaloGrids.save_hdf5(grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N}, path::AbstractString;
                               name::AbstractString = "field", kwargs...) where {T,N}
    HaloGrids.save_hdf5(grid, (field,), path; names = (name,), kwargs...)
    return path
end

function HaloGrids.load_hdf5!(grid::HaloGrids.Grid{N}, fields::NTuple{M,HaloGrids.Field}, path::AbstractString;
                                names::NTuple{M,AbstractString} = default_field_names(M),
                                group::AbstractString = HDF5_IO_DEFAULT_GROUP,
                                repartition::Bool = false) where {N,M}
    check_field_batch(grid, fields, names)

    h5open(path, "r") do h5
        parent = resolve_group(h5, group, false)
        for i in 1:M
            read_field_dataset!(parent, group, grid, fields[i], names[i]; repartition = repartition)
        end
    end
    return fields
end

function HaloGrids.load_hdf5!(grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N}, path::AbstractString;
                                name::AbstractString = "field", kwargs...) where {T,N}
    HaloGrids.load_hdf5!(grid, (field,), path; names = (name,), kwargs...)
    return field
end
