export save_pvtk

const PVTKEXT_INSTALL_MSG = "requires the optional WriteVTK extension: run `using WriteVTK` " *
    "(after `]add WriteVTK`) to enable HaloGridsWriteVTKExt."

"""
    save_pvtk(grid::Grid, field::Field, path; name="field", dx=ntuple(_->1.0,N))
    save_pvtk(grid::Grid, fields::NTuple{M,Field}, path; names, dx=ntuple(_->1.0,N))

Write one or more [`Field`](@ref)s to ParaView's parallel VTK XML image-data
format (`.pvti` header + one `.vti` piece per rank), using
[WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl)'s MPI-aware
`pvtk_grid`. This is a **write-only** extension: ParaView (or `ReadVTK.jl`,
used only in this repository's own tests) is the reader, so there is no
`load_pvtk!` counterpart to `save_hdf5`/`load_hdf5!`.

Unlike [`save_hdf5`](@ref), which trims halo cells that duplicate a
neighbor's data, `save_pvtk` writes each rank's **entire local array,
halo included** — parallel VTK *requires* neighboring pieces to overlap at
shared boundaries so ParaView can stitch them into one seamless volume.
Since `Grid` stores no physical spacing, `dx` must be supplied explicitly
(same convention as [`coord_g`](@ref)/[`x_g`](@ref)/[`y_g`](@ref)/[`z_g`](@ref)),
defaulting to unit spacing in every dimension.

`path` is a **base name with no extension**, not a literal file path: a call
with `path="fields"` produces `fields.pvti` plus a `fields/` subdirectory
holding one `fields_<rank+1>.vti` file per rank — this is WriteVTK's own
`pvtk_grid` convention, not `save_hdf5`'s (whose `path` is the literal file
to write).

Every field passed to one `save_pvtk` call must share the same
`halowidths`: mixed halo widths would imply a different global domain size
per field (see [`nglobal`](@ref)/[`iglobal`](@ref)), which cannot be
represented by the single shared coordinate grid one `pvtk_grid` file set
uses — save fields with different halowidths in separate calls instead.

**Periodic topologies are not supported.** Parallel VTK extents are a flat,
non-wrapping index space, so a periodic dimension's wrap-around halo cannot
be written as a contiguous extension of the same coordinate axis without
producing spatially folded (incorrect) geometry; `save_pvtk` raises a clear
error naming the offending dimension(s) rather than silently writing wrong
data.

This is a stub method: calling it without `WriteVTK` loaded raises this
error rather than a bare `MethodError`. Loading `WriteVTK` (`using WriteVTK`)
resolves the call to `HaloGridsWriteVTKExt`'s implementation via ordinary
multiple dispatch — the same "declare a stub in core, add methods from the
extension" pattern as [`save_hdf5`](@ref) and `pack_target` in
`backend.jl` — so no `Base.get_extension` lookup is needed at the call site.

See the "Parallel ParaView (pvtk) I/O" guide for the on-disk layout, the
halo-inclusive overlap convention, and worked single-/multi-field examples.
"""
save_pvtk(::Grid, ::Union{Field,Tuple}, ::AbstractString; kwargs...) =
    error("save_pvtk $PVTKEXT_INSTALL_MSG")
