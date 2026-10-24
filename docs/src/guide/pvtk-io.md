# Parallel ParaView (pvtk) I/O

`save_pvtk` writes `Field` data to ParaView's parallel VTK XML image-data
format (`.pvti` header + one `.vti` piece per rank), using the `Grid`'s
Cartesian topology to place each rank's piece at its correct position. This
is an optional extension: it activates only once `WriteVTK` is loaded
alongside `HaloGrids`. It is **write-only** — ParaView (or `ReadVTK.jl`,
used only in this repository's own tests) is the reader, so there is no
`load_pvtk!` counterpart to `save_hdf5`/`load_hdf5!`.

```julia
using HaloGrids
using WriteVTK   # loads HaloGridsWriteVTKExt

t = topology((0, 0))
g = grid(t, (66, 66))
f = field(rand(66, 66); halowidths=(1, 1))

save_pvtk(g, f, "state"; name="temperature", dx=(0.1, 0.1))
```

This writes `state.pvti` plus a `state/` subdirectory holding one
`state_<rank+1>.vti` file per rank. Open `state.pvti` in ParaView to see the
whole domain as one seamless volume, stitched together from every rank's
piece.

Calling `save_pvtk` without `using WriteVTK` raises a clear error naming the
missing dependency, rather than a bare `MethodError` — see [`save_pvtk`](@ref)
in the API reference.

## `path` is a base name, not a literal file path

Unlike `save_hdf5`'s `path` (a literal file to write), `save_pvtk`'s `path`
is a base name that WriteVTK expands into a `.pvti` header file plus a
same-named subdirectory of per-rank `.vti` pieces — this is WriteVTK's own
`pvtk_grid` convention. Passing `"state.vti"` produces `state.vti.pvti` and a
`state.vti/` directory, not a single `state.vti` file — pass the bare name
(`"state"`) instead.

## Halo cells *are* stored — the opposite of `save_hdf5`

[`save_hdf5`](./hdf5-io.md#halo-cells-are-not-stored) deliberately strips
halo cells that duplicate a neighbor's data, so its saved dataset has no
redundant storage. `save_pvtk` does the opposite: it writes each rank's
**entire local array, halo included**. This is a real requirement of the
parallel VTK format, not a stylistic choice — per WriteVTK's docs, "the
extents for different processes overlap ... This is a requirement of VTK,
and without it the full data cannot be visualised in ParaView."

`HaloGrids` already keeps halo cells exactly synchronized with each
neighbor's real data via [`update_halo!`](./halo-exchange.md), so as long as
you call it before `save_pvtk` (or your fill routine already produces
globally-consistent halo values, e.g. because every cell's value is a pure
function of its global position), the overlap between neighboring ranks'
pieces is correct by construction, with no extra bookkeeping. Neighboring
ranks' pieces overlap by exactly `2*halowidths[d]` grid points in every
dimension.

## `dx` is required

`Grid` stores no physical spacing, so `save_pvtk` needs it supplied
explicitly — the same convention [`coord_g`](@ref)/[`x_g`](@ref)/
[`y_g`](@ref)/[`z_g`](@ref) use. It defaults to `1.0` in every dimension
(unlabeled index space) when omitted.

## Multiple fields in one file set

```julia
u = field(zeros(66, 66); halowidths=(1, 1))
v = field(zeros(66, 66); halowidths=(1, 1))
save_pvtk(g, (u, v), "state"; names=("u", "v"), dx=(0.1, 0.1))
```

All fields in one call are attached to the *same* `pvtk_grid` file set (one
`.pvti` + one `.vti`-per-rank set holding every named field as a separate
point-data array) — closer to how a real solver writes one multi-field
ParaView snapshot per timestep than `save_hdf5`'s one-dataset-per-field
model. `names` defaults to `("field1", "field2", ...)` by position when
omitted, same as `save_hdf5`. Field names must be unique within one call.

**Every field in one call must share the same `halowidths`.** Two fields
with different halowidths on the same `Grid` correspond to different global
domain sizes (see [`nglobal`](@ref)), so they cannot both be placed
correctly on the single shared coordinate grid one `pvtk_grid` file set
uses. `save_pvtk` rejects a mismatched batch before writing anything; save
fields with different halowidths in separate calls instead.

## Periodic domains are not supported

`save_pvtk` raises a clear error, naming the offending dimension(s), if the
grid's topology is periodic in any dimension. Parallel VTK extents are a
flat, non-wrapping index space: a periodic dimension's wrap-around halo
holds the *opposite* end's data after `update_halo!`, and writing that as a
contiguous extension of the same coordinate axis would silently produce
spatially folded (incorrect) geometry rather than a real seam. There is no
workaround in this extension yet — periodic domains simply cannot use
`save_pvtk`.

## Why every rank's write is independent

Unlike [`save_hdf5`](./hdf5-io.md#why-every-rank-still-calls-these-together),
which has no parallel-HDF5 build to make a genuinely simultaneous multi-rank
write meaningful and so serializes ranks into a turn-taking schedule,
parallel VTK sidesteps that problem entirely: every rank writes its own,
completely independent serial `.vti` file, and only one rank (`ismain`,
which `save_pvtk` sets from the rank's own Cartesian rank) additionally
writes the small `.pvti` header. There is no shared file, no file-locking
contention, and nothing to serialize — every rank's write is a genuinely
independent filesystem operation. `save_pvtk` is still collective in the
sense that every rank in the grid's topology must call it together (an
internal `MPI.Allgather` blocks otherwise), same as `save_hdf5`.

## CPU arrays only

Like `save_hdf5`, `save_pvtk` requires every field's array to be a
`StridedArray` (`Array`, or a contiguous `SubArray`/`ReshapedArray`/
`PermutedDimsArray` of one) — i.e. ordinary, host-addressable memory. A
`CuArray`/`ROCArray` field must be staged to the host first:

```julia
save_pvtk(g, field(Array(gpu_field.data); halowidths=gpu_field.halowidths), "state")
```

## Runnable example

The runnable [`gaussian3d_pvtk.jl`](https://github.com/albert-de-montserrat/HaloGrids.jl/blob/master/examples/gaussian3d_pvtk.jl)
example writes a 3D Gaussian bump, distributed across however many ranks the
launcher gives it, as a parallel ParaView file set — a good way to
hand-verify that neighboring ranks' pieces really do stitch into one
seamless volume, with no gaps or double-counted cells at rank boundaries.
From an environment with HaloGrids, MPI and WriteVTK (for example `]dev` this
repository, then `]add MPI WriteVTK`) and an MPI launcher matching MPI.jl, run:

```sh
mpiexec -n 8 julia --project=<env> examples/gaussian3d_pvtk.jl
```

then open the resulting `gaussian3d.pvti` in ParaView.
