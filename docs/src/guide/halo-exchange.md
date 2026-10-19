# Halo exchange

## Blocking updates

```julia
update_halo!(g, f)
update_halo!(g, temperature, pressure)
update_halo!(g, f; dims=(1,))
```

These snippets assume matching grids and fields have already been constructed.
Each side sends the interior strip adjacent to its halo and receives into the
outer halo strip. The other dimensions span the full local array, including
their halos. Processing dimensions sequentially propagates corner values.

The default dimension order is `(1, 2)` or `(1, 2, 3)`. Supply a nonempty tuple
of valid dimension indices to exchange a subset or change the order. Use matching
field order, compatible types and sizes, and the same dimension order on peers.
An exchange restricted to one dimension does not refresh the other halos.

## Overlaps wider than two halos

![Two ranks along one dimension. Halo width is 2 in both cases. With the default overlap of 4 the strip rank A sends is next to its halo. With overlap 6 the sent strip moves two cells inward. Either way it lands in the halo of rank B.](../assets/overlaps.png)

By default neighboring ranks share `2*halowidths[d]` lines. Pass `overlaps=`
to `field` to share more, for staggered or padded arrays:

```julia
g = grid(t, (9, 8))
f = field(zeros(9, 8); halowidths=(1, 1), overlaps=(3, 2))
update_halo!(g, f)
```

The halo stays `halowidths[d]` wide, but the strip sent to a neighbor is lines
`overlaps[d]-halowidths[d]+1:overlaps[d]` (low side) and
`n-overlaps[d]+1:n-overlaps[d]+halowidths[d]` (high side), where `n` is the
local size, instead of the lines next to the halo. `overlaps[d]` must satisfy
`2*halowidths[d] <= overlaps[d] <= size(A,d)`. Coordinate helpers and
`region=:faces` honor it; `save_hdf5`, `load_hdf5!` and `save_pvtk` reject it.

## Face halos for axial stencils

![Full exchange fills every halo cell including corners. Face exchange fills only the face slabs and leaves the corner halos untouched.](../assets/faces.png)

```julia
update_halo!(g, temperature; region=:faces)
handle = halo_exchange_start!(g, temperature; region=:faces)
try
    # Compute into separate output storage in the inner box. Every requested
    # face remains pending, so stencil reads must avoid all of those faces.
finally
    halo_exchange_finish!(handle)
end
```

The default `region=:full` includes transverse halos and propagates edges and
corners sequentially. Choose `region=:faces` when the stencil reads only axial
neighbors, such as a five-point or seven-point Laplacian. A face slab spans the
owned interior in every other grid dimension. Cells in halo edges/corners,
unrequested faces, and physical boundaries without neighbors remain unchanged.
Extra trailing field dimensions, including batched CellArray components, are
exchanged in full.

Face mode packs and posts every requested dimension before returning from
`start!`, and defers all unpacking until `finish!`. With halo width one and a
radius-one stencil, the inner box `3:nlocal[d]-2` in every grid dimension is
safe to compute while the exchange is pending. Compute the remaining owned
boundary slabs after finishing, and keep input fields unchanged throughout.
For larger stencil radii, shrink the inner box accordingly.

Peers must use matching regions, dimensions, field order, types, halo widths,
and compatible slab sizes. Face mode rejects repeated dimensions. Both modes
require `overlaps[d] + halowidths[d] <= size(data,d)` in each requested
dimension (`3*halowidths[d]` with default overlaps), so the sent strip holds
only owned cells. Dimensions with a single rank and no periodicity are
exempt. Diagonal stencils and mixed derivatives that read edges or corners
require `region=:full`. Both modes share the same
one-pending-exchange-per-grid guard. More pending dimensions permit broader
overlap, but do not guarantee that MPI progresses during computation.

The following example checks the face-overlapped stencil against full exchange:

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "..", "examples", "face_exchange.jl"), String) * "\n```")
```

Run it from the package project:

```sh
julia --project=. examples/face_exchange.jl
```

## Overlapping independent computation ("hiding" communication)

![Schematic timelines. Blocking: a kernel over the whole array, then update_halo!. Split phase: outer shell compute, start!, then the inner box overlapping the last-dimension communication, then finish!.](../assets/timeline.png)

```julia
handle = halo_exchange_start!(g, f)
try
    # Compute into a separate output array using only data that does not
    # depend on the pending halos. Keep f.data unchanged until finish!.
finally
    halo_exchange_finish!(handle)
end
# All exchanged halos can now be read.
```

The two-call API is backend-independent: `halo_exchange_start!` returns a
handle; you do compute in between; `halo_exchange_finish!` completes the
handle. `update_halo!` is just `start!` immediately followed by `finish!`, and
is what you want unless you specifically intend to overlap something. A
ParallelStencil convenience macro for the common outer/inner stencil split is
introduced below. See [Hiding communication and boundary widths](./hide-communication.md)
for diagrams, runnable examples, sizing rules, and tuning guidance.

### ParallelStencil convenience macro

Loading ParallelStencil activates the optional `@hide_halo_communication`
extension. The macro uses full exchange; use the split API above to request
face exchange:

```julia
@hide_halo_communication g Tf boundary_width=(2, 2, 2) begin
    @parallel (1:nx-2, 1:ny-2, 1:nz-2) update_temperature!(
        T, Cp, qx, qy, qz, dt, dx, dy, dz)
    @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
    @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
    @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
    @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
end
```

The macro expands this into the following schedule:

1. Partition the first launch's range into nonoverlapping outer slabs and one
   inner box.
2. Launch `update_temperature!` once for every outer slab.
3. Execute the remaining boundary-condition launches.
4. Synchronize ParallelStencil so halo packing sees completed outer values.
5. Start the HaloGrids exchange for `Tf`.
6. Launch `update_temperature!` over the inner box.
7. Finish the exchange in a `finally` clause.

Without `boundary_width=...`, the macro uses the componentwise maximum halo
width of its fields. This is the smallest naturally inferred safe width. An
explicit larger value is a tuning parameter: it increases the work completed
before communication and reduces the inner work available for overlap. It
does not alter `Field.halowidths`, MPI message sizes, or the computed result.
For `(16, 2, 2)`, overlap requires at least 33 points in the x range and at
least 5 in the y/z ranges. If any range length is at most twice its width,
the macro runs the full kernel, applies the boundary updates, synchronizes,
and performs a blocking `update_halo!`. This fallback applies at any rank count.
Ranges must start at `1`, matching ParallelStencil's partitioning contract.

The first block statement must be the explicitly ranged computation to split.
All later statements execute only once, before communication, and should be
restricted to boundary data. They must not read values that are produced only
by the still-unlaunched inner kernel. Applying physical boundary kernels on
every rank is valid when halo exchange follows: received neighbor data
overwrites provisional values at internal interfaces, while nonperiodic domain
walls retain the imposed values.

### Why only the last dimension overlaps in full exchange

![Three panels of a 2D array. Before exchange all halo cells are stale. After dimension 1 the left and right halos are filled. After dimension 2 the top and bottom halos are filled, including the four corners, which arrive via two hops.](../assets/corners.png)

Full halo exchange is sequential across dimensions: dimension 2's
packing has to read what dimension 1's unpacking just wrote, so that a corner
cell ends up holding the correct diagonal neighbor's value. `start!` therefore
performs every dimension in `dims` **except the last** fully synchronously —
pack, send/recv, wait, unpack — before it ever returns control to you. Only
the *last* dimension in `dims` is actually left outstanding; that is the one
piece of work `finish!` has left to do, and the only part your inserted
compute overlaps with. This is the scope of the default `region=:full`
algorithm. The optional `region=:faces` excludes edges and corners and can
therefore leave every requested dimension pending.

Concretely, right after `start!(g, f; dims=(1, 2))` returns:

- `f`'s dimension-1 halo (the first/last slab along dimension 1) is already
  correct — that exchange ran to completion before `start!` returned.
- `f`'s dimension-2 halo (the first/last slab along dimension 2, since `2` is
  the last entry in `dims`) is **not yet correct** — the receive is still in
  flight, and the buffer it will unpack into is only safe to touch from
  `finish!`.

So the data that is safe to read while a handle is pending is: every other
field's data unconditionally, plus `f.data` itself *except* the still-in-flight
last-dimension halo slab. The data that is unsafe is `f.data`'s last-dimension
halo slab (being written by the pending receive) and, by extension, anything
you compute that reads it.

### Worked example: overlapping a stencil update with its own halo exchange

The most common real use of this API is not "do something unrelated while
waiting" but "update the part of my own field that doesn't need the halo yet,
while the part that does is still in flight." For a 5-point stencil, a point
at column `j` only reads columns `j-1` and `j+1`, so every column whose
neighbors avoid the still-pending last-dimension halo columns can be updated
immediately; only the one column on each side immediately adjacent to that
halo has to wait for `finish!`:

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "..", "examples", "hide_communication.jl"), String) * "\n```")
```

The example computes the same diffusion step twice — once the straightforward
way (`update_halo!` then the whole interior in one pass) and once with the
overlap split — and asserts the two results are identical, to make the point
that splitting the compute changes *when* work happens, not *what* it
computes. Run it the same way as [the quickstart](./getting-started.md):

```sh
julia --project=. examples/hide_communication.jl
```

### Progress, buffers, and the one-in-flight-exchange rule

Whether communication actually makes progress *during* your inserted compute —
as opposed to only once you call into MPI again — depends on the MPI
implementation and its progress engine (some implementations only advance
non-blocking requests inside an MPI call, so the overlap is a correctness/
scheduling convenience even where it isn't a wall-clock win). On a single-rank
periodic dimension there is no MPI call at all: the local copy is prepared
during `start!` and its unpack is simply deferred until `finish!`.

Only one exchange may be pending per `Grid` at a time: starting a second one
before finishing the first raises a clear error rather than corrupting
buffers, since the last dimension's send/recv buffers stay live in the grid's
cache until `finish!` unpacks them. Wrap `finish!` in a `try`/`finally` (as
above) so it still runs if your inserted compute throws — an unfinished
handle otherwise leaves the grid permanently "pending" and every later
exchange on it will error. Treat the handle itself as opaque. Different
`Grid`s remain fully independent — see
[multiple grids](./concepts.md#multiple-grids) for communicator isolation.

## Boundaries

Periodic single-rank dimensions copy the opposite interior strip locally.
Nonperiodic sides without neighbors are not received into for that dimension.
The solver remains responsible for physical boundary values and for their
consistency at edges and corners when other dimensions are exchanged.

A periodic dimension split across exactly **2** ranks is a real, common
topology (an 8-rank 2D grid with a balanced `(4, 2)` split, or simply
`mpiexec -n 2` with one periodic axis) — and, for one commit, was a
genuinely miscorrected case: the two "sides" of such a dimension are the
same physical neighbor rank, and messages were tagged only by which slot
posted them, not by which physical direction (low vs. high) they carried.
MPI's message-ordering guarantee then matched them by post order instead of
direction, so a rank's low halo silently received its neighbor's low-side
interior strip instead of the intended high-side one. Every message is now
tagged by direction rather than by slot or by which rank happens to be on
which side, which is unambiguous even when both neighbors coincide; see the
regression test in `test/test_halo_exchange.jl` (the one whose name starts
with "periodic dimension split across exactly 2 ranks") for a minimal
reproduction. This is the kind of bug a rank-uniform test fixture cannot
see — every existing correctness test filled each rank's whole array with
one repeated value, which happens to make the swapped and correct halos
identical — which is why the [coordinate helpers](./coordinates.md) worked
example (position-dependent data, checked against an exact conservation
law) is a better correctness oracle for this class of issue than "assert
against the rank number."

## Packing and memory

Communication buffers belong to the grid and grow for the fields and slab sizes
encountered. Reuse the grid and warm up representative exchanges before measuring
steady-state allocations or timing. Cache growth and compilation affect first calls.

```julia
g = grid(t, (258, 258, 258); threaded_pack=true) # t must be 3D
```

Threaded packing is disabled by default. Opting in uses spawned tasks for large
slabs when Julia has multiple threads (for example, `julia -t 4 --project=.`).
The current threshold is 65,536 elements per slab. Tasks allocate, and MPI calls
still run on the calling thread. Benchmark your workload before enabling this.
