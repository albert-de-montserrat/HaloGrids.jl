# Hiding communication and boundary widths

`@hide_halo_communication` schedules the outer part of a ParallelStencil
kernel before halo exchange and the inner part while the final dimension's
exchange is pending. `boundary_width=(bx, by, bz)` specifies how many
**kernel points on each side of each dimension** belong to that outer part.
The widths describe a computation split; the exchanged halo thickness still
comes from `Field.halowidths`.

The split applies to device backends such as CUDA and AMDGPU. With a CPU
backend (`Threads` or `Polyester`), the macro always takes the blocking
schedule; see [CPU backends run blocking](#cpu-backends-run-blocking).

Loading `ParallelStencil` activates this optional macro. The underlying
`halo_exchange_start!` / `halo_exchange_finish!` API also works without
ParallelStencil; see [Halo exchange](./halo-exchange.md).

## What the numbers mean

For a three-dimensional kernel:

| Entry | Meaning |
| :--- | :--- |
| `bx` | Number of kernel points at **each** of the low-x and high-x sides computed before exchange |
| `by` | Number of kernel points at **each** of the low-y and high-y sides computed before exchange |
| `bz` | Number of kernel points at **each** of the low-z and high-z sides computed before exchange |

Use `(bx, by)` for a two-dimensional kernel. A width of `2` means two
points at each side, so it removes four points from that dimension of the
inner box. The units are point counts, not bytes, ranks, threads, or physical
distances.

For one dimension of a launch range `1:12`, a width of `3` gives:

```text
Kernel index:    1  2  3 | 4  5  6  7  8  9 | 10 11 12
                 outer |       inner       |  outer
Point count:       3   |         6         |    3
Computed:       before | exchange pending  | before
```

In multiple dimensions, the outer region is the shell around the inner box.
A point belongs to the shell if it lies within the selected width of **any**
side. ParallelStencil partitions this shell into disjoint slabs, so corners
and edges are computed once, rather than once per face.

![Two-dimensional computation splits with the same one-point halos: widths (1, 1) leave a 10 by 8 inner box, while widths (3, 2) leave a 6 by 6 inner box.](../assets/boundary-width-regions.png)

The sketch uses a `14 × 12` local array with a one-point halo. The kernel
updates its `12 × 10` owned region. Gray cells are the array's halos; orange
cells are computed before exchange; blue cells are computed while the final
exchange is pending. Increasing the computation widths leaves the gray halo
unchanged. The same construction in 3D produces an outer shell and an inner
rectangular box.

## Halo width, launch range, and array indices

These three quantities have different roles:

| Quantity | Controls | Example |
| :--- | :--- | :--- |
| `Grid.nlocal` | Full local array shape, including halos | `(32, 32, 32)` |
| `Field.halowidths` | Thickness of halo cells and exchanged interior strips | `(1, 1, 1)` |
| First `@parallel` call's ranges | Kernel index space that the macro partitions | `(1:30, 1:30, 1:30)` |
| `boundary_width` | Outer computation thickness within those ranges | `(2, 2, 2)` |

The macro does not infer the launch ranges from the grid, and it does not
inspect which array elements the kernel accesses. In the diffusion benchmark,
kernel index `(i, j, k)` writes array element `(i+1, j+1, k+1)`. For `N=32`:

```text
Array x index:    1 | 2  3 | 4 ... 29 | 30 31 | 32
                 H | outer|  inner   | outer | H
Kernel x index:    | 1  2 | 3 ... 28 | 29 30 |
```

Here `H` marks the one-point halo and `boundary_width[1]=2`. The kernel
range is `1:30`, and its inner range is `3:28`. The corresponding inner array
indices are `4:29`. The exchanged interior strip is still just one array cell
thick on each side.

For a range length `L[d]` and width `b[d]`, the inner range is:

```julia
(b[d] + 1):(L[d] - b[d])
```

Its length is `L[d] - 2*b[d]`. Every launch range must start at `1`; apply
any offset inside the kernel, as in the example above.

## Execution order

The first statement in the block must be one explicitly ranged `@parallel`
call. Subsequent statements run once as boundary updates:

```julia
# T, Cp, qx/qy/qz, g, Tf, and the kernels are already defined.
# This is the temperature-update portion of the diffusion benchmark.
@hide_halo_communication g Tf boundary_width=(2, 2, 2) begin
    @parallel (1:nx-2, 1:ny-2, 1:nz-2) update_temperature!(
        T, Cp, qx, qy, qz, dt, dx, dy, dz)
    @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
    @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
    @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
    @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
end
```

When an inner box exists, the schedule is:

```text
Compute outer slabs
        |
Apply boundary updates; synchronize ParallelStencil
        |
halo_exchange_start!
  exchange x completely
  exchange y completely
  pack and start z exchange
        |
Compute inner box  <---- z exchange is pending
        |
halo_exchange_finish!: wait and unpack z
```

This sketch assumes a 3D topology and the macro's default dimension order
`(1, 2, 3)`. In 2D, x completes inside `start!`, and y is the pending
dimension. Earlier dimensions complete before later ones are packed so that
corner data propagates correctly. The macro does not overlap all dimensions
at once.

The synchronization before packing ensures that the outer values are ready,
including on a device backend. `finish!` runs in a `finally` clause, so an
exception in the inner launch still completes the exchange. Only one exchange
may be unfinished per grid.

## Conditions for a correct split

The outer computation must produce every updated value that halo packing
will send. The inner computation must be independent of the pending receive
halos, and must not change values that belong to the communication boundary.
Choose widths that cover the packed interior strips **under your kernel's
index mapping**. For the simple shifted owned-region mapping above, widths
at least `overlaps - halowidths` (the halo width, for default overlaps) cover
those strips.

When `boundary_width` is omitted, the macro uses the componentwise maximum
of the exchanged fields' `overlaps - halowidths`. With default overlaps this
is the halo width: fields with widths `(1, 2)` and `(2, 1)` infer `(2, 2)`. A
field with `halowidths=(1, 1)` and `overlaps=(4, 4)` sends cells 3 deep and
infers `(3, 3)`. An explicit width must contain positive integers
and match the dimensionality of the launch ranges. The macro checks these
shape requirements, but it cannot verify kernel dependencies or whether an
explicit width covers your send strips.

A useful pattern is to compute a new field from a separate input field whose
halos are already valid. Another is the benchmark's update from precomputed
flux arrays. Each output point can then be computed independently. Splitting
a stencil that overwrites the same neighbor values it reads can change its
result; a width choice alone does not make such a kernel safe.

Boundary statements must update only boundary data and must not read values
that the inner kernel has yet to produce. They run after the outer kernels
and before packing. Physical boundary kernels may run on every rank when
subsequent halo exchange replaces provisional values at internal interfaces;
nonperiodic walls retain the solver's imposed values. Boundary dependencies
still need to be checked, including edges and corners.

## Small ranges use a blocking fallback

Overlap requires a nonempty inner box in every dimension:

```julia
L[d] > 2 * b[d]  # for every dimension d
```

If any range length is **equal to or smaller than** twice its width, the
macro takes this schedule instead:

```text
Compute the full kernel once
        |
Apply boundary updates; synchronize ParallelStencil
        |
update_halo!: complete every exchange dimension
```

The fallback preserves the halo exchange and the boundary updates. It applies
at any rank count and does not silently reduce the requested width. Invalid
widths or ranges that do not start at `1` still raise an error.

The fallback is selected by local launch size, not communicator size. On one
rank, a sufficiently large range on a device backend still uses the split
schedule. There is no
inter-rank network transfer: periodic dimensions use local copies, and
nonperiodic sides without neighbors receive nothing. Splitting may therefore
add overhead without a communication benefit on one rank.

## CPU backends run blocking

When the calling module initialized ParallelStencil with `Threads` or
`Polyester`, the macro uses the blocking schedule above for every launch
size. ParallelStencil's own `@hide_communication` makes the same choice on CPU.
Widths and ranges are still validated, and the result is the same.

CPU kernel launches are synchronous, so the inner launch cannot start until
the outer slabs finish, and MPI may not progress during computation. The
`Threads` backend distributes each launch over its last dimension, so a z
slab of width `bz` occupies at most `bz` threads, and narrow x slabs access
memory with a large stride. In the CPU diffusion benchmark (local `128³`,
4 threads per rank, 1-4 ranks on one 16-core machine), the split ran 10-50%
slower than the blocking step. Exchange took 0.1-0.2 ms of a 3-8 ms step.

## Worked sizes: local 32³ and 64³ arrays

For the diffusion benchmark's one-point halos, the temperature launch has
`L=(N-2, N-2, N-2)`. These are **local** array sizes per rank, including
halos. A different global decomposition matters only insofar as it changes
the local launch ranges.

| Local array | Kernel box | Width | Inner kernel box | Main-kernel points computed before exchange |
| :--- | :--- | :--- | :--- | ---: |
| `32³` | `30³` | `(1, 1, 1)` | `28³` | 18.7% |
| `32³` | `30³` | `(2, 2, 2)` | `26³` | 34.9% |
| `32³` | `30³` | `(16, 2, 2)` | None: blocking fallback | 100% |
| `64³` | `62³` | `(1, 1, 1)` | `60³` | 9.4% |
| `64³` | `62³` | `(2, 2, 2)` | `58³` | 18.1% |
| `64³` | `62³` | `(16, 2, 2)` | `30 × 58 × 58` | 57.7% |

These percentages count points, exclude the extra boundary-condition launches,
and are not estimates of elapsed time. With overlap, the inner point count
is `prod(L .- 2 .* b)`, and the outer fraction is
`1 - prod(L .- 2 .* b)/prod(L)`.

At `32³`, `(16, 2, 2)` asks for 16 x points from each side of a 30-point
range. Those sides leave no inner region, so it falls back. At `64³`, the
same choice leaves `62-32=30` inner x points and `62-4=58` inner points in
each other dimension.

The largest integer width allowing overlap is `floor((L[d]-1)/2)`:
14 for a 30-point range and 30 for a 62-point range. These are limits, not
tuning recommendations; widths near the limit leave very little inner work.

## Choosing widths and measuring performance

Start with the inferred halo widths. For this benchmark, `(1, 1, 1)` and
`(2, 2, 2)` are useful candidates at both sizes. At `64³`, also compare
`(16, 2, 2)` if you want to test the wider x shell. These are starting points
for measurement, rather than verified optimal settings.

Smaller widths complete less work before exchange and leave a larger inner
kernel to overlap it. Larger widths delay the start of exchange and reduce
the remaining inner work. Kernel launch overhead, slab shapes, cache behavior,
thread count, and device execution can change the balance. The asymmetric
choice `(16, 2, 2)` gives x a wider computation boundary; it does not change
the exchange order or make x communication overlap the inner kernel.

Use the same local size, rank layout, thread count, and iteration count when
comparing widths. Warm up kernels and exchanges, compare repeated timings,
and check the numerical result against a blocking reference. Changing
`boundary_width` does not change `Field.halowidths`, message sizes, or the
mathematical pointwise update when the dependency conditions above hold and
the width is at least the inferred default.

Compare with the blocking implementation as well. A pending MPI request
does not guarantee that transport progresses during Julia computation;
progress depends on the MPI implementation. Device extensions also stage
data through host buffers around host-only MPI transport. The schedule
allows overlap, but a speedup must be measured for the actual workload.

## Runnable ParallelStencil example

This example computes one periodic 2D Jacobi step into a separate output
field. It compares the macro's result, including refreshed halos, against a
full-kernel blocking reference. The input halos are exchanged first, so the
inner kernel reads only completed input data.

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "..", "examples", "parallelstencil_overlap.jl"), String) * "\n```")
```

ParallelStencil is an optional dependency. From an environment with HaloGrids,
MPI and ParallelStencil (for example `]dev` this repository, then
`]add MPI ParallelStencil`), run from the repository root:

```sh
julia --project=<env> --startup-file=no examples/parallelstencil_overlap.jl 32 2
```

The arguments are the local side length including halos and the computation
width used in both dimensions. The example uses the `Threads` backend, so it
always runs the blocking schedule. With a device backend, `6 2` would take the
small-range fallback (the launch is `4 × 4`, with `4 == 2*2`), and `32 2` would
split with a `26 × 26` inner box. For four MPI ranks, with an MPI launcher
available:

```sh
mpiexec -n 4 julia --project=<env> --startup-file=no examples/parallelstencil_overlap.jl 32 2
```

For an example using the split API directly, see
[the independent-computation example](./halo-exchange.md#worked-example-overlapping-a-stencil-update-with-its-own-halo-exchange).

## Reproducing the sketch

The figure is generated by `docs/figures/boundary_width.jl` using CairoMakie
from the `docs/figures` project (regenerate all figures with `docs/figures/make_all.jl`):

```sh
julia --project=docs/figures --startup-file=no docs/figures/boundary_width.jl
```

The script saves PNG and SVG versions under `docs/src/assets/`. Figure
generation is separate from the documentation build.
