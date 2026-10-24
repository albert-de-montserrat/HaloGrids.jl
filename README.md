# HaloGrids.jl

[![CI](https://github.com/albert-de-montserrat/HaloGrids.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/albert-de-montserrat/HaloGrids.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://albert-de-montserrat.github.io/HaloGrids.jl/dev/)
[![Julia compatibility](https://img.shields.io/badge/Julia-1.9%2B-9558B2.svg)](Project.toml)

HaloGrids.jl exchanges halo cells between Julia arrays distributed over
Cartesian MPI process layouts in 2D and 3D. Each application explicitly owns
its topology, grids, and fields, making communication state and array layout
visible at the call site.

- **`Topology`** owns a Cartesian communicator, process coordinates, and neighbors.
- **`Grid`** describes the full local array shape, including halos, and owns reusable communication buffers.
- **`Field`** wraps an existing array and declares its halo thickness in each dimension.

The package provides periodic and nonperiodic boundaries, multi-field exchange,
global coordinate helpers, gathering, and scalar MPI reductions. Sequential
dimension exchanges propagate edge and corner values.

Read the [getting-started guide](docs/src/guide/getting-started.md) or browse
the [online documentation](https://albert-de-montserrat.github.io/HaloGrids.jl/dev/).

## Installation

Requires Julia 1.9 or newer. To add the package from GitHub to your active Julia
environment:

```julia
using Pkg
Pkg.add(url="https://github.com/albert-de-montserrat/HaloGrids.jl.git")
```

For an existing checkout, instantiate its project from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## A periodic halo exchange

Each rank below stores a `10 × 10` array: an `8 × 8` owned interior and a
one-cell halo on every side. Zeros in the topology dimensions let MPI choose
the process layout; they do not specify array sizes.

```julia
using HaloGrids
import MPI

MPI.Init()
try
    t = topology((0, 0); periods=(true, true))
    g = grid(t, (10, 10))
    A = fill(-1.0, g.nlocal)
    A[2:end-1, 2:end-1] .= t.rank
    f = field(A; halowidths=(1, 1))

    update_halo!(g, f)
    # A's halos now contain the neighboring ranks' interior values.
finally
    MPI.Finalize()
end
```

Run [the complete example](examples/quickstart.jl), which also checks the
received face values:

```sh
julia --project=. --startup-file=no examples/quickstart.jl
```

For four ranks, start a Julia REPL with `julia --project=.` and use MPI.jl's
configured launcher. This also works when `mpiexec` is absent from your shell's
`PATH`:

```julia
import MPI
MPI.mpiexec() do exe
    run(`$exe -n 4 $(Base.julia_cmd()) --project=. --startup-file=no examples/quickstart.jl`)
end
```

The solver sets physical boundary conditions at nonperiodic domain walls.
All participating ranks must execute matching communication calls in the same order.

## Overlapping computation and communication

`update_halo!` completes the exchange before returning. The split API,
`halo_exchange_start!` / `halo_exchange_finish!`, lets independent computation
run while the final dimension's exchange is pending. Earlier dimensions finish
inside `start!` so corner data can propagate. Only one exchange may be pending
per grid; finish it before starting another.

Loading ParallelStencil enables `@hide_halo_communication`, which computes
outer kernel slabs before exchange and an inner box while communication is
pending. `boundary_width=(bx, by, bz)` gives the computation thickness at
**each side** of each dimension. It leaves `Field.halowidths` and message sizes
unchanged. Small launch ranges and CPU backends (`Threads`, `Polyester`) use a
full-kernel, blocking-exchange fallback.
See [hiding communication and boundary widths](docs/src/guide/hide-communication.md)
for sketches, safe kernel patterns, and worked `32³`/`64³` examples.

The schedule permits overlap; performance depends on the workload and MPI's
progress behavior.

## Optional integrations

Extensions activate when their corresponding packages are loaded:

| Package | Integration |
| :--- | :--- |
| CUDA | `CuArray` halo exchange, tested on real NVIDIA hardware |
| AMDGPU | `ROCArray` halo exchange; implemented but untested on real AMD hardware |
| CellArrays | Halo exchange for `CellArray` data, including a tested CUDA path |
| ParallelStencil | Outer/inner computation splitting with `@hide_halo_communication` |
| HDF5 | Saving and loading distributed fields in a shared HDF5 file |
| WriteVTK | Saving distributed fields as parallel ParaView (pvtk) files |

GPU packing and unpacking use device buffers, with host staging around MPI
transport by default. With a CUDA- or ROCm-aware MPI library,
`grid(...; gpu_aware_mpi=true)` passes device buffers to MPI directly (tested
for CUDA with Open MPI 5 and UCX). The application selects each rank's device.
HDF5 I/O uses ranks in turn and does not require a parallel-HDF5 build. Parallel VTK I/O is write-only
(no `load_pvtk!`); every rank writes its own independent file, so unlike
HDF5 it needs no shared-file write coordination.

See the [public API](docs/src/api.md) for signatures and the
[examples directory](examples/) for runnable applications.

## Development and documentation

Run checks from the repository root:

```sh
julia --project=. --startup-file=no test/test_field.jl
julia --project=. --startup-file=no test/runtests.jl
```

The test driver starts MPI subprocesses at 1, 2, 4, and 8 ranks, plus threaded
packing checks at 1 and 2 ranks with four Julia threads. Run it directly,
without an outer `mpiexec`. Optional integration tests run under
`julia --project=. -e 'using Pkg; Pkg.test()'` and report skipped checks when dependencies or usable hardware are unavailable.

The documentation uses Documenter.jl and DocumenterVitepress.jl. Build from the
repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs --startup-file=no docs/make.jl
```

DocumenterVitepress manages the Node dependencies. On Windows, it asks you to
complete the VitePress HTML build separately. See
[development instructions](docs/src/contributing.md) for previewing the site,
optional tests, and GitHub Pages deployment. Benchmark harnesses under
[benchmarks/](benchmarks/) run separately from the correctness tests.
