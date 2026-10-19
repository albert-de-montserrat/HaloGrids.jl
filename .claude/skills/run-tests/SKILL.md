---
name: run-tests
description: Pick and run the right HaloGrids.jl tests for a change (focused file, MPI matrix, threaded, test-project, GPU) and report skips accurately. Use after any source edit or when asked to verify.
---

# Run HaloGrids tests

Run from the repository root. Never wrap `test/runtests.jl` in `mpiexec`.

## 1. Choose scope

| Change | Run |
| --- | --- |
| `Field`, validation only | `julia --project=. --startup-file=no test/test_field.jl` |
| topology, grid, exchange, pack, gather, reduce, coordinates | one MPI file at 2 and 4 ranks, then full driver |
| `threaded_pack` code | `test_threaded_pack.jl`, `test_face_exchange_threaded.jl` (driver runs them with `-t 4`) |
| allocation-sensitive hot path | `test_allocations.jl` (CPU), `test_allocations_cuda.jl` (GPU) |
| HDF5, WriteVTK, CellArrays, ParallelStencil ext | `Pkg.test()` (the plain driver skips them) |
| CUDA/AMDGPU ext | `test_cuda.jl`, `test_face_exchange_cuda.jl` via `Pkg.test()` |
| docs only | no MPI tests; see `update-docs` skill |

## 2. Commands

One MPI file (change `-n` and file name):

```sh
julia --project=. --startup-file=no -e 'using MPI; MPI.mpiexec() do exe; run(`$exe -n 4 $(Base.julia_cmd()) --project=. --startup-file=no test/test_halo_exchange.jl`); end'
```

Full driver (1, 2, 4, 8 ranks, plus threaded files; skips weakdep files):

```sh
julia --project=. --startup-file=no test/runtests.jl
```

Including weakdep and GPU files (installs the root `[targets]` test deps):

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

## 3. Report

- List each file run and the rank counts.
- A file the driver prints as `skipped` was not tested. Say so. Do not call it passing.
- GPU files skip when no functional GPU exists. AMDGPU cannot run on this machine.
- If oversubscription fails on 8 ranks, set `OMPI_MCA_rmaps_base_oversubscribe=1`
  and say you did.
- Quote the shortest failing line and its rank count on failure.
