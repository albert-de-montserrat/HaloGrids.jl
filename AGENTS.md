# Working on HaloGrids.jl

HaloGrids is a Julia package for explicit Cartesian MPI topologies and halo
exchange in 2D and 3D. Read the relevant source and tests before editing.
The current source is authoritative. `../REDESIGN_PLAN.md` (workspace parent, not
in this repo) gives design context and may describe features that do not exist yet.

This file is read by both Claude Code (via `CLAUDE.md`) and Codex. Keep it
tool-neutral. Directory-specific rules live in `src/AGENTS.md`, `test/AGENTS.md`,
and `docs/AGENTS.md`. Task playbooks live in `.agents/skills/`.

## Repository map

Core (`src/`, included in this order by `src/HaloGrids.jl`):

- `topology.jl`: `Topology`, `topology`, `subtopology`, `has_neighbor`.
- `field.jl`: `Field` with per-dimension `halowidths` and `overlaps`.
- `exchange_cache.jl`: byte buffers, MPI requests, `pending` flag, `threaded_pack`.
- `grid.jl`: `Grid`, local sizes (`nlocal` includes halo), owned cache.
- `validation.jl`: shape and duplicate-array checks.
- `backend.jl`: dispatch stubs (`pack_target`, `unpack_source`, `stage_send!`,
  `stage_recv!`) that extensions override.
- `halo_exchange.jl`: full exchange, `update_halo!`, split `halo_exchange_start!`/
  `halo_exchange_finish!`, `@hide_halo_communication` (needs ParallelStencil).
- `face_exchange.jl`: `region=:faces` exchange, separate tag space and buffer slots.
- `gather.jl`: `gather!` with MPI derived datatypes.
- `reduce.jl`: `global_sum/max/min/reduce/allreduce/norm`. Reduce local data with
  plain Julia first and pass one scalar per rank. Only `global_norm` takes an array.
- `coordinates.jl`: `nglobal`, `iglobal`, `coord_g` and `nx_g`/`ix_g`/`x_g` families.
- `HDF5IO.jl`, `PVTKIO.jl`: stubs for `save_hdf5`/`load_hdf5!`/`save_pvtk`.

Optional backends. Each extension file in `ext/` is thin glue that `include`s
real code from a same-named `src/*Ext/` directory:

- `HaloGridsCUDAExt` + `src/CUDAExt/`: tested on this machine (GPU present).
- `HaloGridsAMDGPUExt` + `src/AMDGPUExt/`: mirror of CUDA, untested (no AMD GPU).
- `HaloGridsCellArraysExt` + `src/CellArraysExt/`: decomposes a `CellArray` into
  component views and reuses the normal `Field` path.
- `HaloGridsHDF5Ext` + `src/HDF5Ext/`, `HaloGridsWriteVTKExt` + `src/WriteVTKExt/`.
- `HaloGridsParallelStencilExt`: `@hide_halo_communication` implementation.
- Device selection (rank-to-GPU) is out of scope. GPU-aware MPI is opt-in per grid
  (`grid(...; gpu_aware_mpi=true)`); GPU test files also run those cases when
  `MPI.has_cuda()` is true, which requires running them against a CUDA-aware MPI
  build rather than the default `MPICH_jll`.

Other directories:

- `test/`: standalone test files plus the subprocess driver `runtests.jl`.
  Weakdep tests get their packages from the root `[extras]`/`[targets]`
  via `Pkg.test()`.
- `docs/`: Documenter.jl + DocumenterVitepress site. `examples/`: runnable scripts.
- `benchmarks/`: comparison harness against ImplicitGlobalGrid.jl. Not
  correctness tests, not run by `test/runtests.jl`. `benchmarks/results*.csv`
  are tracked and overwritten by every `run_comparison*.sh` run.

## Implementation conventions

- Keep explicit `Topology` / `Grid` ownership. Do not add a global current grid,
  a shared communication cache, or runtime backend flags.
- Halo widths and overlaps belong to `Field`. `Grid.nlocal` includes halo cells.
- Allow one unfinished exchange per grid. Finish earlier dimensions before packing
  later ones so corner data propagates. Face exchange (`region=:faces`) is the
  exception: it excludes transverse halos and keeps all requested dimensions pending.
- Use separate topologies (separate communicators) for independent concurrent
  communication. Cache ownership alone does not isolate MPI tags.
- Keep tuple specialization and the thin keyword/vararg wrapper pattern in
  exchange code. Measure allocations before simplifying a hot path.
- Packing threads are opt-in (`threaded_pack` at `grid` construction). Never call
  MPI from spawned packing tasks.
- Backend code goes in an extension and dispatches on the array type. Do not add
  `if cuda` branches to core. Optional features follow the stub-in-core,
  methods-in-extension pattern (see `src/HDF5IO.jl`).
- Match the four-space Julia style and mutating `!` naming. Update docstrings and
  the matching docs page when public behavior changes. The API page is generated
  from docstrings, so fix the docstring, not the page.
- Do not describe planned GPU support or performance targets as verified features.
- Preserve existing staged and unstaged work. Do not stage or commit unless asked.

## Validation

Run from the repository root. Do not wrap the driver in `mpiexec`.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no test/test_field.jl          # no MPI needed
julia --project=. --startup-file=no test/runtests.jl            # full matrix
```

Run one MPI test file at a time while developing:

```sh
julia --project=. --startup-file=no -e 'using MPI; MPI.mpiexec() do exe; run(`$exe -n 4 $(Base.julia_cmd()) --project=. --startup-file=no test/test_halo_exchange.jl`); end'
```

The driver runs CPU files at 1, 2, 4, 8 ranks. Threaded files run at 1 and 2 ranks
with `-t 4`. Files that need weakdeps (GPU, CellArrays, HDF5, ParallelStencil,
WriteVTK) are skipped with a printed message under `julia --project=. test/runtests.jl`.
Run `julia --project=. -e 'using Pkg; Pkg.test()'` to include them; it installs the
root `[targets]` test deps. GPU files also skip when no functional GPU is present.

- Use focused tests while developing. Run the full MPI matrix for any change to
  communication, packing, topology, or grid code.
- Allocation changes need warmed-up allocation tests. Threading changes need the
  threaded tests.
- Report skipped checks and environment limits exactly. Do not claim a GPU, HDF5,
  or AMDGPU path passed if the driver skipped it.

CI (`.github/workflows/ci.yml`) is CPU-only on Julia 1.9 and 1.x. It deletes
`Manifest.toml`, runs `test/runtests.jl`, runs four `examples/*.jl` scripts, and
builds docs. Keep code compatible with Julia 1.9 (`Project.toml` compat).

Documentation changes:

```sh
julia --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=docs --startup-file=no docs/make.jl
```

Output goes to `docs/build/1`, which is gitignored. No Node.js step is needed.
Keep internal links valid, public signatures consistent with source, and examples
runnable from the package project. A docs-only edit does not need the MPI matrix;
run new or changed example scripts.
