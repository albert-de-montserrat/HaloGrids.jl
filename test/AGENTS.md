# Rules for `test/`

Also follow the root `AGENTS.md`. Use the `run-tests` skill to choose a command.

- Name files `test_*.jl`. The driver runs every such file automatically.
  `plan` in `runtests.jl` picks rank counts and flags from the file name;
  extend it for a new special case.
- A test needing CUDA, AMDGPU, CellArrays, HDF5, ParallelStencil, WriteVTK, or
  ReadVTK runs under `Pkg.test()`. Name it so `plan` marks it as needing
  weakdeps, and add the package to root `[extras]`/`[targets]`.
  Never add those packages to the root `[deps]`.
- GPU tests must skip with `functional()` when no GPU exists. They must not fail.
- Tests run under many rank counts (1, 2, 4, 8). Write them to pass for all of
  them. Never hard-code rank count. Test periodic and non-periodic topologies.
- Prefer a position-dependent oracle (value encodes global index) over constant
  fill. Constant fill hides wrong-neighbor and wrong-strip bugs.
- Allocation tests must warm up first, then measure. Keep them in
  `test_allocations*.jl`.
- A skipped test is not a pass. Say which files were skipped and why.
