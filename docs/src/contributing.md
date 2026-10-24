# Development and documentation

## Julia checks

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no test/test_field.jl
julia --project=. --startup-file=no test/runtests.jl
```

The full driver starts its own MPI subprocesses at 1, 2, 4, and 8 ranks. Threaded
packing checks run separately with four Julia threads at 1 and 2 ranks. Run the
driver directly, without an outer MPI launcher. Allow enough process slots for
eight ranks, or run focused files with an appropriate MPI.jl launcher while developing.

## Continuous integration

GitHub Actions runs on pushes, pull requests, and manual dispatches. The Julia
jobs test Julia 1.9 and the latest stable 1.x release on Linux, using the full
MPI and threaded test driver and the quickstart example. Dependencies are resolved
from `Project.toml` in each job rather than the development manifest. The CPU
driver is invoked directly; the CUDA/AMDGPU extensions are implemented and
tested locally (see the package's `REDESIGN_PLAN.md`) but CI does not install
their test extras (the root `[extras]`/`[targets]`, installed by `Pkg.test()`) —
doing so would add real, un-opted-into time and network cost to every job for
runners that have no GPU to test against anyway. `test/runtests.jl` detects
this and prints a one-line skip for the GPU test files rather than failing.

The GPU test files also run `gpu_aware_mpi=true` cases when `MPI.has_cuda()`
is `true`. The test driver launches ranks with MPI.jl's default `MPICH_jll`,
which is not CUDA-aware, so it skips those cases. To run them, launch a GPU test
file with a CUDA-aware MPI whose MPIPreferences are set for the `test`
environment, for example:

```sh
mpiexec -n 2 julia --project=test --startup-file=no test/test_cuda.jl
```

See [GPU arrays and GPU-aware MPI](./guide/gpu.md) for configuring MPI.jl.

A separate job builds this site with `docs/make.jl`. Documenter's own link and
docstring checks run as part of that build.

## Work on this site

This site is built with [Documenter.jl](https://documenter.juliadocs.org/stable/)
and rendered to HTML by [DocumenterVitepress.jl](https://github.com/LuxDL/DocumenterVitepress.jl),
which drives [VitePress](https://vitepress.dev/) internally — no repository-level
Node.js setup is needed, since DocumenterVitepress manages its own Node
dependencies. Only Julia is required. From the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

This writes the site to `docs/build/1` (see DocumenterVitepress's own docs for
why the numbered subfolder: locally it is always `1`, since a local build is
neither a dev, version, nor PR-preview build). Preview it with
[LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl):

```julia
using Pkg; Pkg.activate("docs"); Pkg.add("LiveServer")
using LiveServer
LiveServer.serve(dir = "docs/build/1")
```

Or, for an automatically rebuilding preview while editing:

```julia
using LiveServer
LiveServer.servedocs(buildfoldername = "build/1")
```

The quickstart, halo-exchange overlap, ParallelStencil boundary-width, and
coordinate-helper pages embed `examples/quickstart.jl`,
`examples/hide_communication.jl`, `examples/parallelstencil_overlap.jl`, and
`examples/diffusion3D.jl` directly via `@eval` blocks that read the file at
build time, so edits to their executable source also update the page. The API
reference (`docs/src/api.md`) is built entirely from `@docs` blocks pulling
docstrings out of `src/`, so it cannot drift from the implementation the way a
hand-written signature list can — update the docstring, not the page, when a
public signature changes.

## Internal validation and backend hooks

The following implementation hooks are documented for contributors. They are
not exported public API; array backends override the dispatch hooks through
package extensions.

```@docs
HaloGrids.check_fields
HaloGrids.pack_target
HaloGrids.mpi_sendbuf
```

## Deployment

`docs/make.jl` ends with a `DocumenterVitepress.deploydocs` call targeting the
`gh-pages` branch of this repository. It only actually pushes when run in a CI
job on a qualifying push (see
[DocumenterVitepress's deployment docs](https://github.com/LuxDL/DocumenterVitepress.jl#deploying));
running `make.jl` locally only ever produces the local `docs/build/1` output
above. Enabling live deployment additionally requires GitHub Pages to be
configured for this repository and, if deploying across repository
permissions rather than the default `GITHUB_TOKEN`, a `DOCUMENTER_KEY` deploy
key — see Documenter.jl's
[hosting documentation](https://documenter.juliadocs.org/stable/man/hosting/)
for the general mechanism (DocumenterVitepress's deploy step follows the same
key/branch conventions).
