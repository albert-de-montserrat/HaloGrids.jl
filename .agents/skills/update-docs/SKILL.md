---
name: update-docs
description: Update HaloGrids.jl documentation, docstrings, guide pages, or runnable examples and build the Documenter/VitePress site. Use when public behavior, signatures, or examples change.
---

# Update docs

Read `docs/AGENTS.md` first.

## Steps

1. Change the **docstring** in `src/` first. `docs/src/api.md` is generated from it.
2. Update the matching guide in `docs/src/guide/`:
   - exchange behavior: `halo-exchange.md`, `hide-communication.md`
   - types, topology, overlaps, subtopology: `concepts.md`
   - I/O: `hdf5-io.md`, `pvtk-io.md`
   - reductions, coordinates, gather: `reductions.md`, `coordinates.md`, `gather.md`
   - ImplicitGlobalGrid differences: `migration.md`
3. Executable code belongs in `examples/*.jl`. Pages embed these through `@eval`
   blocks. Edit the example, not a pasted copy.
4. New page: register it in `pages` in `docs/make.jl`.
5. Update `README.md` if a headline feature or install instruction changed.

## Verify

```sh
julia --project=docs --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=docs --startup-file=no docs/make.jl
```

Documenter fails on broken links and missing docstrings. Fix them. Do not silence them.

Run every new or changed example from the package project, for example:

```sh
julia --project=. --startup-file=no examples/quickstart.jl
```

CI runs `quickstart.jl`, `hide_communication.jl`, `face_exchange.jl`, `diffusion3D.jl`.
Examples that need weak dependencies (`gaussian3d_pvtk.jl`, `checkpoint_restart.jl`,
`parallelstencil_overlap.jl`) need an environment with that package; say if you ran them.

Do not commit `docs/build/`, `node_modules/`, or cache directories.

## Figures

Makie scripts live in `docs/figures/` (read the Figures section of `docs/AGENTS.md`).
After changing behavior that a figure depicts (halo geometry, overlaps, exchange order,
face exchange, split-phase), update the script and regenerate:

```sh
julia --project=docs/figures --startup-file=no docs/figures/make_all.jl
```

Then run `scripts/sync-agent-skills.sh` if you edited this skill.
