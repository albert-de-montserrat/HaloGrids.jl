---
name: add-optional-extension
description: Add or modify an optional weak-dependency feature or backend (new array type, file format, GPU backend) in HaloGrids.jl using the stub-in-core, methods-in-extension pattern. Use when adding anything that needs a package outside the root [deps].
---

# Add an optional extension

Pattern, as used by HDF5 (`src/HDF5IO.jl`, `ext/HaloGridsHDF5Ext.jl`, `src/HDF5Ext/`):

1. **Stub in core.** Add `src/FooIO.jl` with an `export`, a docstring, and a stub method
   that raises an error naming the package to load. Include it in `src/HaloGrids.jl`
   and list it in the module docstring.
2. **Real code in `src/FooExt/`.** Split by concern (`shared.jl`, `io.jl`, ...).
3. **Thin glue in `ext/HaloGridsFooExt.jl`.** A module that only `include`s the files
   above through `joinpath(@__DIR__, "..", "src", "FooExt", ...)`. Add a header
   comment saying what the extension does not do.
4. **`Project.toml`.** Add the package to `[weakdeps]`, `[extensions]`
   (`HaloGridsFooExt = "Foo"`), `[compat]`, `[extras]`, and `[targets] test`.
5. **Test file.** Add `test/test_foo.jl` and extend the name patterns in
   `plan` (`test/runtests.jl`) so it is marked as needing weakdeps (GPU files
   also skip themselves with `functional()`).
6. **Docs.** Add a guide page, register it in `docs/make.jl`, add API entries
   in `docs/src/api.md`.

Rules:

- Core must not `import` the weak package.
- Array-type backends dispatch on the array type (see `backend.jl`). No flags.
- Extend `Field`/`Grid` machinery. Do not copy exchange logic into the extension.
- Name collisions with the weak package (for example `field`) are handled with
  qualified imports, and documented in the glue file.
- Do not claim support on hardware you cannot test. Say what was and was not run.
- Loading the extension must not change the behavior of code that does not use it.
