# Runs every test_*.jl file in its own `mpiexec -n N` subprocess, over the rank
# counts `plan` picks from the file name. Weakdep files (GPU, CellArrays, HDF5,
# ParallelStencil, WriteVTK) need the root `[targets]` test deps, which only the
# `Pkg.test()` sandbox has; a plain `julia --project=. test/runtests.jl` (CI)
# skips them. No `import Pkg`: the Pkg.test sandbox does not have it.
import MPI

# (rank counts, extra julia flags, needs weakdeps); rank count 0 means no mpiexec.
function plan(f)
    f == "test_field.jl" && return [0], ``, false
    occursin("threaded", f) && return [1, 2], `-t 4`, false
    occursin(r"cuda|amdgpu", f) && return [1, 2], ``, true    # one local GPU at most
    occursin(r"cellarrays|hdf5|parallelstencil|pvtk", f) && return [1, 2, 4, 8], ``, true
    return [1, 2, 4, 8], ``, false
end

function runfile(f, n, flags)
    # julia_cmd keeps Pkg.test's --check-bounds and coverage flags.
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no $flags $(joinpath(@__DIR__, f))`
    n == 0 && return success(pipeline(cmd; stdout, stderr))
    return MPI.mpiexec(exe -> success(pipeline(`$exe -n $n $cmd`; stdout, stderr)))
end

function main()
    has_weakdeps = Base.find_package("CellArrays") !== nothing
    failed = String[]
    for f in sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
        nprocs, flags, weak = plan(f)
        if weak && !has_weakdeps
            println("\n-- $f (skipped: needs weakdeps, run via Pkg.test()) --")
            continue
        end
        for n in nprocs
            label = strip("$f $(n == 0 ? "(no MPI)" : "-n $n") $(join(flags.exec, ' '))")
            println("\n-- $label --")
            runfile(f, n, flags) || push!(failed, label)
        end
    end
    isempty(failed) || printstyled("\nFailed runs:\n  ", join(failed, "\n  "), "\n"; color=:red)
    return length(failed)
end

exit(main())
