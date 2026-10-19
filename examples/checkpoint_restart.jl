# Run in an environment with HaloGrids, MPI and the optional HDF5 dependency.
# Save on one rank count, then load the same file on another (see the I/O guide).
using HaloGrids
using HDF5
import MPI

length(ARGS) == 2 || error("Usage: checkpoint_restart.jl save|load checkpoint.h5")
action, path = ARGS
action in ("save", "load") || error("Expected save or load, got $action.")

MPI.Init()
try
    t = topology((0, 0); periods = (true, true))
    global_shape = (64, 64)
    all(d -> global_shape[d] % t.dims[d] == 0, 1:2) ||
        error("The process dimensions $(t.dims) must divide $global_shape.")
    nlocal = ntuple(d -> global_shape[d] ÷ t.dims[d] + 2, 2)
    g = grid(t, nlocal)
    f = field(zeros(nlocal))
    indices = LinearIndices(global_shape)
    expected(ci) = indices[iglobal(g, f, 1, ci[1]), iglobal(g, f, 2, ci[2])]

    if action == "save"
        for ci in CartesianIndices(f.data)
            f.data[ci] = expected(ci)
        end
        save_hdf5(g, f, path)
    else
        load_hdf5!(g, f, path; repartition = true)
        update_halo!(g, f)
        @assert all(ci -> f.data[ci] == expected(ci), CartesianIndices(f.data))
    end
    MPI.Barrier(t.comm)
    t.rank == 0 && println("$action: $path on $(MPI.Comm_size(t.comm)) ranks, topology $(t.dims)")
finally
    MPI.Finalize()
end
