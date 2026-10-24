# Run from the repository root in an environment with HaloGrids, MPI and ParallelStencil.
# Arguments: local side length including halos, computation width per side.
using HaloGrids
using MPI
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function jacobi_step!(B, A)
    B[i + 1, j + 1] = 0.25 * (
        A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])
    return nothing
end

function main()
    N = isempty(ARGS) ? 32 : parse(Int, ARGS[1])
    b = length(ARGS) < 2 ? 2 : parse(Int, ARGS[2])
    N >= 4 || error("local side length must be at least 4")
    b > 0 || error("computation width must be positive")
    t = topology((0, 0); periods=(true, true))
    g = grid(t, (N, N))
    A = [sin(i + 2j + 10t.rank) for i in 1:N, j in 1:N]
    input = field(A; halowidths=(1, 1))
    output = field(zeros(N, N); halowidths=(1, 1))
    reference = field(zeros(N, N); halowidths=(1, 1))
    B, Bref = output.data, reference.data
    ranges = (1:N-2, 1:N-2)

    # Prepare the input for both computations.
    update_halo!(g, input)

    # Full kernel followed by blocking output exchange.
    @parallel ranges jacobi_step!(Bref, A)
    @synchronize
    update_halo!(g, reference)

    # On a device backend: compute output boundaries first, then exchange
    # output while computing its inner region. A is a separate, already
    # exchanged input array. This file's Threads backend always takes the
    # blocking schedule, so it shows the macro's usage and result only.
    @hide_halo_communication g output boundary_width=(b, b) begin
        @parallel ranges jacobi_step!(B, A)
    end
    @synchronize
    @assert B == Bref

    t.rank == 0 && println("@hide_halo_communication matches the blocking reference on $(prod(t.dims)) rank(s).")
end

MPI.Init()
try
    main()
finally
    MPI.Finalize()
end
