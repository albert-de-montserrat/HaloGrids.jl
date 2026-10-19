# Demonstrates hiding the last dimension's halo communication behind
# independent compute, using one step of 2D five-point-stencil diffusion as
# the motivating example.
#
# halo_exchange_start! completes every dimension except the last fully
# synchronously; only the last dimension's communication is left in flight
# when it returns. That means, right after start! returns with dims=(1, 2):
#   - T's dimension-1 halo (rows 1 and end)      : already correct.
#   - T's dimension-2 halo (columns 1 and end)   : NOT yet correct — in flight.
#
# A 5-point stencil at column j reads columns j-1 and j+1. So the columns
# whose stencil never touches a dimension-2 halo column can be updated while
# that exchange is still outstanding; only the two columns immediately next
# to the dimension-2 edges must wait for halo_exchange_finish!.
using HaloGrids
import MPI

MPI.Init()
try
    t = topology((0, 0); periods=(true, true))
    g = grid(t, (18, 18))  # 16x16 interior, halowidths=(1,1)
    hw = (1, 1)

    T = fill(0.0, g.nlocal)
    T[2:end-1, 2:end-1] .= t.rank + 1.0  # distinguishable per-rank interior values
    fT = field(T; halowidths=hw)

    nrows, ncols = size(T)
    stencil(i, j) = 0.25 * (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1])

    # -- Reference: the straightforward blocking computation -----------------
    # Same input, full synchronous exchange, then the whole interior in one
    # pass. This is what every call site should reach for by default; it is
    # only reproduced here to prove the overlapped version below computes the
    # exact same answer, not a different (if plausible-looking) one.
    Tref = copy(T)
    update_halo!(g, field(Tref; halowidths=hw))
    Tnew_ref = similar(T)
    for j in 2:ncols-1, i in 2:nrows-1
        Tnew_ref[i, j] = 0.25 * (Tref[i-1, j] + Tref[i+1, j] + Tref[i, j-1] + Tref[i, j+1])
    end

    # -- Overlapped version ----------------------------------------------------
    Tnew = similar(T)

    # dims=(1, 2): dimension 1 is exchanged synchronously inside start!;
    # dimension 2 (the last requested) is left outstanding in the handle.
    handle = halo_exchange_start!(g, fT; dims=(1, 2))

    # Safe to compute now: every column whose stencil neighbors (j-1, j+1)
    # avoid the still-in-flight dimension-2 halo columns (1 and end).
    safe_cols = (hw[2]+2):(ncols-hw[2]-1)
    for j in safe_cols, i in 2:nrows-1
        Tnew[i, j] = stencil(i, j)
    end

    # This is the point at which real work would go: independent compute
    # (a different field's update, a diagnostic reduction, packing output,
    # ...) that doesn't read T's dimension-2 halo and doesn't write T itself.
    # Here it's the safe_cols loop above standing in for that work.

    halo_exchange_finish!(handle)  # dimension 2's halo is valid from here on

    # Only now: the two columns adjacent to the dimension-2 edges, whose
    # stencil needed the halo we just finished waiting for.
    for j in (hw[2]+1, ncols-hw[2]), i in 2:nrows-1
        Tnew[i, j] = stencil(i, j)
    end

    @assert Tnew[2:end-1, 2:end-1] == Tnew_ref[2:end-1, 2:end-1]
    MPI.Barrier(t.comm)
    t.rank == 0 && println("Overlapped diffusion step matches the blocking reference on $(prod(t.dims)) rank(s).")
finally
    MPI.Finalize()
end
