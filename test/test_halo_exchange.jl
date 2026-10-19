# Correctness test for update_halo!/halo_exchange_start!+finish!.
#
# Strategy: fill each rank's local array with a value that encodes *both* its
# own rank and this cell's position within that rank's array (via
# LinearIndices), not just the rank. A rank-only fill cannot tell a correctly
# sourced neighbor value from one read at the wrong offset within that
# neighbor's array — every cell on one side of a halo boundary holds the same
# value either way, which is exactly the blind spot that let the periodic-2-
# rank tag bug below go undetected until it was found by hand (see that
# test's comment). After update_halo!, every cell's expected value is derived
# analytically by walking the dimensions in the same order the exchange
# processed them and tracking not just which rank's data ends up occupying
# that cell but the exact source index within that rank's *original* array —
# this decomposes independently per dimension even for a corner cell (halo in
# two dimensions at once): dimension d's source-index mapping depends only on
# `ci[d]`/`halowidths[d]`/`nlocal[d]` (identical on every rank, since a grid's
# `nlocal` and a field's `halowidths` are shared across ranks), and which
# neighbor serves dimension d is likewise independent of any other
# dimension's outcome, so applying each dimension's mapping directly to the
# original `ci` gives exactly the same result the real sequential exchange's
# chained corner propagation would (verified by hand against the existing
# 2-rank tag-bug test's known-correct indices before relying on it here).
# Non-periodic domain boundaries still work exactly as before (a halo side
# with no neighbor is simply never written, so both the rank and the source
# index stay this rank's own).
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Large enough to keep (rank, linear index) pairs distinguishable for every
# nlocal/nprocs combination this file uses, without colliding across ranks.
const FILL_SCALE = 100_000.0

fill_value(rank::Integer, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N} =
    Float64(rank) * FILL_SCALE + Float64(LinearIndices(nlocal)[ci])

filled_array(rank::Integer, nlocal::NTuple{N,Int}) where {N} =
    Float64[fill_value(rank, nlocal, ci) for ci in CartesianIndices(nlocal)]

# Expected value at local CartesianIndex `ci` of a field with `halowidths` on
# `grid`, after `update_halo!(grid, field; dims)`.
function expected_value(grid::Grid{N}, halowidths::NTuple{N,Int}, dims, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N}
    topo = grid.topology
    offset = zeros(Int, N)
    source_idx = collect(Tuple(ci))
    for d in dims
        hw = halowidths[d]
        idx = ci[d]
        dir = idx <= hw ? -1 : (idx > nlocal[d] - hw ? 1 : 0)
        dir == 0 && continue
        if topo.periods[d] || (0 <= topo.coords[d] + dir <= topo.dims[d] - 1)
            offset[d] += dir
            nA = nlocal[d]
            # This rank's low halo receives its lo-neighbor's high-direction
            # send (interior strip nA-2hw+1:nA-hw); the high halo receives
            # its hi-neighbor's low-direction send (interior strip
            # hw+1:2hw) — see the EXCHANGE_TAG_LO/HI comment in
            # halo_exchange.jl for why direction, not post order, decides
            # this. Both reduce to the same affine map in terms of `idx`.
            source_idx[d] = dir == -1 ? (nA - 2hw + idx) : (idx - nA + 2hw)
        end
        # else: no neighbor on that side in dimension d; nothing changes for dim d.
    end
    if all(iszero, offset)
        return fill_value(rank, nlocal, ci)
    end
    target_coords = ntuple(N) do d
        c = topo.coords[d] + offset[d]
        topo.periods[d] ? mod(c, topo.dims[d]) : c
    end
    target_rank = MPI.Cart_rank(topo.comm, collect(target_coords))
    return fill_value(target_rank, nlocal, CartesianIndex(source_idx...))
end

function check_halo(grid::Grid{N}, f::Field{Float64,N}, dims, nlocal) where {N}
    ok = true
    for ci in CartesianIndices(f.data)
        expected = expected_value(grid, f.halowidths, dims, nlocal, ci)
        if f.data[ci] != expected
            @error "mismatch" rank ci got=f.data[ci] expected
            ok = false
        end
    end
    return ok
end

@testset "update_halo! correctness (N=$N, periods=$periods, dims_order=$dims_order)" for N in (2, 3),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N)),
    dims_order in (Tuple(1:N), Tuple(reverse(1:N)))

    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = filled_array(rank, nlocal)
    B = filled_array(rank, nlocal)
    hwA = ntuple(_ -> 1, N)
    hwB = ntuple(d -> d == 1 ? 2 : 1, N)
    fA = field(A; halowidths=hwA)
    fB = field(B; halowidths=hwB)

    update_halo!(g, fA, fB; dims=dims_order)

    @test check_halo(g, fA, dims_order, nlocal)
    @test check_halo(g, fB, dims_order, nlocal)
end

@testset "start!/finish! matches update_halo! (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)
    A = filled_array(rank, nlocal)
    fA = field(A; halowidths=ntuple(_ -> 1, N))

    handle = halo_exchange_start!(g, fA)
    # Work inserted between start! and finish! must not corrupt the result.
    scratch = zeros(4, 4)
    scratch .+= 1.0
    halo_exchange_finish!(handle)

    @test check_halo(g, fA, 1:N, nlocal)
end

@testset "bare-array convenience update_halo!/halo_exchange_start! (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    # A bare array: gets field's default halowidths (1 in every dimension).
    A = filled_array(rank, nlocal)
    update_halo!(g, A)
    @test check_halo(g, field(A), 1:N, nlocal)

    # Mixed call: B stays a bare array (default width), C is pre-wrapped with
    # an explicit, different width — this is the "variable width in one call"
    # case, resolved per-argument rather than by one call-wide default.
    B = filled_array(rank, nlocal)
    C = filled_array(rank, nlocal)
    hwC = ntuple(d -> d == 1 ? 2 : 1, N)
    update_halo!(g, B, field(C; halowidths=hwC))
    @test check_halo(g, field(B), 1:N, nlocal)
    @test check_halo(g, field(C; halowidths=hwC), 1:N, nlocal)

    # start!/finish! convenience form matches the update_halo! convenience form.
    D = filled_array(rank, nlocal)
    handle = halo_exchange_start!(g, D)
    halo_exchange_finish!(handle)
    @test check_halo(g, field(D), 1:N, nlocal)
end

@testset "one in-flight exchange per Grid" begin
    topo = topology((0, 0); comm=comm, periods=(true, true))
    nlocal = (6, 6)
    g = grid(topo, nlocal)
    A = fill(Float64(rank), nlocal)
    fA = field(A; halowidths=(1, 1))

    handle = halo_exchange_start!(g, fA)
    @test_throws ErrorException halo_exchange_start!(g, fA)
    halo_exchange_finish!(handle)
    # After finishing, starting again must work.
    handle2 = halo_exchange_start!(g, fA)
    halo_exchange_finish!(handle2)
end

@testset "stale handles, bad dims, and too-small arrays are rejected (region=$region)" for region in (:full, :faces)
    topo = topology((0, 0); comm=comm, periods=(true, true))
    g = grid(topo, (6, 6))
    fA = field(fill(Float64(rank), 6, 6))
    fB = field(fill(Float64(rank), 6, 6))

    # Double finish, and finishing a stale handle while a newer exchange is
    # pending, must error and leave the newer exchange intact.
    h1 = halo_exchange_start!(g, fA; region)
    halo_exchange_finish!(h1)
    @test_throws ErrorException halo_exchange_finish!(h1)
    h2 = halo_exchange_start!(g, fB; region)
    @test_throws ErrorException halo_exchange_finish!(h1)
    @test g.excache.pending
    halo_exchange_finish!(h2)
    @test !g.excache.pending

    # A non-integer dim errors before the grid is marked pending.
    @test_throws ErrorException update_halo!(g, fA; dims=(1.0,), region)
    @test !g.excache.pending

    # overlaps + halowidths > size: the sent strip would contain halo cells.
    small = field(zeros(6, 6); halowidths=(3, 1))     # 6 + 3 > 6 in dim 1
    @test_throws ErrorException update_halo!(g, small; dims=(1,), region)
    update_halo!(g, small; dims=(2,), region)         # dim 2 is fine
    wide = field(zeros(6, 6); overlaps=(4, 2))        # 4 + 1 <= 6
    update_halo!(g, wide; region)
    tight = field(zeros(6, 6); overlaps=(6, 2))       # 6 + 1 > 6
    @test_throws ErrorException update_halo!(g, tight; region)
    @test !g.excache.pending

    # No rank communicates in a non-periodic single-rank dimension: no check there.
    nprocs = MPI.Comm_size(comm)
    gthin = grid(topology((nprocs, 1); comm=comm, periods=(true, false)), (8, 2))
    update_halo!(gthin, field(zeros(8, 2)); region)
    @test !gthin.excache.pending
end

@testset "periodic dimension split across exactly 2 ranks: direction, not post order, decides which strip fills which halo" begin
    # Regression test for a real bug found 2026-09-13: when a periodic
    # dimension has exactly 2 ranks, `lo` and `hi` (this rank's two neighbors
    # in that dimension) are the *same* physical rank. With a single shared
    # MPI tag for every message (the original implementation), MPI's
    # "non-overtaking" ordering guarantee matches messages by the order they
    # were posted, not by which physical direction they represent — and both
    # ranks post their low-direction send/recv before their high-direction
    # one, so this rank's low halo silently received its neighbor's *low*-side
    # interior strip instead of the intended *high*-side one (and vice versa).
    # The existing `check_halo`-based tests above cannot catch this: they fill
    # every cell with this rank's own number uniformly, so the low-side and
    # high-side interior strips hold identical values and a swap between them
    # is invisible. This test fills each rank's array with distinct,
    # position-dependent values instead, on a communicator restricted to
    # exactly 2 ranks (so the bug is exercised regardless of how many ranks
    # this test file actually runs under, as long as there are at least 2).
    if MPI.Comm_size(comm) >= 2
        color = rank < 2 ? 0 : 1
        subcomm = MPI.Comm_split(comm, color, rank)
        if rank < 2
            t2 = topology((2, 1); comm=subcomm, periods=(true, false))
            nlocal2 = (8, 3)
            data = zeros(8, 3)
            for j in 1:3
                data[:, j] = collect(1.0:8.0) .+ 100 * t2.rank
            end
            g2 = grid(t2, nlocal2)
            f2 = field(data; halowidths=(1, 1))
            update_halo!(g2, f2; dims=(1,))

            other = 1 - t2.rank
            # Low halo must hold the neighbor's *high*-side interior strip
            # (local index end-1 = 7 on an 8-wide array with halowidth 1).
            @test all(f2.data[1, :] .== 7.0 + 100 * other)
            # High halo must hold the neighbor's *low*-side interior strip
            # (local index 2).
            @test all(f2.data[end, :] .== 2.0 + 100 * other)
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_halo_exchange.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
