# GPU counterpart to test_allocations.jl (phase 8's remaining "apply the same
# allocation standard to the GPU code" item — see REDESIGN_PLAN.md).
#
# Unlike the CPU path, the CUDA exchange path is NOT expected to measure 0
# bytes/call, even at nprocs=1 self-periodic. Bisected the same way as the CPU
# work (isolated, warmed-up, amortized @allocated measurements run *inside a
# function*, comparing against a byte-for-byte-identical CPU Vector control to
# separate "HaloGrids' code" from "CUDA.jl/GPUArrays.jl's own cost"), the
# floor comes from two costs entirely internal to CUDA.jl/GPUArrays.jl, not
# from anything this package's code does:
#
#   1. ~80 B per newly-constructed `CuArray` wrapper (a `view` or a
#      `reinterpret`, additive when composed, as `typed_buf!` in
#      exchange_cache.jl does). Running the identical wrapper-construction
#      code (`view(reinterpret(T, buf), 1:len)`) on a plain CPU `Vector{UInt8}`
#      measures exactly 0 B — proving this is `CuArray`'s own construction
#      cost (it needs to register/track a new device-memory reference for its
#      GC-integrated buffer pool), not a HaloGrids.jl inlining failure. This
#      part *is* addressed: `CUDACache`/`ROCCache` (src/CUDAExt/shared.jl,
#      src/AMDGPUExt/shared.jl) now cache the constructed view itself
#      (`ViewCache`/`cached_typed_view!`), keyed on `(eltype, len, rawlen)`,
#      and reuse it across calls once a field's buffer size stabilizes —
#      `rawlen` (the raw buffer's `length` when the view was built) is part of
#      the key because `resize!` mutates the raw buffer in place, so a view
#      built before a resize would otherwise reference stale memory after one.
#      Measured saving: ~3840 B/call for the 2-field/2-dim setup below (see
#      REDESIGN_PLAN.md's GPU allocation audit for the before/after).
#   2. A device `copyto!` with a non-contiguous (strided) source or
#      destination goes through GPUArrays.jl's generic broadcast machinery:
#      ~2.3 KB per call on CUDA.jl 6.3 (~1-1.2 KB on older versions) versus
#      ~112 B for a contiguous copy. Addressed: strided slabs now use a
#      one-line copy kernel (~670 B, CUDA.jl's own launch cost; see
#      src/CUDAExt/halo_exchange.jl).
#
# Total scales with the number of pack/unpack/self-periodic-copy operations
# per call (~field count x dimension count x 2 directions), not with array
# size. Measured on this machine (RTX 5070 Ti, CUDA.jl on CUDA 13.4 driver),
# 2 fields (Float64 halowidths=(1,1), Float32 halowidths=(2,1)) over 2 dims,
# 20x20 local size: a rock-solid 14336 B/call at nprocs=1 (self-periodic only,
# isolating the GPU-array cost above from MPI transport; identical across
# repeated runs, as expected with no real network transport in play) and
# 17000-30000 B/call at nprocs=2 (adds MPI.jl's own Isend/Irecv/Wait overhead
# plus the device<->host staging copies stage_send!/stage_recv! do) — that
# nprocs=2 range is itself noisy run-to-run (observed 16664-29981 across
# repeated trials, `update_halo!` and the equivalent manual `start!`+`finish!`
# pair sometimes differing by close to 2x from each other despite running
# identical code), consistent with the already-documented "nprocs>1 residual
# is MPI.jl's own call overhead, not this package's code, so not chased
# further" call made for the CPU path's real-MPI-transport residual (see
# test_allocations.jl) — just present at nprocs=1 too here (deterministically,
# unlike the nprocs=2 noise), because that part of the cost lives in the GPU
# array library, not in MPI.
#
# This test exists to catch a real regression (e.g. losing buffer/view reuse
# and allocating fresh device objects every call), not to hold the GPU path to
# the CPU path's zero-allocation bar. Bounds are set well above the measured
# numbers above to avoid flakiness across CUDA.jl/GPUArrays.jl versions, GPU
# models, and the observed nprocs>1 run-to-run noise, while still catching an
# order-of-magnitude regression.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using CUDA

if !CUDA.functional()
    println("test_allocations_cuda.jl: CUDA not functional on this machine; skipping.")
    exit(0)
end

using HaloGrids

function run_gpu_allocation_tests()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    t = topology((0, 0); comm=comm, periods=(true, true))
    g = grid(t, (20, 20))
    A = CuArray(fill(Float64(rank), 20, 20))
    B = CuArray(fill(Float32(rank), 20, 20))
    fA = field(A; halowidths=(1, 1))
    fB = field(B; halowidths=(2, 1))

    # Warmup: let device buffers grow to their high-water mark and every
    # method specialization JIT-compile before measuring.
    for _ in 1:5
        update_halo!(g, fA, fB)
    end
    CUDA.synchronize()

    niter = 200
    total_update = @allocated for _ in 1:niter
        update_halo!(g, fA, fB)
    end
    CUDA.synchronize()
    bytes_per_update = total_update / niter

    total_pair = @allocated for _ in 1:niter
        h = halo_exchange_start!(g, fA, fB)
        halo_exchange_finish!(h)
    end
    CUDA.synchronize()
    bytes_per_pair = total_pair / niter

    if rank == 0
        println("nprocs=$nprocs  update_halo! (GPU): $bytes_per_update B/call  start!+finish!: $bytes_per_pair B/call")
    end

    @testset "GPU steady-state allocation bound (nprocs=$nprocs)" begin
        if nprocs == 1
            # Self-periodic only (no real MPI transport): isolates the
            # CuArray-wrapper + copy floor described above. Measured a stable
            # 8448 B/call for this 2-field/2-dim setup on CUDA.jl 6.3 (23296
            # with the broadcast copyto! for strided slabs) — tighter bound
            # than nprocs>1 since this case has no transport noise.
            @test bytes_per_update < 12000
            @test bytes_per_pair < 12000
        else
            # Real device<->host staging + MPI Isend/Irecv/Wait transport is
            # also in play now, and genuinely noisy run-to-run (observed
            # 16664-29981 B/call across repeated trials on this machine) —
            # bound loosely enough to never be flaky against that.
            @test bytes_per_update < 40000
            @test bytes_per_pair < 40000
        end
    end

    MPI.Barrier(comm)
    rank == 0 && println("test_allocations_cuda.jl: all tests passed on $nprocs ranks.")
    MPI.Finalize()
end

run_gpu_allocation_tests()
