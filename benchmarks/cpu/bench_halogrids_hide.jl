# CPU (multi-threaded) benchmark worker for HaloGrids.jl's "hide
# communication" pattern — a ParallelStencil `Threads`-backend port of
# benchmarks/bench_halogrids_hide.jl (itself a 3D generalization of
# examples/hide_communication.jl's overlap-vs-blocking concept to a
# 7-point-Laplacian diffusion step), instrumented for this directory's
# metric set (see metrics_common.jl). Run with `julia -tN` selecting the
# ParallelStencil thread count. Rank 0 saves one result record to the JLD2
# path in `ARGS[3]`.
#
# T_eff: the compute kernel reads {A} and writes {Anew} => one read-write pair (A -> Anew),
# n_readwrite=1, n_readonly=0 (field B is exchanged only to match communication volume, per
# the plain CPU worker's own comment — it's not part of the compute kernel,
# so excluded here too).
#
# Unlike benchmarks/bench_halogrids_hide.jl's plain nested for-loop, this
# uses a ParallelStencil kernel launched over the requested k-plane range —
# same kernel shape as gpu/bench_halogrids_hide.jl, just backed by
# `Threads` instead of `CUDA`.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)
import MPI
using HaloGrids
include(joinpath(@__DIR__, "metrics_common.jl"))

@parallel_indices (i, j, k) function laplacian_kernel!(Anew, A, dt)
    Anew[i, j, k] = A[i, j, k] + dt * (
        A[i - 1, j, k] + A[i + 1, j, k] + A[i, j - 1, k] +
        A[i, j + 1, k] + A[i, j, k - 1] + A[i, j, k + 1] - 6 * A[i, j, k])
    return nothing
end

function launch_stencil!(Anew, A, dt, N, kstart, kstop)
    nk = kstop - kstart + 1
    nk <= 0 && return nothing
    @parallel (2:N-1, 2:N-1, kstart:kstop) laplacian_kernel!(Anew, A, dt)
    return nothing
end

# One overlapped diffusion step: safe k-planes launched while dimension 3's
# exchange is outstanding, the two dimension-3-adjacent planes launched after
# finish! — same split as benchmarks/bench_halogrids_hide.jl (plain CPU) and
# examples/hide_communication.jl.
function step_overlapped!(g, fA, fB, Anew, A, dt, N)
    handle = halo_exchange_start!(g, fA, fB; dims=(1, 2, 3))
    launch_stencil!(Anew, A, dt, N, 3, N - 2)
    halo_exchange_finish!(handle)
    launch_stencil!(Anew, A, dt, N, 2, 2)
    launch_stencil!(Anew, A, dt, N, N - 1, N - 1)
    return nothing
end

# Reference: the straightforward blocking step — used once, before timing,
# to prove the overlapped split above computes the same answer.
function step_blocking!(g, fA, fB, Anew, A, dt, N)
    update_halo!(g, fA, fB)
    launch_stencil!(Anew, A, dt, N, 2, N - 1)
    return nothing
end

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    output = ARGS[3]
    warmup = max(5, niter ÷ 10)
    dt     = 0.1

    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    t  = topology((0, 0, 0); comm=comm, periods=(true, true, true))
    g  = grid(t, (N, N, N))
    A  = fill(Float64(rank), N, N, N)
    B  = fill(Float64(rank) + 0.5, N, N, N)
    fA = field(A; halowidths=(1, 1, 1))
    fB = field(B; halowidths=(1, 1, 1))
    Anew = similar(A)
    rank == 0 && println(
        "grid: local=($N, $N, $N) global=($(nx_g(g, fA)), $(ny_g(g, fA)), $(nz_g(g, fA))) nprocs=$nprocs")

    # Correctness gate, once, before any timing: overlapped split must match
    # the blocking reference exactly.
    Aref, Bref = copy(A), copy(B)
    fAref = field(Aref; halowidths=(1, 1, 1))
    fBref = field(Bref; halowidths=(1, 1, 1))
    Anew_ref = similar(A)
    step_blocking!(g, fAref, fBref, Anew_ref, Aref, dt, N)
    step_overlapped!(g, fA, fB, Anew, A, dt, N)
    @assert Anew[2:end-1, 2:end-1, 2:end-1] == Anew_ref[2:end-1, 2:end-1, 2:end-1]

    for _ in 1:warmup
        step_overlapped!(g, fA, fB, Anew, A, dt, N)
    end
    MPI.Barrier(comm)
    GC.gc()

    stats = @timed for _ in 1:niter
        step_overlapped!(g, fA, fB, Anew, A, dt, N)
    end

    gpu_bytes = @allocated for _ in 1:niter
        step_overlapped!(g, fA, fB, Anew, A, dt, N)
    end

    numel = N * N * N
    teff = t_eff_gbs(1, 0, numel, sizeof(Float64), stats.time / niter)
    peakbw = rank == 0 ? measure_peak_bandwidth_gbs() : 0.0

    if rank == 0
        save_result(output, "HaloGrids", nprocs, N, niter, stats, gpu_bytes, teff, peakbw)
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
