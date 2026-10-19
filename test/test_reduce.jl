push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

@testset "global_allreduce / global_sum / global_max / global_min" begin
    t = topology((0, 0); comm=comm)

    @test global_allreduce(t, Float64(rank), +) == Float64(sum(0:nprocs-1))
    @test global_sum(t, Float64(rank)) == Float64(sum(0:nprocs-1))
    @test global_max(t, Float64(rank)) == Float64(nprocs - 1)
    @test global_min(t, Float64(rank)) == 0.0

    # Every rank must see the same, fully-reduced result, not just rank 0.
    @test global_sum(t, 1) == nprocs
end

@testset "global_reduce (root only)" begin
    t = topology((0, 0); comm=comm)

    result = global_reduce(t, rank, +; root=0)
    if rank == 0
        @test result == sum(0:nprocs-1)
    end

    result_max = global_reduce(t, rank, max)  # default root=0
    if rank == 0
        @test result_max == nprocs - 1
    end
end

@testset "Grid convenience methods match Topology methods" begin
    t = topology((0, 0); comm=comm)
    g = grid(t, (4, 4))

    @test global_sum(g, Float64(rank)) == global_sum(t, Float64(rank))
    @test global_max(g, Float64(rank)) == global_max(t, Float64(rank))
    @test global_min(g, Float64(rank)) == global_min(t, Float64(rank))
    @test global_allreduce(g, rank, +) == global_allreduce(t, rank, +)

    r_grid = global_reduce(g, rank, +; root=0)
    r_topo = global_reduce(t, rank, +; root=0)
    rank == 0 && @test r_grid == r_topo
end

@testset "reduces a local array-derived scalar, not the array itself" begin
    # The documented usage pattern: reduce your own local data with plain
    # Julia first (excluding halos however is appropriate), then Allreduce
    # the resulting scalar. Every rank's local array here is filled with its
    # own rank, so summing every rank's local sum must equal nprocs*(nprocs-1)/2
    # times the number of local elements.
    t = topology((0, 0); comm=comm)
    nlocal = (5, 5)
    A = fill(Float64(rank), nlocal)

    local_sum = sum(A)
    total = global_sum(t, local_sum)
    @test total == Float64(sum(0:nprocs-1)) * length(A)

    local_max = maximum(A)
    @test global_max(t, local_max) == Float64(nprocs - 1)
end

@testset "global_norm" begin
    t = topology((0, 0); comm=comm)
    nlocal = (4, 3)
    A = fill(Float64(rank + 1), nlocal)  # rank 0 -> 1.0, rank 1 -> 2.0, ...

    # L2 (default): sqrt(sum over every rank of length(A) * (rank+1)^2).
    expected_l2 = sqrt(sum((r + 1)^2 for r in 0:nprocs-1) * length(A))
    @test global_norm(t, A) ≈ expected_l2

    # L1: sum over every rank of length(A) * (rank+1).
    expected_l1 = sum(r + 1 for r in 0:nprocs-1) * length(A)
    @test global_norm(t, A, 1) ≈ expected_l1

    # Linf: the largest |value| across every rank, i.e. nprocs.
    @test global_norm(t, A, Inf) == Float64(nprocs)

    # A plain scalar works the same way as a single-element array.
    expected_scalar_l2 = sqrt(sum((r + 1)^2 for r in 0:nprocs-1))
    @test global_norm(t, Float64(rank + 1)) ≈ expected_scalar_l2

    # Grid convenience method matches the Topology method.
    g = grid(t, nlocal)
    @test global_norm(g, A) ≈ global_norm(t, A)
    @test global_norm(g, A, Inf) ≈ global_norm(t, A, Inf)

    # Sanity check on the docs' warning: combining each rank's own local norm
    # directly (rather than combining sums of powers, then rooting once) is
    # NOT the same as the true global norm for p != 1, so the two approaches
    # had better disagree here (nprocs > 1) — this pins down that the wrong
    # approach really is wrong, not just theoretically so.
    if nprocs > 1
        wrong_l2 = global_sum(t, sqrt(sum(abs2, A)))
        @test wrong_l2 != global_norm(t, A)
    end
end

@testset "NaN propagates through global_max/global_min" begin
    t = topology((0, 0); comm=comm)
    val = rank == nprocs - 1 ? NaN : 1.0
    @test isnan(global_max(t, val))
    @test isnan(global_min(t, val))
    @test isnan(global_norm(t, [val, 3.0], Inf))
    @test global_max(t, Float32(rank)) === Float32(nprocs - 1)
    @test global_min(t, Float64(rank)) === 0.0
    @test global_max(t, rank) == nprocs - 1
end

@testset "global_norm matches LinearAlgebra.norm conventions" begin
    t = topology((0, 0); comm=comm)
    @test global_norm(t, Int8[100, 0]) ≈ 100 * sqrt(nprocs)    # no Int8 overflow
    @test global_norm(t, Int32[50_000]) ≈ 50_000 * sqrt(nprocs)
    @test global_norm(t, [-3.0, Float64(rank + 1)], -Inf) == 1.0
    @test_throws ArgumentError global_norm(t, [1.0], 0)
    @test_throws ArgumentError global_norm(t, [1.0], -1)
end

MPI.Barrier(comm)
rank == 0 && println("test_reduce.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
