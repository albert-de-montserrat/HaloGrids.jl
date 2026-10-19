push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

@testset "gather! (N=$N)" for N in (2, 3)
    t = topology(ntuple(_ -> 0, N); comm=comm)
    nlocal = ntuple(_ -> 3, N)
    A = fill(Float64(rank), nlocal)
    A_global = rank == 0 ? zeros(Float64, nlocal .* t.dims) : nothing

    gather!(A, A_global, t)

    if rank == 0
        ok = true
        for I in CartesianIndices(A_global)
            coords = ntuple(d -> (Tuple(I)[d] - 1) ÷ nlocal[d], N)
            expected = MPI.Cart_rank(t.comm, collect(coords))
            ok &= (A_global[I] == expected)
        end
        @test ok
    end
end

@testset "gather! via Grid convenience method" begin
    t = topology((0, 0); comm=comm)
    g = grid(t, (3, 3))
    A = fill(Float64(rank), g.nlocal)
    A_global = rank == 0 ? zeros(Float64, g.nlocal .* t.dims) : nothing
    gather!(A, A_global, g)
    if rank == 0
        @test sum(A_global) == sum(0:nprocs-1) * 9
    end
end

@testset "gather! validates on every rank" begin
    t = topology((0, 0); comm=comm)
    # The last rank passes a smaller A: every rank must error, none may hang.
    A = rank == nprocs - 1 && nprocs > 1 ? zeros(2, 3) : zeros(3, 3)
    A_global = rank == 0 ? zeros(3 .* t.dims) : nothing
    if nprocs > 1
        @test_throws ErrorException gather!(A, A_global, t)
    end
    # A root-only argument error reaches every rank too.
    bad_global = rank == 0 ? zeros(1, 1) : nothing
    @test_throws ErrorException gather!(zeros(3, 3), bad_global, t)
end

@testset "gather! with ndims(A_global) < ndims(topology)" begin
    t = topology((nprocs, 1); comm=comm)
    A = fill(Float64(rank), 3)
    A_global = rank == 0 ? zeros(3nprocs) : nothing
    gather!(A, A_global, t)
    rank == 0 && @test A_global == repeat(0.0:nprocs-1; inner=3)
    if nprocs > 1
        t2 = topology((1, nprocs); comm=comm)   # trailing topology dim > 1: rejected
        @test_throws ErrorException gather!(A, rank == 0 ? zeros(3) : nothing, t2)
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_gather.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
