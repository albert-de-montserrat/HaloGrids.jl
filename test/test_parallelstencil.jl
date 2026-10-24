using Test
using MPI
using HaloGrids
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)
# The public macro runs blocking on CPU backends; the internal split macro
# keeps the overlap schedule covered on Threads.
const PSExt = Base.get_extension(HaloGrids, :HaloGridsParallelStencilExt)

@parallel_indices (i, j, k) function increment_owned!(A)
    A[i + 1, j + 1, k + 1] += i + 10j + 100k
    return nothing
end

# ParallelStencil keeps initialization per module, so exercise its 2D backend
# separately from the 3D kernels above.
module TwoDimensionalTests
using Test, MPI, HaloGrids, ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
const PSExt = Base.get_extension(HaloGrids, :HaloGridsParallelStencilExt)

@parallel_indices (i, j) function increment_owned!(A)
    A[i + 1, j + 1] += i + 10j
    return nothing
end

function run_tests()
    @testset "2D ParallelStencil ranges" begin
        t = topology((0, 0); periods=(true, true))
        for (n, b) in ((6, 1), (6, 2), (32, 2), (32, 16))
            g = grid(t, (n, n))
            f = field(fill(Float64(t.rank), n, n); halowidths=(1, 1))
            reference = field(copy(f.data); halowidths=(1, 1))
            A, B = f.data, reference.data
            ranges = (1:n-2, 1:n-2)
            for _ in 1:2
                @parallel ranges increment_owned!(B)
                @synchronize
                update_halo!(g, reference)
                PSExt.@hide_halo_communication_split g f boundary_width=(b, b) begin
                    @parallel ranges increment_owned!(A)
                end
                @synchronize
                @test A == B
                @test !g.excache.pending
            end
            @parallel ranges increment_owned!(B)
            @synchronize
            update_halo!(g, reference)
            PSExt.@hide_halo_communication_split g f begin
                @parallel ranges increment_owned!(A)
            end
            @synchronize
            @test A == B
        end
    end
    # The sent strip is `overlaps - halowidths` deep, so the default width must
    # cover it; a width of `halowidths` alone sends stale cells.
    @testset "2D default width with overlaps > 2*halowidths" begin
        t = topology((0, 0); periods=(true, true))
        n, ol = 10, 4
        g = grid(t, (n, n))
        f = field(fill(Float64(t.rank), n, n); overlaps=(ol, ol))
        reference = field(copy(f.data); overlaps=(ol, ol))
        ranges = (1:n-2, 1:n-2)
        ext = Base.get_extension(HaloGrids, :HaloGridsParallelStencilExt)
        @test ext.boundary_width(f) == (ol - 1, ol - 1)
        @test ext.boundary_width(zeros(n, n)) == (1, 1)
        for _ in 1:2
            @parallel ranges increment_owned!(reference.data)
            @synchronize
            update_halo!(g, reference)
            PSExt.@hide_halo_communication_split g f begin
                @parallel ranges increment_owned!(f.data)
            end
            @synchronize
            @test f.data == reference.data
        end
    end
end
end

MPI.Init()
@testset "ParallelStencil overlap and blocking fallback" begin
    t = topology((0, 0, 0); periods=(true, true, true))
    ext = Base.get_extension(HaloGrids, :HaloGridsParallelStencilExt)
    @test_throws ArgumentError ext.can_overlap((0, 1, 1), (1:4, 1:4, 1:4))
    @test_throws ArgumentError ext.can_overlap((1, 1), (1:4, 1:4, 1:4))
    @test_throws ArgumentError ext.can_overlap((1, 1, 1), (2:5, 1:4, 1:4))
    for (n, widths) in ((6, (1, 1, 1)), (6, (2, 1, 1)), (6, (3, 1, 1)), (32, (16, 2, 2)))
        g = grid(t, (n, n, n))
        f = field(fill(Float64(t.rank), n, n, n); halowidths=(1, 1, 1))
        reference = field(copy(f.data); halowidths=(1, 1, 1))
        A = f.data
        B = reference.data
        ranges = (1:n-2, 1:n-2, 1:n-2)
        boundary_calls = Ref(0)
        for _ in 1:2
            @parallel ranges increment_owned!(B)
            B[1, :, :] .= -7
            @synchronize
            update_halo!(g, reference)
            PSExt.@hide_halo_communication_split g f boundary_width=widths begin
                @parallel ranges increment_owned!(A)
                A[1, :, :] .= -7
                boundary_calls[] += 1
            end
            @synchronize
            @test A == B
            @test !g.excache.pending
        end
        @test boundary_calls[] == 2
        @parallel ranges increment_owned!(B)
        @synchronize
        update_halo!(g, reference)
        PSExt.@hide_halo_communication_split g f begin
            @parallel ranges increment_owned!(A)
        end
        @synchronize
        @test A == B
    end
end

@testset "public macro is blocking on a CPU backend" begin
    @test !PSExt.split_backend(@__MODULE__)
    t = topology((0, 0, 0); periods=(true, true, true))
    n = 32
    g = grid(t, (n, n, n))
    f = field(fill(Float64(t.rank), n, n, n); halowidths=(1, 1, 1))
    reference = field(copy(f.data); halowidths=(1, 1, 1))
    ranges = (1:n-2, 1:n-2, 1:n-2)
    # Boundary statements run after the full kernel when blocking, but before
    # the inner box when split, so they see whether the center was updated.
    c = n ÷ 2
    for _ in 1:2
        @parallel ranges increment_owned!(reference.data)
        @synchronize
        update_halo!(g, reference)
        center = Ref(NaN)
        @hide_halo_communication g f boundary_width=(2, 2, 2) begin
            @parallel ranges increment_owned!(f.data)
            center[] = f.data[c, c, c]
        end
        @synchronize
        @test center[] == reference.data[c, c, c]
        @test f.data == reference.data
        @test !g.excache.pending
    end
    # Widths and ranges are still validated on the blocking path.
    @test_throws ArgumentError @hide_halo_communication g f boundary_width=(0, 1, 1) begin
        @parallel ranges increment_owned!(f.data)
    end
end
TwoDimensionalTests.run_tests()
MPI.Finalize()
