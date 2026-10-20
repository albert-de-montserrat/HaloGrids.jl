# Correctness tests for the MPI/HDF5 file I/O extension (src/HDF5Ext/,
# ext/HaloGridsHDF5Ext.jl).
#
# Fill strategy: every rank fills its *entire* local array (halo cells
# included) from `iglobal`, so a cell's value depends only on its global
# position, not on which rank computed it. This makes save_hdf5's
# interior-only payload (halo cells that duplicate a neighbor's data are
# deliberately never written, see shared.jl's local_interior_range) provably
# correct in one round trip: after load_hdf5! into a zero-initialized field,
# every cell that *was* written must already match the original exactly, and
# every cell that wasn't (a real halo cell) is a stale zero until
# update_halo! repopulates it from the now-loaded neighbor data — at which
# point the two arrays must agree everywhere, including at non-periodic
# domain edges (which are genuine, uniquely-owned data, not halo, and so were
# already loaded correctly and untouched by the exchange).
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids
using HDF5

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

ext = Base.get_extension(HaloGrids, :HaloGridsHDF5Ext)
@test ext !== nothing

function cleanup!(path::AbstractString)
    MPI.Barrier(comm)
    rank == 0 && isfile(path) && rm(path; force = true)
    MPI.Barrier(comm)
end

global_fill_value(grid::Grid{N}, field::Field{T,N}, ci::CartesianIndex{N}) where {T,N} =
    sum(HaloGrids.iglobal(grid, field, d, ci[d]) * 1000.0^(d - 1) for d in 1:N)

function fill_globally!(grid::Grid, field::Field)
    for ci in CartesianIndices(field.data)
        field.data[ci] = eltype(field.data)(global_fill_value(grid, field, ci))
    end
    return field
end

# Whether *this* rank has a real neighbor (possibly itself, via periodic
# self-wrap when a periodic dimension has only one rank) on either side of
# any dimension — i.e. whether save_hdf5 trims anything at all for this rank.
any_real_neighbor(topo::Topology{N}) where {N} =
    any(d -> has_neighbor(topo.neighbors[d][1]) || has_neighbor(topo.neighbors[d][2]), 1:N)

@testset "HDF5Metadata reflects grid/field values" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f = field(zeros(6, 6); halowidths = (1, 1))
    meta = ext.metadata_from_grid(g, f, "x")
    @test meta.schema_version == ext.HDF5_IO_SCHEMA_VERSION
    @test meta.topology_dims == Tuple(topo.dims)
    @test meta.nlocal == Tuple(g.nlocal)
    @test meta.halowidths == Tuple(f.halowidths)
    @test meta.dtype_name == string(Float64)
end

@testset "single-field save_hdf5/load_hdf5! + update_halo! round trip (N=$N, periods=$periods)" for N in (2, 3),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N))

    topo = topology(ntuple(_ -> 0, N); comm = comm, periods = periods)
    nlocal = ntuple(d -> d == 1 ? 8 : 6, N)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    g = grid(topo, nlocal)

    f = field(zeros(nlocal...); halowidths = hw)
    fill_globally!(g, f)

    path = joinpath(tempdir(), "halogrids_hdf5_single_N$(N)_p$(join(Int.(periods)))_n$(nprocs).h5")
    cleanup!(path)

    save_hdf5(g, f, path)
    @test !isfile(path * ".tmp")   # mode="w" writes to .tmp, then renames

    loaded = field(zeros(nlocal...); halowidths = hw)
    load_hdf5!(g, loaded, path)

    if any_real_neighbor(topo)
        @test loaded.data != f.data
    end

    update_halo!(g, loaded)
    @test loaded.data == f.data

    cleanup!(path)
end

@testset "multi-field tuple save_hdf5/load_hdf5! with explicit names and mixed dtypes (N=3)" begin
    topo = topology((0, 0, 0); comm = comm, periods = (true, false, true))
    g = grid(topo, (6, 5, 4))

    fu = field(zeros(Float64, 6, 5, 4); halowidths = (1, 1, 1))
    fv = field(zeros(Float32, 6, 5, 4); halowidths = (2, 1, 1))
    fill_globally!(g, fu)
    fill_globally!(g, fv)

    path = joinpath(tempdir(), "halogrids_hdf5_multi_n$(nprocs).h5")
    cleanup!(path)

    save_hdf5(g, (fu, fv), path; names = ("u", "v"))

    gu = field(zeros(Float64, 6, 5, 4); halowidths = (1, 1, 1))
    gv = field(zeros(Float32, 6, 5, 4); halowidths = (2, 1, 1))
    load_hdf5!(g, (gu, gv), path; names = ("u", "v"))
    update_halo!(g, gu, gv)

    @test gu.data == fu.data
    @test gv.data == fv.data

    cleanup!(path)
end

@testset "multi-field save_hdf5/load_hdf5! fall back to positional names when omitted" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f1 = field(fill(1.0, 6, 6); halowidths = (1, 1))
    f2 = field(fill(2.0, 6, 6); halowidths = (1, 1))

    path = joinpath(tempdir(), "halogrids_hdf5_default_names_n$(nprocs).h5")
    cleanup!(path)

    save_hdf5(g, (f1, f2), path)

    g1 = field(zeros(6, 6); halowidths = (1, 1))
    g2 = field(zeros(6, 6); halowidths = (1, 1))
    load_hdf5!(g, (g1, g2), path)
    update_halo!(g, g1, g2)

    @test g1.data == f1.data
    @test g2.data == f2.data

    cleanup!(path)
end

@testset "validation and safety" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f = field(zeros(6, 6); halowidths = (1, 1))
    fill_globally!(g, f)

    path = joinpath(tempdir(), "halogrids_hdf5_validation_n$(nprocs).h5")
    cleanup!(path)
    save_hdf5(g, f, path; name = "x")

    @testset "halowidths mismatch on load" begin
        bad = field(zeros(6, 6); halowidths = (2, 2))
        @test_throws Exception load_hdf5!(g, bad, path; name = "x")
    end

    @testset "nlocal mismatch on load" begin
        g2 = grid(topo, (8, 8))
        bad = field(zeros(8, 8); halowidths = (1, 1))
        @test_throws Exception load_hdf5!(g2, bad, path; name = "x")
    end

    @testset "unknown field name" begin
        other = field(zeros(6, 6); halowidths = (1, 1))
        @test_throws Exception load_hdf5!(g, other, path; name = "does-not-exist")
    end

    @testset "duplicate field names rejected before any file is touched" begin
        f1 = field(zeros(6, 6); halowidths = (1, 1))
        f2 = field(zeros(6, 6); halowidths = (1, 1))
        dup_path = joinpath(tempdir(), "halogrids_hdf5_dupname_n$(nprocs).h5")
        @test_throws Exception save_hdf5(g, (f1, f2), dup_path; names = ("dup", "dup"))
        @test !isfile(dup_path)
    end

    @testset "re-writing an existing dataset name without a fresh file errors" begin
        @test_throws Exception save_hdf5(g, f, path; name = "x", mode = "r+")
    end

    @testset "non-CPU-strided field data is rejected before touching the file" begin
        bad = field(falses(6, 6); halowidths = (1, 1))
        bad_path = joinpath(tempdir(), "halogrids_hdf5_badtype_n$(nprocs).h5")
        @test_throws Exception save_hdf5(g, bad, bad_path)
        @test !isfile(bad_path)
    end

    cleanup!(path)
end

@testset "different-rank restart (N=$N, periods=$periods, counts=$counts)" for N in (2, 3),
    periods in (ntuple(_ -> true, N), ntuple(_ -> false, N), ntuple(d -> isodd(d), N)),
    counts in unique(((1, nprocs), (nprocs, 1), (nprocs, nprocs),
                      (min(4, nprocs), min(2, nprocs)), (min(4, nprocs), nprocs)))

    path = joinpath(tempdir(), "halogrids_restart_N$(N)_n$(nprocs).h5")
    cleanup!(path)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    # Fixed global domain, divisible by every rank count in the test matrix.
    core = 16
    writer_count, reader_count = counts
    writer_comm = MPI.Comm_split(comm, rank < writer_count ? 0 : nothing, rank)
    if rank < writer_count
        dims = ntuple(d -> d == 1 ? writer_count : 1, N)
        t = topology(dims; comm = writer_comm, periods = periods)
        nl = ntuple(d -> core ÷ dims[d] + 2 * hw[d], N)
        g = grid(t, nl)
        fields = (field(zeros(Float64, nl); halowidths = hw),
                  field(zeros(Int64, nl); halowidths = hw))
        foreach(f -> fill_globally!(g, f), fields)
        save_hdf5(g, fields, path; names = ("u", "v"), group = "state")
    end
    MPI.Barrier(comm)

    reader_comm = MPI.Comm_split(comm, rank < reader_count ? 0 : nothing, rank)
    if rank < reader_count
        # Change process-grid shape even when the rank count stays the same.
        dims = ntuple(d -> d == 2 ? reader_count : 1, N)
        t = topology(dims; comm = reader_comm, periods = periods)
        nl = ntuple(d -> core ÷ dims[d] + 2 * hw[d], N)
        g = grid(t, nl)
        fields = (field(fill(-1.0, nl); halowidths = hw),
                  field(fill(Int64(-1), nl); halowidths = hw))
        if nprocs > 1
            @test_throws ErrorException load_hdf5!(g, fields, path;
                                                  names = ("u", "v"), group = "state")
        end
        @test load_hdf5!(g, fields, path; names = ("u", "v"), group = "state",
                         repartition = true) === fields
        for f in fields
            owned, _ = ext.local_and_global_ranges(g, f)
            for ci in CartesianIndices(f.data)
                expected = all(d -> ci[d] in owned[d], 1:N) ? global_fill_value(g, f, ci) : -1
                @test f.data[ci] == expected
            end
        end
        update_halo!(g, fields...)
        for f in fields, ci in CartesianIndices(f.data)
            @test f.data[ci] == global_fill_value(g, f, ci)
        end

        # Global-size equality alone cannot detect changing boundary conventions.
        changed_periods = ntuple(d -> !periods[d], N)
        global_shape = ntuple(d -> core + 2 * hw[d] * !periods[d], N)
        if all(d -> (global_shape[d] - 2 * hw[d] * !changed_periods[d]) % dims[d] == 0, 1:N)
            bad_t = topology(dims; comm = reader_comm, periods = changed_periods)
            bad_nl = ntuple(d -> (global_shape[d] - 2 * hw[d] * !changed_periods[d]) ÷ dims[d] + 2 * hw[d], N)
            bad_g = grid(bad_t, bad_nl)
            bad_f = field(fill(-1.0, bad_nl); halowidths = hw)
            @test_throws ErrorException load_hdf5!(bad_g, bad_f, path; name = "u",
                                                  group = "state", repartition = true)
            @test all(==(-1.0), bad_f.data)
        end
        bad_f = field(zeros(Float32, nl); halowidths = hw)
        @test_throws ErrorException load_hdf5!(g, bad_f, path; name = "u",
                                              group = "state", repartition = true)
        bad_f = field(zeros(nl); halowidths = ntuple(_ -> 1, N))
        @test_throws ErrorException load_hdf5!(g, bad_f, path; name = "u",
                                              group = "state", repartition = true)
        bad_g = grid(t, ntuple(d -> nl[d] + 1, N))
        bad_f = field(zeros(bad_g.nlocal); halowidths = hw)
        @test_throws ErrorException load_hdf5!(bad_g, bad_f, path; name = "u",
                                              group = "state", repartition = true)
        single = field(zeros(nl); halowidths = hw)
        @test load_hdf5!(g, single, path; name = "u", group = "state", repartition = true) === single
    end
    cleanup!(path)
end

MPI.Barrier(comm)
rank == 0 && println("test_hdf5_io.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
