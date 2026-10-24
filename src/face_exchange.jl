# Face slabs exclude halos in other Cartesian grid dimensions. Trailing
# component dimensions are always retained in full, independent of their
# inert Field halo-width placeholders.
function face_range(f::Field{T,NF}, ::Val{N}, dim, side, sending::Bool) where {T,NF,N}
    hw = f.halowidths[dim]
    ol = f.overlaps[dim]
    len = size(f.data, dim)
    start = sending ? (side == 1 ? ol - hw + 1 : len - ol + 1) :
        (side == 1 ? 1 : len - hw + 1)
    return ntuple(NF) do d
        d == dim ? (start:start+hw-1) :
            (d <= N ? (f.halowidths[d]+1:size(f.data, d)-f.halowidths[d]) : (1:size(f.data, d)))
    end
end

function face_length(f::Field{T,NF}, ::Val{N}, dim) where {T,NF,N}
    len = f.halowidths[dim]
    for d in 1:NF
        d == dim && continue
        len *= d <= N ? size(f.data, d) - 2f.halowidths[d] : size(f.data, d)
    end
    return len
end

# Separate tag space from full exchange. Encode physical dimension and send
# direction so posting all dimensions together is unambiguous, including
# periodic two-rank dimensions where both sides name the same peer.
face_tag(dim, side) = 100 + 2(dim - 1) + side

@inline pack_faces!(cache, ::Tuple{}, grid_dims, dim, lo, hi, i) = nothing
@inline function pack_faces!(cache, fields::Tuple, grid_dims, dim, lo, hi, i)
    f = fields[1]
    len = face_length(f, grid_dims, dim)
    has_neighbor(lo) && pack!(pack_target(cache, i, 1, eltype(f), len, f.data),
        f.data, face_range(f, grid_dims, dim, 1, true), cache.threaded_pack)
    has_neighbor(hi) && pack!(pack_target(cache, i, 2, eltype(f), len, f.data),
        f.data, face_range(f, grid_dims, dim, 2, true), cache.threaded_pack)
    pack_faces!(cache, Base.tail(fields), grid_dims, dim, lo, hi, i + 1)
    return nothing
end

@inline copy_self_faces!(cache, ::Tuple{}, grid_dims, dim, i) = nothing
@inline function copy_self_faces!(cache, fields::Tuple, grid_dims, dim, i)
    f = fields[1]
    len = face_length(f, grid_dims, dim)
    copyto!(unpack_source(cache, i, 2, eltype(f), len, f.data),
        pack_target(cache, i, 1, eltype(f), len, f.data))
    copyto!(unpack_source(cache, i, 1, eltype(f), len, f.data),
        pack_target(cache, i, 2, eltype(f), len, f.data))
    copy_self_faces!(cache, Base.tail(fields), grid_dims, dim, i + 1)
    return nothing
end

@inline post_face_recvs!(cache, ::Tuple{}, grid_dims, dim, lo, hi, comm, i) = nothing
@inline function post_face_recvs!(cache, fields::Tuple, grid_dims, dim, lo, hi, comm, i)
    f = fields[1]
    len = face_length(f, grid_dims, dim)
    has_neighbor(lo) && MPI.Irecv!(mpi_recvbuf(cache, i, 1, eltype(f), len, f.data),
        comm, cache.reqs[i, 1, 1]; source=lo, tag=face_tag(dim, 2))
    has_neighbor(hi) && MPI.Irecv!(mpi_recvbuf(cache, i, 2, eltype(f), len, f.data),
        comm, cache.reqs[i, 2, 1]; source=hi, tag=face_tag(dim, 1))
    post_face_recvs!(cache, Base.tail(fields), grid_dims, dim, lo, hi, comm, i + 1)
    return nothing
end

@inline post_face_sends!(cache, ::Tuple{}, grid_dims, dim, lo, hi, comm, i) = nothing
@inline function post_face_sends!(cache, fields::Tuple, grid_dims, dim, lo, hi, comm, i)
    f = fields[1]
    len = face_length(f, grid_dims, dim)
    if has_neighbor(lo)
        stage_send!(cache, i, 1, eltype(f), len, f.data)
        MPI.Isend(mpi_sendbuf(cache, i, 1, eltype(f), len, f.data),
            comm, cache.reqs[i, 1, 2]; dest=lo, tag=face_tag(dim, 1))
    end
    if has_neighbor(hi)
        stage_send!(cache, i, 2, eltype(f), len, f.data)
        MPI.Isend(mpi_sendbuf(cache, i, 2, eltype(f), len, f.data),
            comm, cache.reqs[i, 2, 2]; dest=hi, tag=face_tag(dim, 2))
    end
    post_face_sends!(cache, Base.tail(fields), grid_dims, dim, lo, hi, comm, i + 1)
    return nothing
end

@inline start_face_dims!(g::Grid, fields::Tuple, ::Tuple{}) = nothing
@inline function start_face_dims!(g::Grid{N}, fields::Tuple, dims::Tuple) where {N}
    dim = dims[1]
    lo, hi = g.topology.neighbors[dim]
    rank = g.topology.rank
    (lo == rank) == (hi == rank) || error("inconsistent self-periodic neighbors in dimension $dim")
    i = (dim - 1) * length(fields) + 1
    pack_faces!(g.excache, fields, Val(N), dim, lo, hi, i)
    if lo == rank
        copy_self_faces!(g.excache, fields, Val(N), dim, i)
    else
        prepare_mpi_fields!(g.excache, fields)
        post_face_recvs!(g.excache, fields, Val(N), dim, lo, hi, g.topology.comm, i)
        post_face_sends!(g.excache, fields, Val(N), dim, lo, hi, g.topology.comm, i)
    end
    start_face_dims!(g, fields, Base.tail(dims))
    return nothing
end

@inline finish_face_fields!(cache, ::Tuple{}, grid_dims, dim, lo, hi, self_periodic, i) = nothing
@inline function finish_face_fields!(cache, fields::Tuple, grid_dims, dim, lo, hi, self_periodic, i)
    f = fields[1]
    len = face_length(f, grid_dims, dim)
    if has_neighbor(lo)
        if !self_periodic
            MPI.Wait(cache.reqs[i, 1, 1])
            stage_recv!(cache, i, 1, eltype(f), len, f.data)
        end
        unpack!(f.data, unpack_source(cache, i, 1, eltype(f), len, f.data),
            face_range(f, grid_dims, dim, 1, false), cache.threaded_pack)
    end
    if has_neighbor(hi)
        if !self_periodic
            MPI.Wait(cache.reqs[i, 2, 1])
            stage_recv!(cache, i, 2, eltype(f), len, f.data)
        end
        unpack!(f.data, unpack_source(cache, i, 2, eltype(f), len, f.data),
            face_range(f, grid_dims, dim, 2, false), cache.threaded_pack)
    end
    finish_face_fields!(cache, Base.tail(fields), grid_dims, dim, lo, hi, self_periodic, i + 1)
    return nothing
end

@inline finish_face_dims!(g::Grid, fields::Tuple, ::Tuple{}) = nothing
@inline function finish_face_dims!(g::Grid{N}, fields::Tuple, dims::Tuple) where {N}
    dim = dims[1]
    lo, hi = g.topology.neighbors[dim]
    self_periodic = lo == g.topology.rank
    i = (dim - 1) * length(fields) + 1
    finish_face_fields!(g.excache, fields, Val(N), dim, lo, hi, self_periodic, i)
    self_periodic || wait_sends_r!(g.excache, fields, lo, hi, i)
    finish_face_dims!(g, fields, Base.tail(dims))
    return nothing
end

@inline contains_dim(dim, ::Tuple{}) = false
@inline contains_dim(dim, dims::Tuple) = dim == dims[1] || contains_dim(dim, Base.tail(dims))
@inline unique_face_dims(::Tuple{}) = true
@inline unique_face_dims(dims::Tuple) =
    !contains_dim(dims[1], Base.tail(dims)) && unique_face_dims(Base.tail(dims))

function _halo_exchange_start!(g::Grid{N}, fields::Tuple, dims::Tuple, ::Val{:faces}) where {N}
    g.excache.pending && error("a halo exchange is already in flight on this grid; finish it first.")
    check_fields(g, fields)
    check_dims(Val(N), dims)
    unique_face_dims(dims) || error("face exchange dims must not repeat; got dims=$dims.")
    check_widths(g.topology, fields, dims)
    ensure_capacity!(g.excache, N * length(fields))
    epoch = begin_exchange!(g.excache)
    start_face_dims!(g, fields, dims)
    return ExchangeHandle{N,typeof(fields),typeof(dims)}(g, fields, 0, false, dims, epoch)
end

function halo_exchange_finish!(handle::ExchangeHandle{N,F,D}) where {N,F,D<:Tuple}
    check_handle(handle)
    finish_face_dims!(handle.grid, handle.fields, handle.dims)
    handle.grid.excache.pending = false
    return nothing
end
