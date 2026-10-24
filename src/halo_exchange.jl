export halo_exchange_start!, halo_exchange_finish!, update_halo!, ExchangeHandle,
    @hide_halo_communication

"""
    @hide_halo_communication grid fields... [boundary_width=widths] begin
        @parallel ranges kernel!(...)
        boundary_updates...
    end

ParallelStencil integration for overlapping a ranged computation with halo
exchange. ParallelStencil must be loaded so that the optional extension is
active.

The first nonempty statement must be exactly one explicitly ranged
`@parallel` call. The macro partitions `ranges` into disjoint outer slabs and
an inner box, launches the kernel over every outer slab, executes the remaining
statements as boundary updates, synchronizes ParallelStencil, starts halo
exchange for `fields`, launches the same kernel over the inner box, and finally
finishes the exchange. `halo_exchange_finish!` runs from a `finally` clause if
the inner launch throws.

By default the partition width is the componentwise maximum of the exchanged
fields' `overlaps - halowidths` (their halo widths, for default overlaps): the
depth of the strip that is sent. Override it with `boundary_width=(bx, by)` or
`boundary_width=(bx, by, bz)`. A width at least that default moves more work
before the exchange and leaves less work available for overlap without changing
the result; a smaller width sends cells before the kernel has updated them. Every width must be
positive and match the dimensionality of `ranges`. If any range length is at
most twice its width, the macro instead launches the full kernel, executes the
boundary updates, synchronizes, and performs a blocking `update_halo!`.
ParallelStencil's automatic partitioner also
requires every range to start at `1`.

The macro takes the same blocking schedule whenever the calling module
initialized ParallelStencil with a CPU backend (`Threads` or `Polyester`), as
ParallelStencil's own `@hide_communication` does. CPU kernel launches are
synchronous, and the thin outer slabs cost more than the final dimension's
exchange can hide. Widths and ranges are still validated.

The boundary statements must update only data outside the inner box and must
not depend on unfinished inner values. They run before packing and are useful
for physical boundary conditions. Halo exchange then overwrites provisional
boundary values on internal process interfaces. As with
`halo_exchange_start!`, only the final Cartesian dimension remains pending and
can overlap the inner launch; earlier dimensions finish synchronously.
"""
macro hide_halo_communication(args...)
    ext = Base.get_extension(HaloGrids, :HaloGridsParallelStencilExt)
    ext === nothing && error("@hide_halo_communication requires ParallelStencil")
    return ext.hide_halo_communication_expr(__module__, args...)
end

"""
    ExchangeHandle

Returned by [`halo_exchange_start!`](@ref); pass it to [`halo_exchange_finish!`](@ref)
to complete the exchange. Opaque — its fields are internal.

With the default `region=:full`, halo exchange is sequential across Cartesian dimensions
(dimension 2's packing must read what dimension 1's unpacking wrote, to get
corners right), `halo_exchange_start!` performs every dimension but the last
fully synchronously, and only the *last* dimension is actually left
outstanding in the handle — this is the honestly-documented scope limit of the
communication/compute overlap for full exchange. `region=:faces` excludes
transverse halo cells and leaves every requested dimension outstanding.
"""
struct ExchangeHandle{N,F<:Tuple,D<:Tuple}
    grid::Grid{N}
    fields::F
    dim::Int          # final full-exchange dimension; faces use dims instead
    self_periodic::Bool
    dims::D          # empty for full exchange; pending dimensions for faces
    epoch::Int       # grid.excache.epoch at start; finish! rejects a stale handle
end

ExchangeHandle{N,F}(grid, fields, dim, self_periodic) where {N,F} =
    ExchangeHandle{N,F,Tuple{}}(grid, fields, dim, self_periodic, (), grid.excache.epoch)

# Number of elements in the halo slab of field `f` in dimension `dim` (all other
# dimensions at full size).
function halo_length(f::Field{T,N}, dim::Integer) where {T,N}
    len = f.halowidths[dim]
    for d in 1:N
        d == dim && continue
        len *= size(f.data, d)
    end
    return len
end

# Index ranges (one UnitRange per dimension) of the strip of `f`'s local array
# sent to neighbor `n` (1=lo, 2=hi) in dimension `dim`: the interior cells
# immediately adjacent to that side's halo.
function sendrange(f::Field{T,N}, dim::Integer, n::Integer) where {T,N}
    hw = f.halowidths[dim]
    ol = f.overlaps[dim]
    nA = size(f.data, dim)
    start = n == 1 ? ol - hw + 1 : nA - ol + 1
    return ntuple(d -> d == dim ? (start:start+hw-1) : (1:size(f.data, d)), N)
end

# Index ranges of the halo strip of `f` that neighbor `n` (1=lo, 2=hi) fills in
# dimension `dim`.
function recvrange(f::Field{T,N}, dim::Integer, n::Integer) where {T,N}
    hw = f.halowidths[dim]
    nA = size(f.data, dim)
    start = n == 1 ? 1 : nA - hw + 1
    return ntuple(d -> d == dim ? (start:start+hw-1) : (1:size(f.data, d)), N)
end

pack!(buf::AbstractVector, A::AbstractArray, ranges::Tuple) = copyto!(buf, view(A, ranges...))
unpack!(A::AbstractArray, buf::AbstractVector, ranges::Tuple) = copyto!(view(A, ranges...), buf)

# Dense CPU arrays: copy the slab one dim-1 column at a time through raw
# pointers. The generic `copyto!` above indexes the reinterpreted byte buffer
# element by element, measured ~9x slower for a contiguous 256x256 Float64 slab.
pack!(buf::StridedVector{T}, A::Array{T}, ranges::Tuple) where {T} = copy_columns!(buf, A, ranges, true)
unpack!(A::Array{T}, buf::StridedVector{T}, ranges::Tuple) where {T} = copy_columns!(buf, A, ranges, false)

function copy_columns!(buf::StridedVector{T}, A::Array{T}, ranges::Tuple, topack::Bool) where {T}
    n = length(ranges[1])
    i1 = first(ranges[1])
    cols = CartesianIndices(Base.tail(ranges))
    length(buf) == n * length(cols) || throw(DimensionMismatch(
        "buffer length $(length(buf)) does not match slab length $(n * length(cols))."))
    GC.@preserve buf A begin
        pb = pointer(buf)
        li = LinearIndices(A)
        o = 0
        @inbounds for I in cols
            pa = pointer(A, li[i1, Tuple(I)...])
            if topack
                @inbounds for k in 1:n
                    unsafe_store!(pb, unsafe_load(pa, k), o + k)
                end
            else
                @inbounds for k in 1:n
                    unsafe_store!(pa, unsafe_load(pb, o + k), k)
                end
            end
            o += n
        end
    end
    return nothing
end

# Below THREADED_PACK_MIN_LEN, `Threads.@spawn` overhead (scheduling latency,
# plus a real ~500-670B/task allocation measured regardless of chunk size —
# see "Performance, memory footprint, and GC-pressure goals" in
# REDESIGN_PLAN.md) outweighs the copy it parallelizes: benchmarked as a net
# *slowdown* for a contiguous 64x64 slab (0.62x) and only break-even at 128x128
# (1.02x). Calibrated at 65536 elements so both the contiguous case (>=1.26x
# at that size and above) and the non-contiguous case (>=2.30x) are already
# clear wins by the time threading kicks in.
const THREADED_PACK_MIN_LEN = 65536

# Splits a pack/unpack copy across `Threads.@spawn` tasks. Only reached when
# the caller opted in (`threaded_pack=true` on `grid`, see [`grid`](@ref)) and
# the slab cleared `THREADED_PACK_MIN_LEN`; never the default path, so it can
# afford the allocation `Threads.@spawn` itself causes. Real OS-thread
# parallelism (distinct from `@task`/`@async`, which only cooperatively
# multitask a single OS thread and measured no benefit for this CPU-bound
# copy — the old package's `@task`-based pack/unpack pipelining was evaluated
# and rejected for exactly this reason, see "Deliberate scope trims" in
# REDESIGN_PLAN.md); requires Julia launched with `-t N>1` to do anything
# (`Threads.nthreads() == 1` by default), and packing/unpacking is the only
# part of the exchange ever threaded this way — the `MPI.Isend`/`Irecv!`/`Wait`
# calls themselves never are, since this package never calls `MPI.Init` with a
# non-default threadlevel.
function pack_threaded!(buf::AbstractVector, A::AbstractArray, ranges::Tuple)
    v = view(A, ranges...)
    n = length(v)
    nchunks = Threads.nthreads()
    chunk = cld(n, nchunks)
    @sync for c in 1:nchunks
        lo = (c - 1) * chunk + 1
        hi = min(c * chunk, n)
        lo > hi && continue
        Threads.@spawn copyto!(view(buf, lo:hi), view(v, lo:hi))
    end
    return nothing
end

function unpack_threaded!(A::AbstractArray, buf::AbstractVector, ranges::Tuple)
    v = view(A, ranges...)
    n = length(v)
    nchunks = Threads.nthreads()
    chunk = cld(n, nchunks)
    @sync for c in 1:nchunks
        lo = (c - 1) * chunk + 1
        hi = min(c * chunk, n)
        lo > hi && continue
        Threads.@spawn copyto!(view(v, lo:hi), view(buf, lo:hi))
    end
    return nothing
end

# Dispatch used at exchange call sites: `threaded` is `cache.threaded_pack`,
# read straight off the (concretely-typed) `ExchangeCache` field, so this
# branch itself costs nothing — it just picks which of the two functions
# above to call. False by construction unless the `Grid` opted in.
@inline pack!(buf::AbstractVector, A::AbstractArray, ranges::Tuple, threaded::Bool) =
    threaded && Threads.nthreads() > 1 && length(buf) >= THREADED_PACK_MIN_LEN ?
        pack_threaded!(buf, A, ranges) : pack!(buf, A, ranges)
@inline unpack!(A::AbstractArray, buf::AbstractVector, ranges::Tuple, threaded::Bool) =
    threaded && Threads.nthreads() > 1 && length(buf) >= THREADED_PACK_MIN_LEN ?
        unpack_threaded!(A, buf, ranges) : unpack!(A, buf, ranges)

# Two direction tags, not one, and picked *by direction* rather than by which
# neighbor rank happens to be on that side — see the long comment on
# `post_irecv!`/`post_isend!` below for why a single shared tag is a real bug,
# not just unnecessary caution.
const EXCHANGE_TAG_LO = 11
const EXCHANGE_TAG_HI = 22

# All of the per-field loops below recurse on the `Tuple` type via `Base.tail`
# rather than indexing `fields[i]` in a `for i in 1:n` loop. `fields` is
# heterogeneous (a different `Field{T,...}` per position), and indexing it
# with a runtime `Int` is not type-stable — the compiler can't know which
# concrete `Field` type comes back at a given `i`, so it boxes. Recursing lets
# each call specialize on the ever-shrinking `Tuple` type and fully
# unroll/inline away, at zero allocation (measured: indexed loops cost
# ~576B/call for 2 fields here vs 0B for this pattern — see "Performance,
# memory footprint, and GC-pressure goals" in REDESIGN_PLAN.md).

@inline pack_dim!(cache, ::Tuple{}, dim, lo, hi, i) = nothing
@inline function pack_dim!(cache, fields::Tuple, dim, lo, hi, i)
    f = fields[1]
    len = halo_length(f, dim)
    has_neighbor(lo) && pack!(pack_target(cache, i, 1, eltype(f), len, f.data), f.data, sendrange(f, dim, 1), cache.threaded_pack)
    has_neighbor(hi) && pack!(pack_target(cache, i, 2, eltype(f), len, f.data), f.data, sendrange(f, dim, 2), cache.threaded_pack)
    pack_dim!(cache, Base.tail(fields), dim, lo, hi, i + 1)
    return nothing
end

# No MPI call: a direct buffer copy, mirroring the old package's
# `sendrecv_halo_local` (send-to-self needs a `self` BTL in some MPI
# implementations, so this sidesteps a real portability issue, not a
# performance choice). Goes through pack_target/unpack_source rather than
# sendbuf/recvbuf so a GPU field's periodic self-copy stays entirely on
# device — no MPI is involved here, so there is nothing to stage through the
# host for.
@inline copy_self_periodic!(cache, ::Tuple{}, dim, i) = nothing
@inline function copy_self_periodic!(cache, fields::Tuple, dim, i)
    f = fields[1]
    len = halo_length(f, dim)
    copyto!(unpack_source(cache, i, 2, eltype(f), len, f.data), pack_target(cache, i, 1, eltype(f), len, f.data))
    copyto!(unpack_source(cache, i, 1, eltype(f), len, f.data), pack_target(cache, i, 2, eltype(f), len, f.data))
    copy_self_periodic!(cache, Base.tail(fields), dim, i + 1)
    return nothing
end

# Tag choice here is not incidental. When a dimension is periodic with exactly
# 2 ranks, `lo` and `hi` are the *same* physical neighbor rank — a real
# topology (e.g. an 8-rank 2D grid with MPI.Dims_create giving dims=(4,2), or
# any 2-rank periodic run). With a single shared tag (as this used to be),
# MPI's message-ordering guarantee ("non-overtaking": messages between a given
# (source,dest,tag,comm) triple are matched in the order they were sent) pairs
# messages by *post order*, not by which physical direction they represent —
# and both ranks post their slot-1 (lo-directed) send/recv before their slot-2
# (hi-directed) one, so slot 1's receive silently ends up matched with the
# neighbor's slot-1 send instead of its slot-2 send. Concretely: this rank's
# low halo (meant to receive its lo-neighbor's *high*-side interior strip)
# instead received that neighbor's *low*-side strip — confirmed by direct
# reproduction (`bench`/ad hoc script filling each cell with a position-
# dependent value; a rank-uniform fill, as the existing correctness tests use,
# cannot detect this because both interior strips hold the same value).
#
# The fix: tag messages by *direction* (low-direction vs high-direction),
# never by which slot number or which physical rank happens to be on that
# side. This rank's slot-1 send (its own low-side strip, going to its
# lo-neighbor) is always tagged `EXCHANGE_TAG_LO`; its slot-2 send (own
# high-side strip, to its hi-neighbor) is always `EXCHANGE_TAG_HI`. Since
# every rank in the topology follows the same convention, a receive posted
# for "the message coming from my direction-lo neighbor, carrying *its*
# high-direction send" can ask for `source=lo, tag=EXCHANGE_TAG_HI` and get
# an unambiguous match by tag alone, even when `lo == hi`.
@inline post_irecv!(cache, ::Tuple{}, dim, lo, hi, comm, i) = nothing
@inline function post_irecv!(cache, fields::Tuple, dim, lo, hi, comm, i)
    f = fields[1]
    len = halo_length(f, dim)
    # My low halo (slot 1) is filled by my lo-neighbor's *high*-direction send.
    has_neighbor(lo) && MPI.Irecv!(mpi_recvbuf(cache, i, 1, eltype(f), len, f.data), comm, cache.reqs[i, 1, 1]; source=lo, tag=EXCHANGE_TAG_HI)
    # My high halo (slot 2) is filled by my hi-neighbor's *low*-direction send.
    has_neighbor(hi) && MPI.Irecv!(mpi_recvbuf(cache, i, 2, eltype(f), len, f.data), comm, cache.reqs[i, 2, 1]; source=hi, tag=EXCHANGE_TAG_LO)
    post_irecv!(cache, Base.tail(fields), dim, lo, hi, comm, i + 1)
    return nothing
end

@inline post_isend!(cache, ::Tuple{}, dim, lo, hi, comm, i) = nothing
@inline function post_isend!(cache, fields::Tuple, dim, lo, hi, comm, i)
    f = fields[1]
    len = halo_length(f, dim)
    # stage_send! is a no-op for CPU fields (pack_target already wrote to the
    # MPI buffer) and for GPU-aware MPI, and a device->host copy otherwise —
    # see backend.jl.
    # Slot 1 = my own low-side strip, heading to my lo-neighbor: a low-direction send.
    if has_neighbor(lo)
        stage_send!(cache, i, 1, eltype(f), len, f.data)
        MPI.Isend(mpi_sendbuf(cache, i, 1, eltype(f), len, f.data), comm, cache.reqs[i, 1, 2]; dest=lo, tag=EXCHANGE_TAG_LO)
    end
    # Slot 2 = my own high-side strip, heading to my hi-neighbor: a high-direction send.
    if has_neighbor(hi)
        stage_send!(cache, i, 2, eltype(f), len, f.data)
        MPI.Isend(mpi_sendbuf(cache, i, 2, eltype(f), len, f.data), comm, cache.reqs[i, 2, 2]; dest=hi, tag=EXCHANGE_TAG_HI)
    end
    post_isend!(cache, Base.tail(fields), dim, lo, hi, comm, i + 1)
    return nothing
end

# Perform the exchange for one dimension. If `overlap` is false, waits for and
# unpacks everything before returning. If `overlap` is true (only ever used for
# the last requested dimension), leaves the receive (and, in the local-copy
# case, the unpack) outstanding and returns a handle for `halo_exchange_finish!`
# to complete.
function exchange_dim!(g::Grid{N}, fields::Tuple, dim::Integer; overlap::Bool) where {N}
    cache = g.excache
    nfields = length(fields)
    ensure_capacity!(cache, nfields)
    lo, hi = g.topology.neighbors[dim]
    rank = g.topology.rank

    (lo == rank) == (hi == rank) || error(
        "inconsistent neighbors in dimension $dim: either both neighbors must equal this rank " *
        "(single-rank periodic) or neither must.")
    self_periodic = (lo == rank)

    # Pack: for every field, write the strips destined for each real neighbor
    # into this grid's own send buffers.
    pack_dim!(cache, fields, dim, lo, hi, 1)

    if self_periodic
        copy_self_periodic!(cache, fields, dim, 1)
        handle = ExchangeHandle{N,typeof(fields)}(g, fields, dim, true)
        overlap || unpack_dim!(g, fields, dim)
    else
        prepare_mpi_fields!(cache, fields)
        post_irecv!(cache, fields, dim, lo, hi, g.topology.comm, 1)
        post_isend!(cache, fields, dim, lo, hi, g.topology.comm, 1)
        handle = ExchangeHandle{N,typeof(fields)}(g, fields, dim, false)
        if !overlap
            wait_recvs_and_unpack!(g, fields, dim)
            wait_sends!(g, fields, dim)
        end
    end
    # Always a handle (only the last dimension's is used): a `Nothing` branch
    # makes the return type a Union and boxes the handle (measured 256 B/call).
    return handle
end

@inline unpack_dim_r!(cache, ::Tuple{}, dim, i) = nothing
@inline function unpack_dim_r!(cache, fields::Tuple, dim, i)
    f = fields[1]
    len = halo_length(f, dim)
    unpack!(f.data, unpack_source(cache, i, 1, eltype(f), len, f.data), recvrange(f, dim, 1), cache.threaded_pack)
    unpack!(f.data, unpack_source(cache, i, 2, eltype(f), len, f.data), recvrange(f, dim, 2), cache.threaded_pack)
    unpack_dim_r!(cache, Base.tail(fields), dim, i + 1)
    return nothing
end
unpack_dim!(g::Grid, fields::Tuple, dim::Integer) = (unpack_dim_r!(g.excache, fields, dim, 1); nothing)

@inline wait_recvs_and_unpack_r!(cache, ::Tuple{}, dim, lo, hi, i) = nothing
@inline function wait_recvs_and_unpack_r!(cache, fields::Tuple, dim, lo, hi, i)
    f = fields[1]
    len = halo_length(f, dim)
    # stage_recv! is a no-op for CPU fields (unpack_source is the MPI buffer)
    # and for GPU-aware MPI, and a host->device copy otherwise — see backend.jl.
    if has_neighbor(lo)
        MPI.Wait(cache.reqs[i, 1, 1])
        stage_recv!(cache, i, 1, eltype(f), len, f.data)
        unpack!(f.data, unpack_source(cache, i, 1, eltype(f), len, f.data), recvrange(f, dim, 1), cache.threaded_pack)
    end
    if has_neighbor(hi)
        MPI.Wait(cache.reqs[i, 2, 1])
        stage_recv!(cache, i, 2, eltype(f), len, f.data)
        unpack!(f.data, unpack_source(cache, i, 2, eltype(f), len, f.data), recvrange(f, dim, 2), cache.threaded_pack)
    end
    wait_recvs_and_unpack_r!(cache, Base.tail(fields), dim, lo, hi, i + 1)
    return nothing
end
function wait_recvs_and_unpack!(g::Grid, fields::Tuple, dim::Integer)
    lo, hi = g.topology.neighbors[dim]
    wait_recvs_and_unpack_r!(g.excache, fields, dim, lo, hi, 1)
    return nothing
end

@inline wait_sends_r!(cache, ::Tuple{}, lo, hi, i) = nothing
@inline function wait_sends_r!(cache, fields::Tuple, lo, hi, i)
    has_neighbor(lo) && MPI.Wait(cache.reqs[i, 1, 2])
    has_neighbor(hi) && MPI.Wait(cache.reqs[i, 2, 2])
    wait_sends_r!(cache, Base.tail(fields), lo, hi, i + 1)
    return nothing
end
function wait_sends!(g::Grid, fields::Tuple, dim::Integer)
    lo, hi = g.topology.neighbors[dim]
    wait_sends_r!(g.excache, fields, lo, hi, 1)
    return nothing
end

# Recurses on the `dims` `Tuple` (all dimensions but the last synchronous,
# the last one left `overlap`ped) instead of a runtime `for dim in dims` loop.
# This isn't just style: a `for`-loop version of this — even after `fields`
# itself was made allocation-free via `Base.tail` recursion elsewhere in this
# file — measured ~600B/call of *further* allocation once combined with
# `check_fields` in the same function, apparently past some inlining-cost
# threshold the compiler uses to decide whether to keep `fields`/`dims`
# unboxed across the loop body. Recursing on the `Tuple` type sidesteps the
# question entirely: each call specializes on a strictly smaller `Tuple` type,
# so the whole thing unrolls and inlines like the direct/manual calls that
# *did* measure zero (see "Performance, memory footprint, and GC-pressure
# goals" in REDESIGN_PLAN.md for the measured before/after).
@inline exchange_dims!(grid, fields, dimtuple::Tuple{Any}) =
    exchange_dim!(grid, fields, dimtuple[1]; overlap = true)
@inline function exchange_dims!(grid, fields, dimtuple::Tuple)
    exchange_dim!(grid, fields, dimtuple[1]; overlap = false)
    exchange_dims!(grid, fields, Base.tail(dimtuple))
end

# Recursion again, not `all(d -> 1 <= d <= N, dims)`: `all(f, ::Tuple)` is a
# generic higher-order Base function, and passing it a closure here measured
# the same ~600B/call as the loop above once combined with the rest of
# `halo_exchange_start!`'s body — recursing directly sidesteps it (measured
# 0B/call; see "Performance, memory footprint, and GC-pressure goals" in
# REDESIGN_PLAN.md).
@inline dims_in_range(::Val{N}, ::Tuple{}) where {N} = true
@inline function dims_in_range(::Val{N}, dims::Tuple) where {N}
    d = dims[1]
    (1 <= d <= N) && dims_in_range(Val(N), Base.tail(dims))
end

# Runs before `pending` is set, so a bad `dims` (e.g. `(1.0,)`) cannot lock the grid.
check_dims(::Val{N}, ::Tuple{}) where {N} = error("dims must contain at least one dimension.")
@inline function check_dims(::Val{N}, dims::Tuple) where {N}
    dims isa Tuple{Vararg{Integer}} || error("dims must contain integer indices; got dims=$dims.")
    dims_in_range(Val(N), dims) || error("dims must be within 1:$N; got dims=$dims.")
    return nothing
end

# The sent strips `ol-hw+1:ol` and `n-ol+1:n-ol+hw` hold owned cells only if
# `overlaps[d] + halowidths[d] <= size(data,d)`; otherwise a rank sends halo
# cells it has not received yet. Skipped only where no rank communicates
# (one rank, non-periodic); that test is rank-independent, so every rank errors
# alike instead of some hanging.
@inline check_widths(topo, ::Tuple{}, dims) = nothing
@inline function check_widths(topo, fields::Tuple, dims)
    check_widths_in_dims(topo, fields[1], dims)
    check_widths(topo, Base.tail(fields), dims)
    return nothing
end
@inline check_widths_in_dims(topo, f, ::Tuple{}) = nothing
@inline function check_widths_in_dims(topo, f, dims::Tuple)
    d = dims[1]
    topo.dims[d] == 1 && !topo.periods[d] ||
        f.overlaps[d] + f.halowidths[d] <= size(f.data, d) ||
        throw_width_error(f.overlaps[d], f.halowidths[d], size(f.data, d), d)
    check_widths_in_dims(topo, f, Base.tail(dims))
    return nothing
end
# Out of line so the error string does not count against inlining the hot path.
@noinline throw_width_error(ol, hw, n, d) = error(
    "exchange in dimension $d needs overlaps[$d] + halowidths[$d] = $(ol + hw) <= size(data,$d) = $n; " *
    "otherwise the sent strip contains halo cells.")

@inline function begin_exchange!(cache::ExchangeCache)
    cache.pending = true
    return cache.epoch += 1
end

@inline function check_handle(handle::ExchangeHandle)
    cache = handle.grid.excache
    cache.pending && handle.epoch == cache.epoch || throw_stale_handle()
    return nothing
end
@noinline throw_stale_handle() = error(
    "this exchange handle was already finished, or a newer exchange was started on its grid; " *
    "call halo_exchange_finish! exactly once per halo_exchange_start!.")

"""
    halo_exchange_start!(grid::Grid, fields::Field...; dims::Tuple=ntuple(identity, ndims(grid)), region::Symbol=:full) :: ExchangeHandle

Start a halo exchange for `fields` on `grid`, over dimensions `dims` (an
`NTuple` of `Int`s, processed in the given order). Every dimension except the
last is completed fully synchronously before returning (sequential-dimension
correctness, for correct corners); the last dimension's receive (and, for a
single-rank-periodic dimension, unpack) is left outstanding. Only one exchange
may be in flight per `Grid` at a time — starting a second before finishing the
first raises an error; multiple *different* grids remain fully concurrent.

With `region=:faces`, exchange only face halos, excluding transverse halo
cells in other grid dimensions. Every requested dimension remains pending
until finish; edges and corners stay unchanged. Use this for axial stencils
that do not read diagonal halos. Extra trailing field dimensions (such as
batched components) retain their full extent. Repeated dimensions are rejected
in face mode. Peers must use the same region, dimensions, and field order.
Keep input fields unchanged while computing into separate output storage.

In both modes, every requested dimension `d` of every field must satisfy
`overlaps[d] + halowidths[d] <= size(data,d)`, so the sent strip holds only
owned cells (`3*halowidths[d]` with default overlaps). Dimensions with a single
rank and no periodicity are exempt, since nothing is sent there.

Call [`halo_exchange_finish!`](@ref) exactly once on the returned handle to
complete it; finishing a handle twice, or a stale handle after a newer start,
raises an error. [`update_halo!`](@ref) is a convenience wrapper that does both.
"""
function halo_exchange_start!(grid::Grid{N}, fields::Field...;
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {N}
    return _halo_exchange_start!(grid, fields, dims, region)
end

# Branch on the Symbol in a tuple-positional wrapper instead of constructing
# Val(region) dynamically in the keyword sorter: the latter loses field
# specialization and allocates even when the default region is constant.
@inline function _halo_exchange_start!(grid, fields::Tuple, dims::Tuple, region::Symbol)
    region === :full && return _halo_exchange_start!(grid, fields, dims)
    region === :faces && return _halo_exchange_start!(grid, fields, dims, Val(:faces))
    error("region must be :full or :faces; got region=$region.")
end

# `halo_exchange_start!` itself is deliberately a thin wrapper forwarding into
# this method, which takes `fields` as a plain positional `Tuple` rather than
# `Field...`. Combining splatted varargs with a keyword argument in the *same*
# method measured ~688B/call here: Julia's auto-generated keyword-sorter body
# method for such a signature types the vararg tail as the abstract
# `Vararg{Field}` (only the first field stays concretely typed), and once the
# body is complex enough that the optimizer doesn't fully inline that
# generated method back into the caller, everything downstream sees an
# abstractly-typed `fields` and boxes. Moving the real work to a method that
# takes `fields::Tuple` positionally (no vararg at all) sidesteps this
# entirely — confirmed by measurement to be 0B/call (see "Performance, memory
# footprint, and GC-pressure goals" in REDESIGN_PLAN.md).
function _halo_exchange_start!(grid::Grid{N}, fields::Tuple, dims::Tuple) where {N}
    grid.excache.pending && error(
        "a halo exchange is already in flight on this grid; call halo_exchange_finish! on it first " *
        "(at most one in-flight exchange per Grid is supported; use separate Grids for concurrent exchanges).")
    check_fields(grid, fields)
    check_dims(Val(N), dims)
    check_widths(grid.topology, fields, dims)

    begin_exchange!(grid.excache)
    return exchange_dims!(grid, fields, dims)
end

"""
    halo_exchange_finish!(handle::ExchangeHandle) :: Nothing

Complete a halo exchange started by [`halo_exchange_start!`](@ref): waits for
outstanding communication and unpacks it into the fields' arrays. Full exchange
finishes the last dimension; face exchange finishes all requested dimensions.
"""
function halo_exchange_finish!(handle::ExchangeHandle{N,F,Tuple{}}) where {N,F}
    check_handle(handle)
    if handle.self_periodic
        unpack_dim!(handle.grid, handle.fields, handle.dim)
    else
        wait_recvs_and_unpack!(handle.grid, handle.fields, handle.dim)
        wait_sends!(handle.grid, handle.fields, handle.dim)
    end
    handle.grid.excache.pending = false
    return nothing
end

"""
    update_halo!(grid::Grid, fields::Field...; dims::Tuple=ntuple(identity, ndims(grid)), region::Symbol=:full) :: Nothing

Update the halo of `fields` on `grid`: `halo_exchange_start!` immediately
followed by `halo_exchange_finish!`. Prefer this unless you specifically want
to overlap communication with other work. `region=:full` propagates edges and
corners sequentially; `region=:faces` fills only faces and permits all requested
dimensions to remain pending in the split API. See [`halo_exchange_start!`](@ref).
"""
function update_halo!(grid::Grid{N}, fields::Field...;
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {N}
    return _update_halo!(grid, fields, dims, region)
end

function _update_halo!(grid, fields::Tuple, dims::Tuple, region::Symbol)
    # Calls the `Tuple`-positional inner methods directly rather than
    # `halo_exchange_start!(grid, fields...; dims=dims)`, for the same reason
    # `halo_exchange_start!` itself is a thin wrapper — see the comment there.
    # Branch on `region` here and finish the handle inside each branch. The
    # Symbol-dispatching `_halo_exchange_start!` returns a `Union` of the full-
    # and faces-handle types, which the compiler may box when it is not inlined
    # into the caller; keeping each handle concretely typed avoids that.
    if region === :full
        halo_exchange_finish!(_halo_exchange_start!(grid, fields, dims))
    elseif region === :faces
        halo_exchange_finish!(_halo_exchange_start!(grid, fields, dims, Val(:faces)))
    else
        error("region must be :full or :faces; got region=$region.")
    end
    return nothing
end

# Migration/ergonomics convenience: let a caller pass bare `AbstractArray`s
# alongside (or instead of) pre-built `Field`s, so a single call can mix a
# default-halowidths array with an explicitly-`field`-wrapped, custom-width
# one — see `field`'s docstring in field.jl. `as_field` is the identity on an
# already-built `Field` (so an explicit per-array width always wins) and
# wraps a bare array with `field`'s own default width otherwise; no second,
# competing default lives here.
#
# Deliberately a *separate* method family from the `Field...`-only ones
# above, not a generalization of them: those are measured zero-allocation
# (see "Performance, memory footprint, and GC-pressure goals" in
# REDESIGN_PLAN.md) and that guarantee must not depend on this convenience
# path's behavior. A call whose arguments are all already `Field`s still
# resolves to the narrower `Field...` methods above (strictly more specific
# than `Union{Field,AbstractArray}...`, so Julia prefers it), leaving the
# tested zero-allocation hot path completely untouched; only a call that
# actually includes a bare array reaches the code below, and wrapping a bare
# array in a `Field` on every call is expected to cost a small, one-time-per-
# call allocation — the same "opt-in convenience has a real cost" trade as
# `threaded_pack`, not a regression of the default path.
as_field(f::Field) = f
as_field(A::AbstractArray) = field(A)

to_fields(::Tuple{}) = ()
to_fields(args::Tuple) = (as_field(args[1]), to_fields(Base.tail(args))...)

"""
    halo_exchange_start!(grid::Grid, args::Union{Field,AbstractArray}...; dims=..., region=:full) :: ExchangeHandle
    update_halo!(grid::Grid, args::Union{Field,AbstractArray}...; dims=..., region=:full) :: Nothing

Convenience forms of [`halo_exchange_start!`](@ref)/[`update_halo!`](@ref) that
accept bare arrays as well as `Field`s, in any mix, in the same call: a bare
`AbstractArray` is wrapped with `field`'s default halowidths (currently `1` in
every dimension), while a `Field` keeps whatever explicit `halowidths` it was
built with — so a single call can give different arrays different widths
without every array needing to be pre-wrapped:

```julia
update_halo!(grid, A, B, field(C; halowidths=(2, 2)))  # A, B default; C custom
```
"""
function halo_exchange_start!(grid::Grid{N}, args::Union{Field,AbstractArray}...;
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {N}
    return _halo_exchange_start!(grid, to_fields(args), dims, region)
end

function update_halo!(grid::Grid{N}, args::Union{Field,AbstractArray}...;
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {N}
    return _update_halo!(grid, to_fields(args), dims, region)
end
