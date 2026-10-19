import HaloGrids
import HaloGrids: ExchangeCache, typed_buf!
using CUDA

"""
    ViewCache

Remembers the last `T`/`len`-typed view `typed_buf!` derived from one raw
device buffer, so a steady-state call that requests the same `(T, len)` against
an unchanged buffer can reuse that view object instead of building a fresh
`reinterpret`+`view` wrapper. This matters only on the GPU: on `Vector{UInt8}`,
`reinterpret`+`view` already measure exactly 0 B/call (elided by escape
analysis), but the identical code on `CuVector{UInt8}` measures ~80 B *per
wrapper* (~160 B for the pair) — confirmed by comparing the byte-for-byte-same
code against a plain `Vector` control — because a `CuArray`-family object must
register/track a device-memory reference for CUDA.jl's GC-integrated buffer
pool on construction, something no amount of Julia-side inlining avoids. See
REDESIGN_PLAN.md's GPU allocation audit for the full measurement.

`rawlen` (the raw buffer's `length` at the time `view` was built) is part of
the cache key precisely because `resize!` on the raw buffer mutates it in
place (same object identity) while reallocating its backing memory — so a
view built before a resize would silently reference stale memory after one.
Comparing `rawlen` against the buffer's current `length` (cheap: just a field
read, no allocation) is sufficient to detect that case: if it, `T`, and `len`
all still match, nothing about the derivation has changed since, so the old
result is still exactly what a fresh call would produce.
"""
mutable struct ViewCache
    eltype::DataType
    len::Int
    rawlen::Int
    view::Any
end
ViewCache() = ViewCache(Nothing, -1, -1, nothing)

function cached_typed_view!(vc::ViewCache, raw::CuVector{UInt8}, ::Type{T}, len::Integer) where {T}
    rawlen = length(raw)
    if vc.eltype === T && vc.len == len && vc.rawlen == rawlen
        return vc.view::AbstractVector{T}
    end
    v = typed_buf!(raw, T, len)
    vc.eltype = T
    vc.len = len
    vc.rawlen = length(raw)
    vc.view = v
    return v
end

"""
    CUDACache

Per-`Grid`-owned device-buffer pool for GPU halo exchange, lazily constructed
into `ExchangeCache.gpu_cache` (see `src/backend.jl` and `src/exchange_cache.jl`
in HaloGrids.jl) on this grid's first `CuArray` field — never module-global,
same rule as `ExchangeCache` itself. Structurally mirrors `ExchangeCache`'s own
`sendbufs`/`recvbufs`: one raw byte buffer per `(field position, neighbor)`,
grown on demand by the same `typed_buf!` HaloGrids already uses for its host
buffers (it is generic over `AbstractVector{UInt8}` for exactly this reason).
`sendviews`/`recvviews` cache the typed view derived from each raw buffer —
see `ViewCache` above.

Never holds an MPI request or touches MPI directly — `ExchangeCache.reqs`
remains the only request array regardless of a field's array type, since MPI
only ever sends/receives the host `sendbuf`/`recvbuf` (see `stage_send!`/
`stage_recv!` in halo_exchange.jl).
"""
mutable struct CUDACache
    sendbufs::Vector{NTuple{2,CuVector{UInt8}}}
    recvbufs::Vector{NTuple{2,CuVector{UInt8}}}
    sendviews::Vector{NTuple{2,ViewCache}}
    recvviews::Vector{NTuple{2,ViewCache}}
end

CUDACache() = CUDACache(
    NTuple{2,CuVector{UInt8}}[], NTuple{2,CuVector{UInt8}}[],
    NTuple{2,ViewCache}[], NTuple{2,ViewCache}[],
)

function ensure_capacity!(gc::CUDACache, nfields::Integer)
    while length(gc.sendbufs) < nfields
        push!(gc.sendbufs, (CuVector{UInt8}(undef, 0), CuVector{UInt8}(undef, 0)))
        push!(gc.recvbufs, (CuVector{UInt8}(undef, 0), CuVector{UInt8}(undef, 0)))
        push!(gc.sendviews, (ViewCache(), ViewCache()))
        push!(gc.recvviews, (ViewCache(), ViewCache()))
    end
    return nothing
end

# Lazily builds (once per grid) and returns this ExchangeCache's CUDACache. The
# core package never constructs or names this type — cache.gpu_cache is a bare
# Ref{Any} it never inspects (see exchange_cache.jl) — so a CPU-only grid never
# allocates one.
function cuda_cache!(cache::ExchangeCache)
    gc = cache.gpu_cache[]
    if gc === nothing
        gc = CUDACache()
        cache.gpu_cache[] = gc
    end
    return gc::CUDACache
end

function gpu_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = cuda_cache!(cache)
    ensure_capacity!(gc, i)
    return cached_typed_view!(gc.sendviews[i][n], gc.sendbufs[i][n], T, len)
end
function gpu_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = cuda_cache!(cache)
    ensure_capacity!(gc, i)
    return cached_typed_view!(gc.recvviews[i][n], gc.recvbufs[i][n], T, len)
end
