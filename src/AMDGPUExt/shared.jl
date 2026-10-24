import HaloGrids
import HaloGrids: ExchangeCache, typed_buf!
using AMDGPU

"""
    ViewCache

Structurally identical to `CUDACache`'s `ViewCache` (src/CUDAExt/shared.jl) —
see that file for the full rationale (a per-buffer cache of the last
`reinterpret`+`view` wrapper, avoiding a fresh `CuArray`/`ROCArray`-family
allocation on every steady-state pack/unpack call). Untested on this machine
(no AMD GPU available), ported mechanically like the rest of this file.
"""
mutable struct ViewCache
    eltype::DataType
    len::Int
    rawlen::Int
    view::Any
end
ViewCache() = ViewCache(Nothing, -1, -1, nothing)

function cached_typed_view!(vc::ViewCache, raw::ROCVector{UInt8}, ::Type{T}, len::Integer) where {T}
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
    ROCCache

Structurally identical to `CUDACache` (src/CUDAExt/shared.jl) — see that file
for the full rationale — with `ROCVector{UInt8}` in place of `CuVector{UInt8}`.
Ported mechanically rather than written independently, since the two backends'
buffer-pool logic has nothing GPU-vendor-specific in it (only `pack_target`/
`unpack_source`/`stage_send!`/`stage_recv!` in halo_exchange.jl touch
backend-specific array types directly). Untested on this machine (no AMD GPU
available; see REDESIGN_PLAN.md) — exercised only as far as
`AMDGPU.functional() == false` in `test/test_amdgpu.jl`.
"""
mutable struct ROCCache
    sendbufs::Vector{NTuple{2,ROCVector{UInt8}}}
    recvbufs::Vector{NTuple{2,ROCVector{UInt8}}}
    sendviews::Vector{NTuple{2,ViewCache}}
    recvviews::Vector{NTuple{2,ViewCache}}
end

ROCCache() = ROCCache(
    NTuple{2,ROCVector{UInt8}}[], NTuple{2,ROCVector{UInt8}}[],
    NTuple{2,ViewCache}[], NTuple{2,ViewCache}[],
)

function ensure_capacity!(gc::ROCCache, nfields::Integer)
    while length(gc.sendbufs) < nfields
        push!(gc.sendbufs, (ROCVector{UInt8}(undef, 0), ROCVector{UInt8}(undef, 0)))
        push!(gc.recvbufs, (ROCVector{UInt8}(undef, 0), ROCVector{UInt8}(undef, 0)))
        push!(gc.sendviews, (ViewCache(), ViewCache()))
        push!(gc.recvviews, (ViewCache(), ViewCache()))
    end
    return nothing
end

function rocm_cache!(cache::ExchangeCache)
    gc = cache.gpu_cache[]
    if gc === nothing
        gc = ROCCache()
        cache.gpu_cache[] = gc
    end
    return gc::ROCCache
end

function gpu_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = rocm_cache!(cache)
    ensure_capacity!(gc, i)
    return cached_typed_view!(gc.sendviews[i][n], gc.sendbufs[i][n], T, len)
end
function gpu_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = rocm_cache!(cache)
    ensure_capacity!(gc, i)
    return cached_typed_view!(gc.recvviews[i][n], gc.recvbufs[i][n], T, len)
end
