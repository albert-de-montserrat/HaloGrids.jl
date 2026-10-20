export ExchangeCache

"""
    ExchangeCache

Per-`Grid`-owned scratch state for halo exchange: raw send/recv byte buffers
and the MPI request array. Full exchange reuses one slot per field position;
face exchange reserves separate slots per dimension and field position so
all requested dimensions can remain pending together. This is a
mutable struct whose fields are reassigned as buffers grow — never a
module-level `let`-captured variable — precisely so that two live `Grid`s
never share or clobber each other's buffers/requests, and so that growing a
buffer is a plain struct-field store rather than the reassignment of a
closure-captured local (which Julia would box, and which we measured as slower
than the allocation it would avoid).

`pending` guards the "one in-flight exchange per `Grid`" v1 restriction: it is
set by `halo_exchange_start!` and cleared by `halo_exchange_finish!`, so a
second `start!` before a matching `finish!` errors instead of corrupting
buffers shared between the two overlapping exchanges. `epoch` counts started
exchanges; each handle records the epoch it was started in, so finishing a
handle twice, or finishing a stale handle while a newer exchange is pending,
errors instead of unpacking the newer exchange's buffers into the old fields.

`reqs` holds one distinct `MPI.Request` per slot, reused in place by every
`Isend`/`Irecv!` so a steady-state exchange allocates no request objects.

`threaded_pack` is the opt-in flag (set at `grid` construction, see
[`grid`](@ref)) for splitting large pack/unpack copies across
`Threads.@spawn` tasks. It defaults to `false`: real OS-thread parallelism
here has a real, unavoidable allocation cost (each spawned task allocates,
measured ~500-670B regardless of chunk size — see "Performance, memory
footprint, and GC-pressure goals" in REDESIGN_PLAN.md), so it must never
turn on under a caller who hasn't asked for it and expects the package's
zero-allocation default to hold.

`gpu_cache` is an untyped (`Ref{Any}`) slot, `nothing` until a GPU extension
(CUDA/AMDGPU) lazily constructs its own device-buffer cache struct into it on
this grid's first GPU-array exchange — see `src/backend.jl`. The core package
never inspects its contents; keeping it `Any`-typed here means the core
doesn't need to know the GPU cache's concrete type, and a CPU-only grid that
never touches a GPU array never allocates one.
"""
mutable struct ExchangeCache
    sendbufs::Vector{NTuple{2,Vector{UInt8}}}  # sendbufs[field_i][neighbor_n] -> raw bytes
    recvbufs::Vector{NTuple{2,Vector{UInt8}}}
    reqs::Array{MPI.Request,3}                 # reqs[field_i, neighbor_n, phase]; phase 1=recv, 2=send
    pending::Bool
    epoch::Int
    threaded_pack::Bool
    gpu_cache::Ref{Any}
end

ExchangeCache(threaded_pack::Bool = false) = ExchangeCache(
    NTuple{2,Vector{UInt8}}[],
    NTuple{2,Vector{UInt8}}[],
    new_requests(0),
    false,
    0,
    threaded_pack,
    Ref{Any}(nothing),
)

# Distinct objects, not `fill(MPI.Request(), ...)`: each slot is reused in place.
new_requests(nfields) = [MPI.Request() for _ in 1:nfields, _ in 1:2, _ in 1:2]

function ensure_capacity!(cache::ExchangeCache, nfields::Integer)
    while length(cache.sendbufs) < nfields
        push!(cache.sendbufs, (UInt8[], UInt8[]))
        push!(cache.recvbufs, (UInt8[], UInt8[]))
    end
    if size(cache.reqs, 1) < nfields
        newreqs = new_requests(nfields)
        newreqs[1:size(cache.reqs, 1), :, :] .= cache.reqs
        cache.reqs = newreqs
    end
    return nothing
end

# Return a length-`len` view of `raw`, reinterpreted as `T`, growing `raw` first if
# needed. MPI.Buffer only knows how to unwrap `Array`/`SubArray`/`Ref`, not a
# `ReinterpretArray` (nor a `ReinterpretArray` of a `SubArray`), so `raw` must be
# reinterpreted whole (giving a plain `ReinterpretArray` over the `Vector{UInt8}`)
# and only *then* viewed down to the needed length (giving the required top-level
# `SubArray`) — never the other way around. That also means `length(raw)` must be
# an exact multiple of `sizeof(T)` at the point of reinterpreting, which may not
# already hold if this slot was last grown for a differently-sized element type.
#
# Typed `AbstractVector{UInt8}` rather than the concrete `Vector{UInt8}` (even
# though only `Vector{UInt8}` reaches an actual MPI call, via `sendbuf`/`recvbuf`
# below) so a GPU extension's device-buffer cache can call this same function
# unchanged for its own `CuVector{UInt8}`/`ROCVector{UInt8}` raw buffers —
# `resize!`/`reinterpret`/`view` all have GPU-array methods, so the logic here
# needs no GPU-specific duplicate.
function typed_buf!(raw::AbstractVector{UInt8}, ::Type{T}, len::Integer) where {T}
    nbytes = len * sizeof(T)
    if length(raw) < nbytes
        resize!(raw, max(nbytes, 2 * length(raw)))
    end
    if length(raw) % sizeof(T) != 0
        resize!(raw, length(raw) + (sizeof(T) - length(raw) % sizeof(T)))
    end
    typed = reinterpret(T, raw)
    return view(typed, 1:len)
end

sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T} =
    typed_buf!(cache.sendbufs[i][n], T, len)
recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T} =
    typed_buf!(cache.recvbufs[i][n], T, len)
