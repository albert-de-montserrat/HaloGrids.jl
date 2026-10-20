"""
GPU dispatch seam for pack/unpack and MPI-transport staging, overridden by the
CUDA/AMDGPU package extensions (`ext/HaloGridsCUDAExt.jl`,
`ext/HaloGridsAMDGPUExt.jl`) and resolved by ordinary multiple dispatch on the
array type — a GPU extension adds methods for `AnyCuArray`/`AnyROCArray` and
nothing else needs to change at the call sites in `halo_exchange.jl`. Unlike
the old package, no runtime "is the extension loaded" flag is needed at all.

GPU-specific cached state (streams, device buffers) lives behind the `Ref{Any}`
`gpu_cache` slot owned by the `Grid`'s `ExchangeCache` (see exchange_cache.jl),
never module-global.

For a plain CPU `AbstractArray`, `pack_target`/`unpack_source` are literally
`sendbuf`/`recvbuf` (the same host buffer packing has always used) and
`stage_send!`/`stage_recv!` are no-ops, so these four defaults reproduce
today's CPU behavior exactly, at zero extra runtime cost once specialized per
concrete array type.

For a GPU array, a device-resident buffer pool (owned by the grid's
`ExchangeCache.gpu_cache`, never module-global) is the actual pack/unpack
target/source — `pack!`/`unpack!` (halo_exchange.jl) then run entirely on
device via `copyto!`/`view`, unmodified, since GPUArrays.jl already gives
device arrays a generic strided `copyto!`. `stage_send!`/`stage_recv!` are the
plain contiguous device<->host copies that bridge that device buffer to the
host `sendbuf`/`recvbuf` MPI actually sends/receives from: MPI transport
itself never becomes GPU-aware, only pack/unpack does. This split is also the
documented seam for a future CUDA-aware-MPI path (make `stage_send!`/
`stage_recv!` no-ops and let `post_isend!`/`post_irecv!` operate on the device
buffer directly) — not implemented here since it requires a CUDA-aware MPI
build to validate, which this project's local Microsoft MPI is not.
"""
pack_target(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    sendbuf(cache, i, n, T, len)
unpack_source(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    recvbuf(cache, i, n, T, len)
stage_send!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing
stage_recv!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing
