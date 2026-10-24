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
device arrays a generic strided `copyto!`. By default `stage_send!`/
`stage_recv!` are the plain contiguous device<->host copies that bridge that
device buffer to the host `sendbuf`/`recvbuf` MPI sends/receives from. On a
grid built with `gpu_aware_mpi=true` they are no-ops, and `mpi_sendbuf`/
`mpi_recvbuf` (below) hand MPI the device buffer itself.
"""
pack_target(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    sendbuf(cache, i, n, T, len)
unpack_source(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    recvbuf(cache, i, n, T, len)
stage_send!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing
stage_recv!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing

"""
Buffers handed to `MPI.Isend`/`MPI.Irecv!`. The defaults are the host
`sendbuf`/`recvbuf`. A GPU extension returns the device buffer instead when the
grid was built with `gpu_aware_mpi=true`, and then `stage_send!`/`stage_recv!`
become no-ops.

`prepare_mpi!` runs once per field and dimension after packing and before any
receive or send is posted. A GPU-aware path uses it to wait for the pack
kernels and for the previous dimension's unpack kernels, which read and write
the same device buffers that MPI is about to access.
"""
mpi_sendbuf(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    sendbuf(cache, i, n, T, len)
mpi_recvbuf(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    recvbuf(cache, i, n, T, len)
prepare_mpi!(cache, ::AbstractArray) = nothing

@inline prepare_mpi_fields!(cache, ::Tuple{}) = nothing
@inline function prepare_mpi_fields!(cache, fields::Tuple)
    prepare_mpi!(cache, fields[1].data)
    prepare_mpi_fields!(cache, Base.tail(fields))
end
