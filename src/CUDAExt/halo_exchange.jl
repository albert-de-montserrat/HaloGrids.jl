import HaloGrids
import HaloGrids: ExchangeCache, sendbuf, recvbuf
using CUDA

# pack_target/unpack_source hand pack!/unpack! a device-resident buffer
# instead of the host one. Contiguous slabs then use GPUArrays' copyto!
# (112 B of host allocation per call). For a strided slab, GPUArrays' copyto!
# goes through its broadcast machinery, measured at ~2.3 KB per call on
# CUDA.jl 6.3, so those use a one-line copy kernel instead (~670 B, all of it
# CUDA.jl's own kernel lookup + launch).
function copy_kernel!(dst, src)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(dst) && @inbounds dst[i] = src[i]
    return nothing
end

function kernel_copy!(dst, src)
    n = length(dst)
    n == 0 && return nothing
    threads = min(n, 256)
    @cuda threads=threads blocks=cld(n, threads) copy_kernel!(dst, src)
    return nothing
end

# A slab is contiguous iff every dimension before its last non-singleton range is full.
function is_contiguous(A, ranges::Tuple)
    k = length(ranges)
    while k > 1 && length(ranges[k]) == 1
        k -= 1
    end
    for d in 1:k-1
        length(ranges[d]) == size(A, d) || return false
    end
    return true
end

function HaloGrids.pack!(buf::AbstractVector, A::AnyCuArray, ranges::Tuple)
    v = view(A, ranges...)
    is_contiguous(A, ranges) ? copyto!(buf, v) : kernel_copy!(buf, v)
    return nothing
end
function HaloGrids.unpack!(A::AnyCuArray, buf::AbstractVector, ranges::Tuple)
    v = view(A, ranges...)
    is_contiguous(A, ranges) ? copyto!(v, buf) : kernel_copy!(v, buf)
    return nothing
end

HaloGrids.pack_target(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_sendbuf(cache, i, n, T, len)
HaloGrids.unpack_source(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_recvbuf(cache, i, n, T, len)

# The device<->host staging copies MPI transport needs (MPI always
# sends/receives the host sendbuf/recvbuf — see backend.jl). These move raw
# bytes directly between the two *unwrapped* Vector{UInt8}/CuVector{UInt8}
# buffers via the 5-argument `copyto!(dest, doffs, src, soffs, n)` form: the
# natural `copyto!(view_of_reinterpreted_host_buf, view_of_reinterpreted_dev_buf)`
# (i.e. through the same typed views pack_target/sendbuf hand out) silently
# falls back to Base's generic element-at-a-time `copyto!` and hits GPU scalar
# indexing (confirmed by direct reproduction), because GPUArrays.jl's fast
# host<->device path is only defined for the plain, unwrapped Array/CuArray
# types — not a SubArray of a ReinterpretArray. Bytes need no reinterpreting
# to move, so operating on the raw buffers sidesteps the whole issue.
# `sendbuf`/`gpu_sendbuf`/`recvbuf`/`gpu_recvbuf` are still called first, for
# their growth side effect (typed_buf! resizes the raw buffer if needed); the
# typed view each returns is otherwise unused here.
function HaloGrids.stage_send!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    nbytes = len * sizeof(T)
    sendbuf(cache, i, n, T, len)
    gpu_sendbuf(cache, i, n, T, len)
    copyto!(cache.sendbufs[i][n], 1, cuda_cache!(cache).sendbufs[i][n], 1, nbytes)
    return nothing
end
function HaloGrids.stage_recv!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    nbytes = len * sizeof(T)
    recvbuf(cache, i, n, T, len)
    gpu_recvbuf(cache, i, n, T, len)
    copyto!(cuda_cache!(cache).recvbufs[i][n], 1, cache.recvbufs[i][n], 1, nbytes)
    return nothing
end
