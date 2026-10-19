export gather!

"""
    gather!(A, A_global, topology_or_grid; root=0)

Gather local array `A` from each member of the Cartesian process topology into
one large array `A_global` on the `root` process (`A_global` may be `nothing`
on every other process). `size(A_global)` must equal `size(A) .* dims`, where
`dims` are the process-topology dimensions. Every rank must pass an `A` of the
same size. `ndims(A) <= ndims(A_global) <= ndims(topology)`; topology
dimensions beyond `ndims(A_global)` must be 1. Invalid arguments on any rank
raise an error on every rank.

Uses the exact zero-copy technique from the old package (a derived
`MPI.Types.create_subarray` + `create_resized` + a single `Gatherv!`/`VBuffer`
call — no manual pack loop); the only change is that `dims`/`comm` are read
directly off the explicit `Topology`/`Grid` argument instead of an implicit
current-grid lookup.
"""
function gather!(A::AbstractArray{T,N2}, A_global::Union{AbstractArray{T,N},Nothing},
                  topology::Topology{ND}; root::Integer = 0) where {T,N,N2,ND}
    comm = topology.comm
    isroot = MPI.Comm_rank(comm) == root
    # Validate on every rank before Gatherv!: a rank with a different size(A)
    # would otherwise land shifted with no error, and a root-only error would
    # leave the other ranks blocked in Gatherv!.
    root_err = isroot ? gather_root_error(A, A_global, topology) : nothing
    sz = (ntuple(d -> size(A, d), ND)..., N2)
    # One max-reduction gives max(sz), -min(sz) (as max of -sz) and the root's error flag.
    lims = MPI.Allreduce([sz..., (.-sz)..., root_err === nothing ? 0 : 1], max, comm)
    lims[1:ND+1] == .-lims[ND+2:2ND+2] || error(
        "gather!: every rank must pass an A of the same size; got sizes between " *
        "$(Tuple(.-lims[ND+2:2ND+1])) and $(Tuple(lims[1:ND])).")
    lims[end] == 0 && return gather_checked!(A, A_global, topology, comm, isroot, root)
    error(root_err === nothing ? "gather!: invalid arguments on the root rank (see its error)." : root_err)
end

# Error message for the root's arguments, or `nothing` if they are valid.
function gather_root_error(A::AbstractArray{T,N2}, A_global, topology::Topology{ND}) where {T,N2,ND}
    A_global === nothing && return "A_global cannot be `nothing` on the root process."
    N = ndims(A_global)
    N2 <= N || return "ndims(A)=$N2 must be <= ndims(A_global)=$N."
    N <= ND || return "ndims(A_global)=$N must be <= the topology's number of dimensions ($ND)."
    all(==(1), topology.dims[N+1:end]) || return "ndims(A_global)=$N needs topology dims " *
        "$(N+1):$ND to be 1, got dims=$(topology.dims)."
    size_A = (size(A)..., ntuple(_ -> 1, N - N2)...)
    expected = topology.dims[1:N] .* size_A
    size(A_global) == expected || return "size(A_global)=$(size(A_global)) must equal size(A) .* dims = $expected."
    return nothing
end

function gather_checked!(A::AbstractArray{T,N2}, A_global, topology, comm, isroot, root) where {T,N2}
    if isroot
        N = ndims(A_global)
        dims = topology.dims[1:N]
        size_A = (size(A)..., ntuple(_ -> 1, N - N2)...)
        offset = ntuple(_ -> 0, N)
        subtype = MPI.Types.create_subarray(size(A_global), size_A, offset, MPI.Datatype(eltype(A_global)))
        subtype = MPI.Types.create_resized(subtype, 0, size(A, 1) * Base.elsize(A_global))
        MPI.Types.commit!(subtype)
        counts = fill(Cint(1), reverse(dims))
        displs = zeros(Cint, reverse(dims))
        csizes = cumprod(size_A[2:end] .* dims[1:end-1])
        strides = (1, csizes...)
        for I in CartesianIndices(displs)
            off = reverse(Tuple(I - oneunit(I)))
            displs[I] = sum(off .* strides)
        end
        recvbuf = MPI.VBuffer(A_global, vec(counts), vec(displs), subtype)
        MPI.Gatherv!(A, recvbuf, comm; root)
    else
        MPI.Gatherv!(A, nothing, comm; root)
    end
    return nothing
end

gather!(A, A_global, grid::Grid; root::Integer = 0) = gather!(A, A_global, grid.topology; root = root)
