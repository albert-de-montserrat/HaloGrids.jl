export field, halowidths, overlaps, Field

"""
    Field{T,N,A}

A local array paired with its own per-dimension halo width. Not tied to any
`Grid` at construction time — the `Grid` is supplied later, at the exchange
call site — so a `Field` never needs to know about multi-grid concerns, and
the same field could (in principle) be exchanged against different grids in
different calls.

`halowidths[d]` is the number of boundary cells, in dimension `d`, that are
exchanged with neighbors: the outer `halowidths[d]` cells on each side of the
local array are treated as halo (overwritten on receive), and the adjacent
interior cells of the same width are what gets sent to the neighbor. This
replaces the old package's separate `overlaps`/`halowidths` pair with a single
concept: there is no grid-wide default and no notion of a halo wider than what
each field itself declares.

`overlaps[d]` is the number of array lines, including the halo, that
neighboring ranks share in dimension `d`. It defaults to `2*halowidths[d]`
(the sent strip is directly adjacent to the halo); larger values move the
sent strip further inward, as ImplicitGlobalGrid does for staggered or padded
arrays. See [`overlaps`](@ref).
"""
struct Field{T,N,A<:AbstractArray{T,N}}
    data::A
    halowidths::NTuple{N,Int}
    overlaps::NTuple{N,Int}
end

"""
    field(A::AbstractArray; halowidths::NTuple{N,Integer} = ntuple(_->1,N),
          overlaps::NTuple{N,Integer} = 2 .* halowidths) :: Field

Wrap array `A` as a `Field` with the given per-dimension `halowidths`, defaulting
to a width of `1` in every dimension (the common case for a single-cell-wide
stencil). Requires, for every dimension `d`: `halowidths[d] >= 1` and
`2*halowidths[d] <= size(A,d)` (the local array must be at least wide enough to
hold two non-overlapping halo copies — one sent, one received — in every
dimension) and `2*halowidths[d] <= overlaps[d] <= size(A,d)`. `eltype(A)`
must be a bits type.

This default lives here, on the constructor, and nowhere else — there is still
no grid-wide or package-wide halowidths default stored anywhere, so different
`Field`s (even wrapping the same array in different calls) remain free to use
different widths. This is what lets [`update_halo!`](@ref)'s bare-`AbstractArray`
convenience methods mix a default-width array and an explicitly-`field`-wrapped,
custom-width array in the very same call.
"""
function field(
        A::AbstractArray{T,N}; halowidths::NTuple{N,Integer} = ntuple(_ -> 1, N),
        overlaps::NTuple{N,Integer} = 2 .* halowidths
    ) where {T,N}
    isbitstype(T) || error("Field element type $T is not a bits type.")
    hw = Int.(halowidths)
    ol = Int.(overlaps)
    for d in 1:N
        hw[d] >= 1 || error("halowidths[$d]=$(hw[d]) must be >= 1.")
        2 * hw[d] <= size(A, d) || error(
            "halowidths[$d]=$(hw[d]) is too large for size(A,$d)=$(size(A,d)): " *
            "need 2*halowidths[$d] <= size(A,$d) so the sent and received halo " *
            "strips don't overlap.")
        ol[d] >= 2 * hw[d] || error(
            "overlaps[$d]=$(ol[d]) must be >= 2*halowidths[$d]=$(2 * hw[d]).")
        ol[d] <= size(A, d) || error(
            "overlaps[$d]=$(ol[d]) must be <= size(A,$d)=$(size(A, d)).")
    end
    return Field{T,N,typeof(A)}(A, Tuple(hw), Tuple(ol))
end

"""
    overlaps(f::Field) :: NTuple{N,Int}

Number of array lines per dimension that neighboring ranks share, including the
halo (`2*halowidths` by default). A larger overlap leaves the halo width
unchanged but sends the interior lines `overlaps[d]-halowidths[d]+1:overlaps[d]`
instead of the ones directly adjacent to the halo, as ImplicitGlobalGrid does for
staggered or padded arrays.
"""
overlaps(f::Field) = f.overlaps

"""
    halowidths(f::Field) :: NTuple{N,Int}

The per-dimension halo widths `f` was constructed with.
"""
halowidths(f::Field) = f.halowidths

Base.size(f::Field, args...) = size(f.data, args...)
Base.ndims(::Field{T,N}) where {T,N} = N
Base.eltype(::Field{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", f::Field{T,N}) where {T,N}
    print(io, "Field{$T,$N}(size=$(size(f.data)), halowidths=$(f.halowidths), overlaps=$(f.overlaps))")
end
