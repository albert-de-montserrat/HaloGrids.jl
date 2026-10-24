export global_reduce, global_allreduce, global_sum, global_max, global_min, global_norm

"""
    global_reduce(topology_or_grid, val, op; root::Integer=0)

Reduce `val` (typically a scalar diagnostic — a local max velocity, a partial
sum, a convergence residual) across every rank sharing `topology_or_grid`'s
communicator, using reduction operator `op` (e.g. `+`, `max`, `min`, or any
other operator `MPI.Reduce` accepts). The reduced result is returned on rank
`root`; the return value on every other rank is whatever `MPI.Reduce` itself
returns there (not meaningful — use [`global_allreduce`](@ref) if every rank
needs the answer).

This is the scalar-diagnostic counterpart to [`gather!`](@ref): where `gather!`
assembles whole per-rank arrays into one global array, `global_reduce` combines
one number per rank into one number. It does not know about halo widths or
owned-vs-halo regions — reduce your own local data with plain Julia first
(`sum(A)`, `maximum(A)`, excluding halo cells however is appropriate for that
computation) to get one scalar per rank, then pass that scalar here. This
split keeps the wrapper itself backend-agnostic: the local step can run on a
`CuArray`/`ROCArray` however it likes, and only a host scalar ever needs to
reach MPI, so no CUDA/ROCm-aware MPI transport is required (unlike a
hypothetical direct reduction over device-resident array data).
"""
function global_reduce(topology::Topology, val, op; root::Integer = 0)
    return MPI.Reduce(val, op, topology.comm; root)
end
global_reduce(grid::Grid, val, op; root::Integer = 0) = global_reduce(grid.topology, val, op; root)

"""
    global_allreduce(topology_or_grid, val, op)

Like [`global_reduce`](@ref), but the reduced result is returned on every
rank instead of only `root`. Use this for values every rank needs to keep
going — a global CFL timestep bound, a global convergence check — rather
than a value only meant for logging/output on one rank.
"""
function global_allreduce(topology::Topology, val, op)
    return MPI.Allreduce(val, op, topology.comm)
end
global_allreduce(grid::Grid, val, op) = global_allreduce(grid.topology, val, op)

"""
    global_sum(topology_or_grid, val)
    global_max(topology_or_grid, val)
    global_min(topology_or_grid, val)

[`global_allreduce`](@ref) specialized to `+`/`max`/`min` — the common cases
(a global sum for a conservation check, a global max for a CFL timestep bound
or a convergence residual, a global min for a stability bound). Every rank
gets the result. `global_max`/`global_min` return `NaN` if any rank passes
`NaN`. For a reduction only rank `root` needs, or any other operator, use
[`global_reduce`](@ref)/[`global_allreduce`](@ref) directly.
"""
global_sum(topology_or_grid, val) = global_allreduce(topology_or_grid, val, +)
global_max(topology_or_grid, val) = global_allreduce(topology_or_grid, val, max)
global_min(topology_or_grid, val) = global_allreduce(topology_or_grid, val, min)

# MPI's MAX/MIN need not propagate NaN (Open MPI drops it), so a diverged rank
# could pass a residual or CFL check. Reduce a NaN flag in the same message
# instead of a second Allreduce, and without a custom op (unsupported on aarch64).
global_max(topology_or_grid, val::AbstractFloat) = nan_aware_extremum(topology_or_grid, val, max)
global_min(topology_or_grid, val::AbstractFloat) = nan_aware_extremum(topology_or_grid, val, min)
function nan_aware_extremum(topology_or_grid, val::T, op) where {T<:AbstractFloat}
    # The flag is +1 under max and -1 under min, so `op` selects it on any NaN rank.
    flag = isnan(val) ? (op === max ? one(T) : -one(T)) : zero(T)
    v, f = global_allreduce(topology_or_grid, [val, flag], op)
    return iszero(f) ? v : T(NaN)
end

@doc (@doc global_sum) global_max
@doc (@doc global_sum) global_min

"""
    global_norm(topology_or_grid, A, p::Real=2)

Global `p`-norm of the data distributed across every rank sharing
`topology_or_grid`'s communicator: `(sum(|Aᵢ|^p) over every element on every
rank)^(1/p)`, matching `LinearAlgebra.norm(A, p)`'s convention (default
`p=2`, the Euclidean norm) but summed over every rank's data, not just one.
`A` is each rank's own local contribution — a plain array or a scalar; pass
the interior only if `A` carries halo cells that would otherwise be
double-counted, as with any other local reduction in this file.

Unlike `global_sum`/`global_max`/`global_min`, this is *not* a case of
"reduce locally, then combine with the same operator": for `p < Inf`, only
the sum of `p`-th powers is combinable across ranks (with a single `p`-th
root taken at the very end) — combining each rank's own *local* norm
directly (say, with `global_sum`) gives the wrong answer, silently, for any
`p` other than `1`. Because that mistake is easy to make by hand, this
function does the full computation (local power-sum via plain `sum`, which
runs on whatever backend `A` already lives on, then one cross-rank
`global_allreduce`, then the root) rather than asking the caller to
pre-reduce as the other functions here do. `p=Inf` computes the global
maximum absolute value instead (the ℓ∞ norm) via `global_max`, since *that*
one is directly combinable, and since the general `sum(abs(x)^p, A)` formula
is not numerically meaningful at `p=Inf`. `p=-Inf` gives the global minimum
absolute value. Other `p <= 0` are rejected. Integer data is converted to
floating point before powering, so it cannot overflow.
"""
function global_norm(topology::Topology, A, p::Real = 2)
    p == Inf && return global_max(topology, float(maximum(abs, A)))
    p == -Inf && return global_min(topology, float(minimum(abs, A)))
    p > 0 || throw(ArgumentError("global_norm needs p > 0 or p = ±Inf; got p=$p."))
    # `float` first: `abs(x)^p` overflows for integer eltypes (Int8[100] gives 16).
    return global_allreduce(topology, sum(x -> abs(float(x))^p, A), +)^(1 / p)
end
global_norm(grid::Grid, A, p::Real = 2) = global_norm(grid.topology, A, p)
