module HaloGridsParallelStencilExt

using HaloGrids
using ParallelStencil

# The sent strip ends `overlaps - halowidths` cells into the kernel's range, so the
# outer slabs must reach that deep before packing (`halowidths` for default overlaps).
function boundary_width(args...)
    fields = map(HaloGrids.as_field, args)
    return ntuple(d -> maximum(HaloGrids.overlaps(f)[d] - HaloGrids.halowidths(f)[d] for f in fields), ndims(first(fields)))
end

function can_overlap(widths, ranges)
    length(widths) == length(ranges) || throw(ArgumentError("boundary_width must match the dimensionality of ranges"))
    all(w -> w isa Integer && w > 0, widths) || throw(ArgumentError("boundary_width entries must be positive integers"))
    all(r -> first(r) == 1, ranges) || throw(ArgumentError("ranges must start at 1"))
    return all(d -> length(ranges[d]) > 2 * widths[d], eachindex(ranges))
end

function hide_halo_communication_expr(args...)
    block = args[end]
    positional = collect(args[1:end-1])
    kw = findfirst(x -> x isa Expr && x.head == :(=) && x.args[1] == :boundary_width, positional)
    width = kw === nothing ? nothing : popat!(positional, kw).args[2]
    length(positional) >= 2 || error("expected grid, fields, and a block")
    grid, fields = positional[1], positional[2:end]
    statements = filter(x -> !(x isa LineNumberNode), block.args)
    launch, boundary_updates = statements[1], statements[2:end]
    launch isa Expr && launch.head == :macrocall && launch.args[1] == Symbol("@parallel") && length(launch.args) == 4 || error("first statement must be `@parallel ranges kernel!(...)`")
    ranges, kernel = launch.args[3:4]
    width_expr = width === nothing ? :($(GlobalRef(@__MODULE__, :boundary_width))($(fields...))) : width
    outer, inner, range, handle = gensym.((:outer, :inner, :range, :handle))
    widths, launch_ranges = gensym.((:widths, :ranges))
    partition_ranges = gensym(:partition_ranges)
    return esc(quote
        $widths = $width_expr
        $launch_ranges = $ranges
        if !$(GlobalRef(@__MODULE__, :can_overlap))($widths, $launch_ranges)
            ParallelStencil.@parallel $launch_ranges $kernel
            $(boundary_updates...)
            ParallelStencil.@synchronize
            HaloGrids.update_halo!($grid, $(fields...))
        else
            $partition_ranges = ParallelStencil.ParallelKernel.promote_ranges($launch_ranges)
            $outer = ParallelStencil.ParallelKernel.get_ranges_outer($widths, $partition_ranges)
            $inner = ParallelStencil.ParallelKernel.get_ranges_inner($widths, $partition_ranges)
            for $range in $outer
                ParallelStencil.@parallel $range $kernel
            end
            $(boundary_updates...)
            ParallelStencil.@synchronize
            $handle = HaloGrids.halo_exchange_start!($grid, $(fields...))
            try
                for $range in $inner
                    ParallelStencil.@parallel $range $kernel
                end
            finally
                HaloGrids.halo_exchange_finish!($handle)
            end
        end
    end)
end

end
