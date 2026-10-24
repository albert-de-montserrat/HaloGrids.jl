# Run in an environment with HaloGrids, MPI and the optional WriteVTK dependency.
# Writes a 3D Gaussian bump, distributed across however many MPI ranks the
# launcher gives it, as a parallel ParaView file set — open the .pvti in
# ParaView to hand-verify that save_pvtk's halo-inclusive overlap convention
# (see the pvtk I/O guide) really does stitch into one seamless volume with
# no gaps or double-counted cells at rank boundaries.
using HaloGrids
using WriteVTK
import MPI

MPI.Init()
try
    t = topology((0, 0, 0); periods = (false, false, false))
    global_shape = (64, 64, 64)
    all(d -> global_shape[d] % t.dims[d] == 0, 1:3) ||
        error("The process dimensions $(t.dims) must divide $global_shape.")
    hw = (1, 1, 1)
    nlocal = ntuple(d -> global_shape[d] ÷ t.dims[d] + 2 * hw[d], 3)

    # centered=true, origin_on_vertex=true puts global index 1 / nglobal at
    # the domain's two extreme vertices, symmetric about origin=(0,0,0) — so
    # the Gaussian below, centered at the physical origin, sits in the
    # middle of the domain regardless of how many ranks/dimensions it's split into.
    g = grid(t, nlocal; origin = (0.0, 0.0, 0.0), origin_on_vertex = true, centered = (true, true, true))
    f = field(zeros(nlocal); halowidths = hw)

    dx = (1.0, 1.0, 1.0)
    sigma = 8.0
    for ci in CartesianIndices(f.data)
        x = coord_g(g, f, 1, ci[1], dx[1])
        y = coord_g(g, f, 2, ci[2], dx[2])
        z = coord_g(g, f, 3, ci[3], dx[3])
        f.data[ci] = exp(-(x^2 + y^2 + z^2) / (2 * sigma^2))
    end
    # Not strictly needed here (the value at every cell, halo included, is a
    # pure function of its global position via coord_g, so neighboring
    # ranks already agree in the overlap), but this is the step a real
    # solver would need before save_pvtk, since it only ever updates its
    # own interior cells directly.
    update_halo!(g, f)

    outpath = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "gaussian3d")
    save_pvtk(g, f, outpath; name = "gaussian", dx = dx)

    MPI.Barrier(t.comm)
    t.rank == 0 && println("wrote $outpath.pvti on $(MPI.Comm_size(t.comm)) ranks, topology $(t.dims)")
finally
    MPI.Finalize()
end
