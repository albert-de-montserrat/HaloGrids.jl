# GPU (CUDA) benchmark worker for ImplicitGlobalGrid.jl: same Cp-weighted 3D
# heat diffusion physics as bench_halogrids_diffusion3d.jl (its twin, see
# its header comment for the full rationale and T_eff read/write count),
# using the same ParallelStencil CUDA kernels as the HaloGrids worker. This
# is the blocking baseline: the full temperature update and wall kernels
# complete before `update_halo!`. Rank 0 saves one result record to the JLD2
# path in `ARGS[3]`.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA                # before ImplicitGlobalGrid, to activate its CUDA support (its own convention)
using ParallelStencil
@init_parallel_stencil(CUDA, Float64, 3)
using ImplicitGlobalGrid
import MPI
include(joinpath(@__DIR__, "metrics_common.jl"))

@parallel_indices (i, j, k) function flux_x!(qx, T, lam, dx)
    qx[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) / dx
    return nothing
end

@parallel_indices (i, j, k) function flux_y!(qy, T, lam, dy)
    qy[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) / dy
    return nothing
end

@parallel_indices (i, j, k) function flux_z!(qz, T, lam, dz)
    qz[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) / dz
    return nothing
end

@parallel_indices (i, j, k) function update_temperature!(T, Cp, qx, qy, qz, dt, dx, dy, dz)
    dTdt = (-(qx[i + 1, j, k] - qx[i, j, k]) / dx -
        (qy[i, j + 1, k] - qy[i, j, k]) / dy -
        (qz[i, j, k + 1] - qz[i, j, k]) / dz) / Cp[i + 1, j + 1, k + 1]
    T[i + 1, j + 1, k + 1] += dt * dTdt
    return nothing
end

@parallel_indices (i, j, k) function noflux_xlo!(T)
    T[i, j, k] = T[i + 1, j, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_xhi!(T)
    T[i, j, k] = T[i - 1, j, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_ylo!(T)
    T[i, j, k] = T[i, j + 1, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_yhi!(T)
    T[i, j, k] = T[i, j - 1, k]
    return nothing
end

function apply_vertical_noflux!(T, nx, ny, nz)
    @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
    @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
    @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
    @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
    return nothing
end

@views function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    output = ARGS[3]
    warmup = max(5, niter ÷ 10)

    lam, cp_min = 1.0, 1.0
    lx, ly, lz  = 10.0, 10.0, 10.0
    nx, ny, nz  = N, N, N
    hw = 1

    me, _, nprocs = init_global_grid(
        nx, ny, nz; periodx=0, periody=0, periodz=1, quiet=true)
    # IGG reads IGG_CUDAAWARE_MPI itself but never checks that MPI accepts
    # device buffers; fail here instead of crashing or corrupting halos.
    gpu_aware = env_flag("IGG_CUDAAWARE_MPI")
    gpu_aware && !MPI.has_cuda() && error(
        "IGG_CUDAAWARE_MPI=1 but MPI.has_cuda() is false for $(MPI.identify_implementation()); " *
        "with Cray MPICH set JULIA_MPI_HAS_CUDA=true and MPICH_GPU_SUPPORT_ENABLED=1.")
    me == 0 && println(
        "grid: local=($nx, $ny, $nz) global=($(nx_g()), $(ny_g()), $(nz_g())) nprocs=$nprocs")
    dx = lx / (nx_g() - 1)
    dy = ly / (ny_g() - 1)
    dz = lz / (nz_g() - 1)

    T  = CUDA.zeros(Float64, nx, ny, nz)
    Cp = CuArray([cp_min +
            5 * exp(-(x_g(i, dx, T) - lx / 1.5)^2 - (y_g(j, dy, T) - ly / 2)^2 - (z_g(k, dz, T) - lz / 1.5)^2) +
            5 * exp(-(x_g(i, dx, T) - lx / 3.0)^2 - (y_g(j, dy, T) - ly / 2)^2 - (z_g(k, dz, T) - lz / 1.5)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    T .= CuArray([
            100 * exp(-((x_g(i, dx, T) - lx / 2) / 2)^2 - ((y_g(j, dy, T) - ly / 2) / 2)^2 - ((z_g(k, dz, T) - lz / 3.0) / 2)^2) +
            50 * exp(-((x_g(i, dx, T) - lx / 2) / 2)^2 - ((y_g(j, dy, T) - ly / 2) / 2)^2 - ((z_g(k, dz, T) - lz / 1.5) / 2)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    apply_vertical_noflux!(T, nx, ny, nz)
    update_halo!(T)

    dt = min(dx * dx, dy * dy, dz * dz) * cp_min / lam / 8.1
    owned = (hw+1:nx-hw, hw+1:ny-hw, hw+1:nz-hw)
    total_energy() = MPI.Allreduce(Float64(sum(Cp[owned...] .* T[owned...])), MPI.SUM, MPI.COMM_WORLD)
    E0 = total_energy()

    qx = CUDA.zeros(Float64, nx - 1, ny - 2, nz - 2)
    qy = CUDA.zeros(Float64, nx - 2, ny - 1, nz - 2)
    qz = CUDA.zeros(Float64, nx - 2, ny - 2, nz - 1)
    function step!()
        # Explicit ranges (matching each qx/qy/qz's own, smaller-than-T shape)
        # are required here: `@parallel kernel(qx, T, ...)` without them infers
        # the launch size as the elementwise *max* over every array argument's
        # size (qx and T disagree in every dimension), which walks 1 past
        # qx/qy/qz's bounds and T's — surfaces on CUDA as a BoundsError deep in
        # kernel execution. See the CPU sibling for the same fix.
        @parallel (1:nx-1, 1:ny-2, 1:nz-2) flux_x!(qx, T, lam, dx)
        @parallel (1:nx-2, 1:ny-1, 1:nz-2) flux_y!(qy, T, lam, dy)
        @parallel (1:nx-2, 1:ny-2, 1:nz-1) flux_z!(qz, T, lam, dz)
        @parallel (1:nx-2, 1:ny-2, 1:nz-2) update_temperature!(T, Cp, qx, qy, qz, dt, dx, dy, dz)
        @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
        @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
        @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
        @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
        update_halo!(T)
        return nothing
    end

    for _ in 1:warmup
        step!()
    end
    MPI.Barrier(MPI.COMM_WORLD)
    CUDA.synchronize()

    E1 = total_energy()
    rel_drift = abs(E1 - E0) / abs(E0)
    rel_drift < 1e-6 || error(
        "diffusion3D (ImplicitGlobalGrid, GPU): energy drifted by a relative $rel_drift after " *
        "$warmup warmup steps (correctness gate failed; not a GPU-precision quirk at this tolerance).")

    GC.gc()
    stats = @timed for _ in 1:niter
        step!()
    end
    CUDA.synchronize()

    gpu_bytes = CUDA.@allocated for _ in 1:niter
        step!()
    end
    CUDA.synchronize()

    numel = nx * ny * nz
    teff = t_eff_gbs(1, 1, numel, sizeof(Float64), stats.time / niter)
    peakbw = me == 0 ? measure_peak_bandwidth_gbs() : 0.0

    if me == 0
        save_result(output, gpu_aware ? "ImplicitGlobalGridAware" : "ImplicitGlobalGrid", nprocs, N, niter, stats, gpu_bytes, teff, peakbw;
            transport=gpu_aware ? "device buffers" : "host-staged")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
