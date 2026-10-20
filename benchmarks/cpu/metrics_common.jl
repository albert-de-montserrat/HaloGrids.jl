# Shared metric-collection helpers for benchmarks/cpu/*.jl. `include`d (not
# `using`d as a module) after `using MPI` in each worker script, same
# include-into-caller's-namespace pattern the gpu/ sibling uses.

# T_eff convention: the standard ParallelStencil.jl-ecosystem "effective
# memory throughput" metric — A_eff (GB) = (2*n_readwrite + n_readonly) * numel *
# sizeof(T) / 1e9, measured against the *ideal* minimal fused-kernel memory
# traffic (not the actual traffic of this repo's deliberately non-fused,
# readable multi-array implementations), T_eff = A_eff / t_step (GB/s).
# n_readwrite counts arrays an ideal single kernel reads and writes once each
# (an update A -> Anew counts as one), n_readonly arrays it only reads. Both
# are documented at
# each call site, since it requires domain judgement about the specific
# kernel — see the header comment in each bench_*.jl worker for its count.
t_eff_gbs(n_readwrite::Integer, n_readonly::Integer, numel::Integer, elemsize::Integer, t_step::Real) =
    (2 * n_readwrite + n_readonly) * numel * elemsize / 1e9 / t_step

# Host memory copyto! bandwidth probe: measures *this* machine's realized
# memory bandwidth directly, so "% of peak" in the results is grounded in an
# actual measurement on the hardware the benchmark ran on, not a datasheet
# figure — matches this project's "measured, don't guess" rule (see
# HaloGrids.jl/REDESIGN_PLAN.md), and mirrors gpu/metrics_common.jl's
# device-to-device probe with the same methodology. Buffer is ~1 GiB of
# Float64, well past any CPU's last-level cache, so the copy is genuinely
# memory-bound regardless of how small the actual problem size under test is.
# The copy is split across `Threads.nthreads()` chunks so the measurement
# reflects the same thread count the ParallelStencil `Threads` backend uses
# for the benchmarked kernels themselves, rather than a single-core figure.
function measure_peak_bandwidth_gbs(; nrep::Integer=20)
    n = 134_217_728  # 2^27 Float64 elements = 1 GiB
    src = rand(Float64, n)
    dst = zeros(Float64, n)
    nthreads = max(1, Threads.nthreads())
    chunk = cld(n, nthreads)
    function threaded_copy!()
        Threads.@threads for t in 1:nthreads
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, n)
            lo > hi && continue
            @inbounds copyto!(dst, lo, src, lo, hi - lo + 1)
        end
        return nothing
    end
    for _ in 1:3  # warmup
        threaded_copy!()
    end
    t0 = time_ns()
    for _ in 1:nrep
        threaded_copy!()
    end
    elapsed = (time_ns() - t0) / 1e9
    bytes_moved = 2 * n * sizeof(Float64) * nrep  # one read + one write per copy
    return bytes_moved / 1e9 / elapsed
end

using JLD2

# `gpu_bytes` keeps its gpu/-sibling name here (rather than being renamed to
# something CPU-specific) so both directories' JLD2 records share one
# schema — visualize.jl (and any future combined cpu-vs-gpu analysis) reads
# either without special-casing. On CPU it's still a real, independent
# measurement (via `@allocated` at each call site, mirroring gpu/'s
# `CUDA.@allocated`), not a copy of `bytes`: `bytes` comes from `@timed`
# wrapping the *timed* loop, `gpu_bytes` from a separate `@allocated`-wrapped
# loop run afterward — see each bench_*.jl's header comment for why both are
# kept even though, on CPU, they measure the same GC heap and should closely
# agree (that agreement is itself a mild sanity check).
function save_result(path, pkg, nprocs, N, niter, stats, gpu_bytes, teff, peakbw)
    gs = stats.gcstats
    pct = peakbw > 0 ? 100 * teff / peakbw : NaN
    result = (; pkg=String(pkg), nprocs, N, niter, time_s=stats.time,
        bytes=stats.bytes, gctime_s=stats.gctime, pauses=gs.pause,
        full_sweep=gs.full_sweep, gpu_bytes, t_eff_gbs=teff,
        peak_bw_gbs=peakbw, pct_of_peak=pct)
    key = "result__$(pkg)__n$(nprocs)__N$(N)"
    jldopen(path, "a+") do file
        file[key] = result
    end
    println("Saved $key to $path")
    return result
end
