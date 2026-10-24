# Shared metric-collection helpers for benchmarks/gpu/*.jl. `include`d (not
# `using`d as a module) after `using CUDA, MPI` in each worker script — same
# include-into-caller's-namespace pattern HaloGrids.jl's own CUDA extension
# uses (src/CUDAExt/shared.jl + halo_exchange.jl), so `CUDA.*` below resolves
# to whatever the including script already `using CUDA`'d.

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

# Device-to-device copyto! bandwidth probe: measures *this* GPU's realized
# memory bandwidth directly, so "% of peak" in the results is grounded in an
# actual measurement on the hardware the benchmark ran on, not a datasheet
# figure — matches this project's "measured, don't guess" rule (see
# HaloGrids.jl/REDESIGN_PLAN.md). Buffer is ~1 GiB of Float64, well past any
# GPU's L2 cache, so the copy is genuinely memory-bound regardless of how
# small the actual problem size under test is.
function measure_peak_bandwidth_gbs(; nrep::Integer=20)
    n = 134_217_728  # 2^27 Float64 elements = 1 GiB
    src = CUDA.rand(Float64, n)
    dst = CUDA.zeros(Float64, n)
    for _ in 1:3  # warmup
        copyto!(dst, src)
    end
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:nrep
        copyto!(dst, src)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) / 1e9
    bytes_moved = 2 * n * sizeof(Float64) * nrep  # one read + one write per copyto!
    return bytes_moved / 1e9 / elapsed
end

using JLD2

# Boolean environment switch (`1`/`0`/`true`/`false`); anything else is an error
# so that a typo cannot silently select the wrong transport.
function env_flag(name::AbstractString)
    v = get(ENV, name, "0")
    b = tryparse(Bool, v)
    b === nothing && error("environment variable $name must be 0/1/true/false; got \"$v\"")
    return b
end

# Repetition index of this run within a SLURM matrix (`BENCH_REP`, default 1).
bench_rep() = parse(Int, get(ENV, "BENCH_REP", "1"))

function save_result(path, pkg, nprocs, N, niter, stats, gpu_bytes, teff, peakbw;
                     transport::AbstractString = "", rep::Integer = bench_rep())
    gs = stats.gcstats
    pct = peakbw > 0 ? 100 * teff / peakbw : NaN
    result = (; pkg=String(pkg), nprocs, N, niter, time_s=stats.time,
        bytes=stats.bytes, gctime_s=stats.gctime, pauses=gs.pause,
        full_sweep=gs.full_sweep, gpu_bytes, t_eff_gbs=teff,
        peak_bw_gbs=peakbw, pct_of_peak=pct, transport=String(transport), rep=Int(rep))
    key = "result__$(pkg)__n$(nprocs)__N$(N)__r$(rep)"
    jldopen(path, "a+") do file
        file[key] = result
    end
    println("Saved $key to $path")
    return result
end
