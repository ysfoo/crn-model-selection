# Process start-up for the SMC drivers (insect/SMC.jl, pinene/SMC.jl): worker processes for `parallel = :distributed`,
# thread pinning for `parallel = :threads` (and :serial), and single-threaded BLAS. Call `setup_SMC_process()` once, near
# the top of the driver. Workers re-run the driver script, so they call it too; there it only sets BLAS.

using Distributed: @everywhere, addprocs, myid, nprocs, workers
using LinearAlgebra
using ThreadPinning

# Starts `n` worker processes for `parallel = :distributed` and makes each of them run `script` (by default the script
# the master is running) with the master's command-line arguments, so that every definition in the problem script (prior
# path, likelihood builder, move function, ...) also exists on the workers. `run_SMC` and `resume_SMC` return at once on
# a worker, so only the master runs SMC; anything else the script does at top level runs on every process, so it must be
# safe to repeat (e.g. `mkpath`, not copying or deleting files). Workers have one thread each.
# Call this before pinning threads: worker processes inherit the CPU affinity of the master thread that starts them.
# It does nothing on a worker, when `n == 0` or when workers already exist. `n` defaults to the environment variable
# SMC_NWORKERS (unset: 0) and `heap_size_hint` (per worker) to SMC_WORKER_HEAP (unset: 2G).
function start_SMC_workers(;
    n=parse(Int, get(ENV, "SMC_NWORKERS", "0")), heap_size_hint=get(ENV, "SMC_WORKER_HEAP", "2G"),
    script=PROGRAM_FILE,
)
    (myid() == 1 && n > 0 && nprocs() == 1) || return nothing
    isempty(script) && error("start_SMC_workers needs the path of the problem script (there is none in an interactive session)")
    script_path = abspath(script)
    args = copy(ARGS)
    t = @elapsed begin
        addprocs(n; exeflags=`--project=$(Base.active_project()) --threads=1 --heap-size-hint=$heap_size_hint`)
        @everywhere workers() begin
            empty!(ARGS)
            append!(ARGS, $args)
            include($script_path)
        end
    end
    @info "Started $n worker processes running $script_path in $(round(t; digits=1)) s"
    flush(stdout)
    flush(stderr)
    return nothing
end

isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""

# Start the workers for `parallel = :distributed` (none unless SMC_NWORKERS is set; see `start_SMC_workers`), then pin
# the threads of a process without workers, then use one BLAS thread. Workers are started before pinning because they
# inherit the CPU affinity of the master thread, and are left unpinned: pinning the single thread of every worker to the
# first allowed CPU would put them all on one core. A run without workers (`parallel = :threads` or `:serial`) is pinned
# to the job's CPUs under SLURM and to the first cores otherwise.
function setup_SMC_process()
    start_SMC_workers()
    pin_threads_without_workers()
    LinearAlgebra.BLAS.set_num_threads(1)
    return nothing
end

# Pin the threads of the master process if it has no workers (`parallel = :threads` or `:serial`): to the job's CPUs under
# SLURM, to the first cores otherwise. With workers it does nothing, and it does nothing on a worker. Call it after any
# workers have been started (they inherit the master thread's CPU affinity). Also used by insect/MCMC_chains.jl.
function pin_threads_without_workers()
    if myid() == 1 && nprocs() == 1
        isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores)
    end
    return nothing
end
