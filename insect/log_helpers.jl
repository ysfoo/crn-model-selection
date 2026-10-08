# Progress logging for scripts that run models in parallel over threads. The main log (stdout, redirected by the
# slurm script) gets one timestamped line when a model starts and one when it finishes; detailed output of a model
# can go to its own file, since stdout is shared by all threads but loggers are task-local.
using Dates, Logging

const LOG_LOCK = ReentrantLock()
const FAILED_MODELS = Int[]

function logmsg(msg)
    lock(LOG_LOCK) do
        println(stdout, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", msg)
        flush(stdout)
    end
end

# Runs `f(io)` for one model and logs its start and end to the main log, where `counter` (a `Threads.Atomic{Int}`)
# counts finished models out of `n_total`. With `logdir`, `io` and the logger (`@info` etc.) write to
# `logdir/model[model_idx].log`, otherwise `io` is `stdout` and `f` should not print. `summary(result)` is appended
# to the line logged at the end. If `f` throws, the error is logged (stack trace in the model log), the model is
# recorded in `FAILED_MODELS` and `nothing` is returned, so that the other models carry on.
function with_model_log(f, model_idx, counter, n_total; logdir=nothing, summary=(_)->"")
    logmsg("model $model_idx start (thread $(Threads.threadid()))")
    t0 = time()
    run(io) = try
        f(io)
    catch e
        e isa InterruptException && rethrow()
        io === stdout || (showerror(io, e, catch_backtrace()); println(io); flush(io))
        lock(LOG_LOCK) do; push!(FAILED_MODELS, model_idx) end
        logmsg("model $model_idx FAILED: $(sprint(showerror, e))")
        nothing
    end
    result = if isnothing(logdir)
        run(stdout)
    else
        open(joinpath(logdir, "model$(model_idx).log"), "w") do io
            with_logger(ConsoleLogger(io)) do
                run(io)
            end
        end
    end
    k = Threads.atomic_add!(counter, 1) + 1
    if !isnothing(result)
        s = summary(result)
        logmsg("model $model_idx done, $(round((time() - t0)/60; digits=2)) min ($k/$n_total)" * (isempty(s) ? "" : ", " * s))
    end
    return result
end

function log_failures()
    isempty(FAILED_MODELS) ? logmsg("all models finished") : logmsg("FAILED models: $(sort(FAILED_MODELS))")
end
