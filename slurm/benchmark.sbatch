#!/bin/bash
#SBATCH --job-name=proxy-benchmark
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=boost_usr_prod
# Backstop walltime. The full 743-task suite (570k generate_until + 2.28M
# loglikelihood requests) does not fit in 6h for mid/large models, so we take
# the Leonardo boost max (24h). lm_eval --use_cache (see run_benchmarks.sh) makes
# any remaining timeout recoverable: a resubmit resumes completed requests.
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# GPUs / CPUs / memory are set per model by slurm/submit_all.sh (tp-matched).

# One benchmark job = one model (all 9 lm-eval suites).
# Required exports (set by submit_all.sh): MODEL=<name>, PROXY_REPO=<repo dir>
set -eu

PROXY_REPO="${PROXY_REPO:-$SLURM_SUBMIT_DIR}"
source "$PROXY_REPO/cluster/env.sh"

: "${MODEL:?Pass the model name with --export=ALL,MODEL=<name>,PROXY_REPO=...}"

proxy_activate main

echo "[benchmark] model=$MODEL node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ----------------------------------------------------------------------------
# Stall watchdog: vLLM's V1 engine can deadlock mid-inference (GPU 0% util,
# VRAM pinned, no log output) and otherwise squats the node until walltime.
# We run the benchmark in its own process group and kill it if the per-model
# log stops advancing for BENCH_STALL_TIMEOUT seconds, so the job FAILS fast
# and can be retried by slurm/submit_benchmark_failed.sh.
# ----------------------------------------------------------------------------
# 6h of no log activity. lm-eval runs the whole loglikelihood pass (2.28M
# requests) BEFORE generate_until, then does a single-threaded, tqdm-silent
# pass turning those 2.28M results into per-doc metrics. On some models that
# silent stretch runs past 2h and the old 7200s window false-killed healthy
# jobs at loglikelihood-100% (Falcon3-3B/10B, Mistral-7B) even though they were
# still aggregating, not deadlocked. 6h keeps the guard against true GPU
# deadlocks (0% util, VRAM pinned) while tolerating the silent aggregation.
STALL_TIMEOUT="${BENCH_STALL_TIMEOUT:-21600}"
LOG_FILE="$PROXY_RESULTS_DIR/raw/$MODEL/raw_log.txt"

setsid bash "$PROXY_REPO/benchmark/run_benchmarks.sh" "$MODEL" &
RUN_PID=$!

watchdog() {
    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep 60
        [ -f "$LOG_FILE" ] || continue
        last=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$(( $(date +%s) - last ))" -gt "$STALL_TIMEOUT" ]; then
            echo "[watchdog] no benchmark log activity for >${STALL_TIMEOUT}s on $MODEL; assuming vLLM deadlock -> killing run." >&2
            kill -TERM -- "-$RUN_PID" 2>/dev/null || true
            sleep 10
            kill -KILL -- "-$RUN_PID" 2>/dev/null || true
            return 0
        fi
    done
}
watchdog &
WATCH_PID=$!

RC=0
wait "$RUN_PID" || RC=$?
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true

if [ "$RC" -ne 0 ]; then
    echo "[benchmark] run for $MODEL exited with code $RC (watchdog-killed deadlock or lm_eval error)." >&2
fi
exit "$RC"
