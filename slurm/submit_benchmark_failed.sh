#!/bin/bash
# Submit benchmark retries for models that failed the Jul 9 production batch.
#
# Usage:
#   ./slurm/submit_benchmark_failed.sh              # all models below
#   ./slurm/submit_benchmark_failed.sh phi-4        # single model from the list
#
# Prerequisite: cancel any zombie jobs first, e.g.
#   scancel 48981545 48981546 48981547

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../cluster/env.sh"

# The 25 models still incomplete after the Jul 9-10 batches:
#   - TIMEOUT at the old 6h walltime (now raised to 24h + lm_eval --use_cache)
#   - watchdog false-kills (Falcon3-3B, granite-4.1-3b, Qwen3-1.7B)
#   - glm-4-9b-chat empty-stop crash (fixed via benchmark/lm_eval_run.py shim)
#   - DeepSeek-V2-Lite-Chat flashinfer RoPE JIT: replaced in models.yaml by the
#     dense deepseek-llm-7b-chat (no MLA -> no flashinfer nvcc JIT at load)
#   - ray data-parallel OOM/timeout/re-init failures (Falcon3-*, Mistral-7B,
#     Mixtral, Olmo-3*, Yi-1.5-*, granite-4.1-*, internlm2_5-*): fixed by the
#     dp=1 single-engine + bench_tp=4 whole-node path in run_benchmarks.sh
FAILED_MODELS=(
    aya-expanse-32b
    c4ai-command-r-08-2024
    DeepSeek-R1-Distill-Qwen-14B
    DeepSeek-R1-Distill-Qwen-32B
    deepseek-llm-7b-chat
    Falcon3-10B-Instruct
    Falcon3-3B-Instruct
    Falcon3-7B-Instruct
    gemma-4-31B-it
    glm-4-9b-chat
    granite-4.1-30b
    granite-4.1-3b
    internlm2_5-7b-chat
    internlm2_5-20b-chat
    Ministral-3-14B-Instruct-2512
    Mistral-7B-Instruct-v0.3
    Mixtral-8x7B-Instruct-v0.1
    Olmo-3-7B-Instruct
    Olmo-3.1-32B-Instruct
    Olmo-3-1125-32B
    phi-4
    Qwen3-1.7B
    Qwen3.6-27B
    Yi-1.5-9B-Chat
    Yi-1.5-34B-Chat
)

# Skip models that already have fresh results (e.g. accidental double-submit).
# Cutoff is after the last stale partial (glm-4 / Ministral-3-14B wrote partial
# results_*.json on Jul 9 ~18:00) but before the earliest good Jul 10 run, so
# those partials are correctly retried while genuine completions are skipped.
BENCH_RETRY_AFTER="${BENCH_RETRY_AFTER:-2026-07-10 00:00:00}"

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${FAILED_MODELS[@]}")
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[retry] no models to submit."
    exit 0
fi

submitted=0
skipped=0
for m in "${MODELS[@]}"; do
    if find "$PROXY_RESULTS_DIR/raw/$m" -name 'results_*.json' -newermt "$BENCH_RETRY_AFTER" 2>/dev/null | grep -q .; then
        echo "[retry] skip $m (fresh results_*.json exists)"
        skipped=$((skipped + 1))
        continue
    fi
    echo "[retry] benchmark $m"
    "$SCRIPT_DIR/submit_all.sh" benchmark "$m"
    submitted=$((submitted + 1))
done

echo "[retry] $submitted benchmark job(s) submitted, $skipped skipped."
