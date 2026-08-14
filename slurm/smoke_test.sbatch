#!/bin/bash
#SBATCH --job-name=proxy-smoke
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

# End-to-end smoke test on the debug QOS (30 min, 1 GPU), split in two steps
# because the full chain doesn't fit one debug window:
#   STEP=generate : vLLM translation (small model, 1 language x 3 corpora)
#                   + a bounded lm-eval run (offline benchmark path)
#   STEP=metrics  : evaluate_mt.py (BLEU..SSA-COMET) + evaluate_metricx.py
#                   on the translations produced by STEP=generate
#
# Submit both with slurm/submit_smoke.sh (chains them with a dependency).
# Exports: STEP, PROXY_REPO; optional SMOKE_MODEL, SMOKE_LANGS.
set -eu

PROXY_REPO="${PROXY_REPO:-$SLURM_SUBMIT_DIR}"
source "$PROXY_REPO/cluster/env.sh"

STEP="${STEP:?Set STEP=generate or STEP=metrics}"
SMOKE_MODEL="${SMOKE_MODEL:-Llama-3.2-1B-Instruct}"
# One language hit by all 3 corpora (FLORES, NTREX, WMT24) exercises every
# code path while keeping MetricX (batch_size=1) inside the debug walltime.
SMOKE_LANGS="${SMOKE_LANGS:-fra}"

echo "[smoke:$STEP] model=$SMOKE_MODEL langs=$SMOKE_LANGS node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"

case "$STEP" in
    generate)
        proxy_activate main

        python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(f"FATAL: CUDA unavailable (is_available={torch.cuda.is_available()}, "
          f"count={torch.cuda.device_count()})", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: {torch.cuda.device_count()} device(s), "
      f"{torch.cuda.get_device_name(0)}")
PYEOF

        echo "--- [smoke] Translation generation ---"
        cd "$PROXY_REPO/translation"
        python run_translation.py --model "$SMOKE_MODEL" --langs "$SMOKE_LANGS"

        echo "--- [smoke] Bounded lm-eval benchmark run (mgsm_direct, limit 8) ---"
        cd "$PROXY_REPO"
        BENCH_TASKS="mgsm_direct" BENCH_EXTRA_ARGS="--limit 8" \
            bash benchmark/run_benchmarks.sh "$SMOKE_MODEL"

        # Fail the job if translation outputs were not produced.
        for corpus in flores-200 ntrex wmt24; do
            f="$PROXY_RESULTS_DIR/translations/${corpus}/${SMOKE_MODEL}/eng-fra.csv"
            if [ ! -s "$f" ]; then
                echo "FATAL: missing translation output: $f" >&2
                exit 1
            fi
        done
        ;;
    metrics)
        echo "--- [smoke] Standard MT metrics (proxy_comet) ---"
        proxy_activate comet
        cd "$PROXY_REPO/translation"
        python evaluate_mt.py --model "$SMOKE_MODEL"

        echo "--- [smoke] MetricX backfill (proxy_metricx) ---"
        proxy_activate metricx
        python evaluate_metricx.py --model "$SMOKE_MODEL"

        # Verify all 7 metrics were written for each corpus.
        for corpus in flores-200 wmt24 ntrex; do
            csv="$PROXY_RESULTS_DIR/metrics/${SMOKE_MODEL}/${corpus}.csv"
            if [ ! -s "$csv" ]; then
                echo "FATAL: missing metrics CSV: $csv" >&2
                exit 1
            fi
            python - <<PYEOF
import pandas as pd, sys
df = pd.read_csv("$csv")
cols = ["bleu","chrf++","rouge-l","meteor","xcomet","ssa-comet","metricx"]
missing = [c for c in cols if c not in df.columns or df[c].isna().all()]
if missing:
    print(f"FATAL: {missing} empty in $csv", file=sys.stderr)
    sys.exit(1)
print(f"OK: $csv has all 7 metrics")
PYEOF
        done
        ;;
    *)
        echo "Unknown STEP='$STEP' (use generate|metrics)" >&2
        exit 1
        ;;
esac

echo "[smoke:$STEP] done."
