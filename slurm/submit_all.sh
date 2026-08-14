#!/bin/bash
# Submit production jobs for the proxy-paper experiments on Leonardo.
#
# Usage:
#   ./slurm/submit_all.sh translate  [model_name]   # vLLM translation generation
#   ./slurm/submit_all.sh benchmark  [model_name]   # lm-eval 9-benchmark suite
#   ./slurm/submit_all.sh eval_mt                    # BLEU..SSA-COMET (array, 1 GPU/model)
#   ./slurm/submit_all.sh metricx                    # MetricX backfill (array, 1 GPU/model)
#
# translate/benchmark submit ONE JOB PER MODEL because the GPU request is
# tp-matched (SLURM arrays cannot vary gres per task):
#   tp=1 -> 1 GPU,  8 CPUs, 120G   (quarter of a boost node)
#   tp=2 -> 2 GPUs, 16 CPUs, 240G  (half node)
#   tp=4 -> 4 GPUs, 32 CPUs, 480G  (full node)
# Models marked skip: true in models.yaml are never submitted.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"

STAGE="${1:?Usage: $0 <translate|benchmark|eval_mt|metricx> [model_name]}"
ONLY_MODEL="${2:-}"

model_table() {
    python - "$REPO_DIR/models.yaml" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    for m in yaml.safe_load(f)["models"]:
        if not m.get("skip", False):
            print(f"{m['name']}|{m['tp']}")
PYEOF
}

resources_for_tp() {
    case "$1" in
        1) echo "--gres=gpu:1 --cpus-per-task=8 --mem=120G" ;;
        2) echo "--gres=gpu:2 --cpus-per-task=16 --mem=240G" ;;
        4) echo "--gres=gpu:4 --cpus-per-task=32 --mem=480G" ;;
        *) echo "ERROR: unsupported tp=$1 (expected 1, 2, or 4)" >&2; return 1 ;;
    esac
}

case "$STAGE" in
    translate|benchmark)
        SBATCH_FILE="$SCRIPT_DIR/$STAGE.sbatch"
        submitted=0
        while IFS='|' read -r name tp; do
            if [ -n "$ONLY_MODEL" ] && [ "$name" != "$ONLY_MODEL" ]; then
                continue
            fi
            if [ "$STAGE" = "benchmark" ]; then
                # Always take a full boost node so run_benchmarks.sh can run
                # data_parallel_size = 4/tp replicas across all 4 GPUs (speed
                # over compute allocation). tp=4 models simply become dp=1.
                res="--gres=gpu:4 --cpus-per-task=32 --mem=480G"
            else
                res="$(resources_for_tp "$tp")"
            fi
            echo "[submit] $STAGE $name (tp=$tp)"
            # shellcheck disable=SC2086
            sbatch $res \
                --job-name="proxy-$STAGE-$name" \
                --output="$PROXY_SLURM_LOGS/%x_%j.out" \
                --export=ALL,MODEL="$name",PROXY_REPO="$REPO_DIR" \
                "$SBATCH_FILE"
            submitted=$((submitted + 1))
        done < <(model_table)
        if [ "$submitted" -eq 0 ]; then
            echo "ERROR: no models matched (filter: '${ONLY_MODEL}')" >&2
            exit 1
        fi
        echo "[submit] $submitted $STAGE job(s) submitted."
        ;;
    eval_mt|metricx)
        N="$(model_table | wc -l)"
        echo "[submit] $STAGE array over $N models"
        sbatch --array="0-$((N - 1))" \
            --output="$PROXY_SLURM_LOGS/%x_%A_%a.out" \
            --export=ALL,PROXY_REPO="$REPO_DIR" \
            "$SCRIPT_DIR/$STAGE.sbatch"
        ;;
    *)
        echo "Usage: $0 <translate|benchmark|eval_mt|metricx> [model_name]" >&2
        exit 1
        ;;
esac
