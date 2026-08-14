#!/bin/bash
# Submit model downloads on Leonardo's lrd_all_serial partition (internet + RAM).
#
# Usage:
#   ./slurm/submit_download.sh                          # all models in models.yaml
#   ./slurm/submit_download.sh Llama-3.3-70B-Instruct    # one model (by name)
#   ./slurm/submit_download.sh Llama-3.3-70B-Instruct Llama-4-Scout-17B-16E-Instruct
#
# Monitor:
#   squeue -u $USER
#   tail -f $SCRATCH/proxy_paper/slurm_logs/proxy-download_*.out

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"

ONLY_MODELS=""
if [ "$#" -gt 0 ]; then
    # Pipe-separated: sbatch --export misparses commas in values.
    ONLY_MODELS="$(IFS='|'; echo "$*")"
    # Validate names against models.yaml
    python - "$REPO_DIR/models.yaml" "$@" <<'PYEOF'
import sys, yaml
names = {m["name"] for m in yaml.safe_load(open(sys.argv[1]))["models"] if not m.get("skip", False)}
for arg in sys.argv[2:]:
    if arg not in names:
        raise SystemExit(f"Unknown or skipped model name: {arg}")
PYEOF
fi

job_id="$(sbatch --parsable \
    --job-name=proxy-download \
    --output="$PROXY_SLURM_LOGS/%x_%j.out" \
    --export=ALL,PROXY_REPO="$REPO_DIR",ONLY_MODELS="$ONLY_MODELS" \
    "$SCRIPT_DIR/download_models.sbatch")"

echo "[submit] download job $job_id (partition=lrd_all_serial, mem=16G)"
if [ -n "$ONLY_MODELS" ]; then
    echo "[submit] models: $ONLY_MODELS"
else
    echo "[submit] models: all (from models.yaml)"
fi
echo "[submit] log: $PROXY_SLURM_LOGS/proxy-download_${job_id}.out"
