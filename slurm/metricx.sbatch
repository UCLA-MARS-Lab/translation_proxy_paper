#!/bin/bash
#SBATCH --job-name=proxy-metricx
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

# Array job: one task per model, 1 GPU each (MetricX-24-XL fits a single
# A100 64GB). Submit via slurm/submit_all.sh metricx, which sets the --array
# range and exports PROXY_REPO. Run AFTER eval_mt has produced metrics CSVs.
set -eu

PROXY_REPO="${PROXY_REPO:-$SLURM_SUBMIT_DIR}"
source "$PROXY_REPO/cluster/env.sh"

# The default TMPDIR (/scratch_local, node-local disk) is frequently full on
# Leonardo compute nodes, which breaks bash here-doc temp files and the
# tempfile.TemporaryDirectory() used below for MetricX I/O ("No space left on
# device"). Redirect TMPDIR to our large scratch area. Safe here: metricx uses
# no Ray/vLLM AF_UNIX sockets, so the 107-byte socket-path limit that forces a
# short RAY_TMPDIR does not apply.
export TMPDIR="$PROXY_ROOT/tmp"
mkdir -p "$TMPDIR"

proxy_activate metricx

MODEL="$(python - "$PROXY_REPO/models.yaml" "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    models = [m for m in yaml.safe_load(f)["models"] if not m.get("skip", False)]
print(models[int(sys.argv[2])]["name"])
PYEOF
)"

echo "[metricx] model=$MODEL node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$PROXY_REPO/translation"
python evaluate_metricx.py --model "$MODEL"
