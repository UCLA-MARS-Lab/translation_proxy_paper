#!/bin/bash
#SBATCH --job-name=proxy-eval-mt
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

# Array job: one task per model, 1 GPU each (XCOMET-XL and SSA-COMET both fit
# a single A100 64GB). Submit via slurm/submit_all.sh eval_mt, which sets the
# --array range and exports PROXY_REPO.
set -eu

PROXY_REPO="${PROXY_REPO:-$SLURM_SUBMIT_DIR}"
source "$PROXY_REPO/cluster/env.sh"

# The default TMPDIR (/scratch_local, node-local disk) is frequently full on
# Leonardo compute nodes, which breaks bash here-doc temp files ("No space left
# on device"). Redirect TMPDIR to our large scratch area. Safe here: eval_mt
# uses no Ray/vLLM AF_UNIX sockets, so the 107-byte socket-path limit that
# forces a short RAY_TMPDIR does not apply.
export TMPDIR="$PROXY_ROOT/tmp"
mkdir -p "$TMPDIR"

# Reduce CUDA allocator fragmentation. XCOMET-XL and SSA-COMET both reside on a
# single A100 64GB, leaving little headroom; without this, longer language pairs
# OOM and the fragmentation cascades into many subsequent pairs (as recommended
# by the torch OOM error itself). Score-neutral.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

proxy_activate comet

MODEL="$(python - "$PROXY_REPO/models.yaml" "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    models = [m for m in yaml.safe_load(f)["models"] if not m.get("skip", False)]
print(models[int(sys.argv[2])]["name"])
PYEOF
)"

echo "[eval_mt] model=$MODEL node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$PROXY_REPO/translation"
python evaluate_mt.py --model "$MODEL"
