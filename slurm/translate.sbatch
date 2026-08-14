#!/bin/bash
#SBATCH --job-name=proxy-translate
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# GPUs / CPUs / memory are set per model by slurm/submit_all.sh (tp-matched).

# One translation job = one model.
# Required exports (set by submit_all.sh): MODEL=<name>, PROXY_REPO=<repo dir>
set -eu

PROXY_REPO="${PROXY_REPO:-$SLURM_SUBMIT_DIR}"
source "$PROXY_REPO/cluster/env.sh"

: "${MODEL:?Pass the model name with --export=ALL,MODEL=<name>,PROXY_REPO=...}"

proxy_activate main

echo "[translate] model=$MODEL node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$PROXY_REPO/translation"
python run_translation.py --model "$MODEL"
