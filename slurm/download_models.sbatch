#!/bin/bash
#SBATCH --job-name=proxy-download
#SBATCH --account=AIH4A_udutech
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Download Hugging Face models onto $SCRATCH with internet access.
#
# Leonardo login shells are memory-capped (OOM kills large hf downloads).
# lrd_all_serial runs on a login-class node via SLURM with a proper memory
# allocation (300G here) and up to 4h walltime. Request only 16G because
# Leonardo QOS caps per-user memory while GPU jobs are running; use
# max-workers=1 on tp=4 models to stay within RAM.
#
# Exports (set by slurm/submit_download.sh):
#   PROXY_REPO      - repo root
#   ONLY_MODELS     - optional comma-separated models.yaml "name" filter
#   DOWNLOAD_MAX_WORKERS - optional override (default set per model tp)
set -eu

PROXY_REPO="${PROXY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROXY_REPO"

echo "[download] node=$(hostname) mem=${SLURM_MEM_PER_NODE:-unset} ONLY_MODELS=${ONLY_MODELS:-all}"

if [ -n "${ONLY_MODELS:-}" ]; then
    IFS='|' read -r -a model_args <<< "$ONLY_MODELS"
    exec ./download_llms.sh "${model_args[@]}"
else
    exec ./download_llms.sh
fi
