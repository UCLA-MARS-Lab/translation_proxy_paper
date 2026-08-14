#!/bin/bash
# Submit translate jobs for newly added models (not yet in results/).
#
# Usage:
#   ./slurm/submit_translate_new.sh              # all models below
#   ./slurm/submit_translate_new.sh glm-4-9b-chat   # single model from the list
#
# Run after downloads finish:
#   tail -f $SCRATCH/proxy_paper/slurm_logs/proxy-download_*.out

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEW_MODELS=(
    Llama-3.1-8B-Instruct
    Mistral-7B-Instruct-v0.3
    Mistral-Nemo-Instruct-2407
    Mixtral-8x7B-Instruct-v0.1
    DeepSeek-V2-Lite-Chat
    Yi-1.5-9B-Chat
    Yi-1.5-34B-Chat
    internlm2_5-7b-chat
    internlm2_5-20b-chat
    glm-4-9b-chat
)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${NEW_MODELS[@]}")
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[new] no models to submit."
    exit 0
fi

submitted=0
for m in "${MODELS[@]}"; do
    echo "[new] translate $m"
    "$SCRIPT_DIR/submit_all.sh" translate "$m"
    submitted=$((submitted + 1))
done

echo "[new] $submitted translate job(s) submitted."
