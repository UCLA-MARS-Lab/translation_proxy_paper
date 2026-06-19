#!/bin/bash

# This script downloads all LLMs used in the paper.
# WARNING: This will require a large amount of disk space.

set -e

# Use the high-throughput Rust downloader (pip install hf_transfer; in proxy_main.yml).
# Set HF_HOME to a local NVMe disk for best throughput.
export HF_HUB_ENABLE_HF_TRANSFER=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t MODELS < <(python -c "import yaml; d=yaml.safe_load(open('${SCRIPT_DIR}/models.yaml')); print('\n'.join([m['path'] for m in d['models']]))")

echo "Starting download of ${#MODELS[@]} LLMs..."

for model in "${MODELS[@]}"; do
    echo "Downloading $model"
    # --max-workers: more concurrent file connections per model.
    # --exclude: skip weights vLLM never loads (it uses safetensors),
    #            e.g. Llama-3.3-70B's ~130GB original/*.pth consolidated copy.
    hf download "$model" --max-workers 16 --exclude "original/*" "*.pth"
    echo "Finished $model"
done

echo "All models downloaded successfully."