#!/bin/bash

# This script downloads all LLMs used in the paper.
# WARNING: This will require a large amount of disk space.
#
# The model list is read from models.yaml (single source of truth) via yq.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t MODELS < <(python -c "import yaml; d=yaml.safe_load(open('${SCRIPT_DIR}/models.yaml')); print('\n'.join([m['path'] for m in d['models']]))")

echo "Starting download of ${#MODELS[@]} LLMs..."

for model in "${MODELS[@]}"; do
    echo "Downloading $model"
    hf download $model 
    echo "Finished $model"
done

echo "All models downloaded successfully."