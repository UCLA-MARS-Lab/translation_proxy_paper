#!/bin/bash

# This script downloads all LLMs used in the paper.
# WARNING: This will require a large amount of disk space.
#
# The model list is read from models.yaml (single source of truth) via yq.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read the HF repo path of every model from models.yaml
mapfile -t MODELS < <(yq '.models[].path' "${SCRIPT_DIR}/models.yaml")

echo "Starting download of ${#MODELS[@]} LLMs..."

for model in "${MODELS[@]}"; do
    echo "Downloading $model"
    hf download $model 
    echo "Finished $model"
done

echo "All models downloaded successfully."