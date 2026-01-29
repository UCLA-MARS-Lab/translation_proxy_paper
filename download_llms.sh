#!/bin/bash

# This script downloads all 14 LLMs used in the paper.
# WARNING: This will require a large amount of disk space.

set -e

MODELS=(
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "Qwen/Qwen3-30B-A3B"
    "microsoft/phi-4"
    "Qwen/Qwen2.5-72B"
    "meta-llama/Llama-3.3-70B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

echo "Starting download of 14 LLMs..."

for model in "${MODELS[@]}"; do
    echo "Downloading $model"
    hf download $model 
    echo "Finished $model"
done

echo "All models downloaded successfully."