#!/bin/bash
# Central cluster configuration for running proxy_paper_runs on Leonardo (CINECA).
#
# Source this file from every login-node script and every SLURM job:
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Modes:
#     source env.sh            -> offline mode (default; for compute nodes)
#     PROXY_ONLINE=1 source env.sh -> online mode (for login-node prefetch)

# ----------------------------------------------------------------------------
# Paths (everything heavy lives on scratch)
# ----------------------------------------------------------------------------
# Always anchor to $SCRATCH/proxy_paper so sbatch --export=ALL cannot inherit a
# stale PROXY_ROOT / XDG_CACHE_HOME from the login shell (breaks offline JIT).
export PROXY_ROOT="${SCRATCH}/proxy_paper"
export PROXY_RESULTS_DIR="${PROXY_RESULTS_DIR:-$PROXY_ROOT/results}"
export PROXY_ENVS_DIR="$PROXY_ROOT/envs"
export PROXY_SLURM_LOGS="$PROXY_ROOT/slurm_logs"
export METRICX_REPO_DIR="${METRICX_REPO_DIR:-$PROXY_ROOT/metricx}"

# Repo root (this file lives in <repo>/cluster/)
export PROXY_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PROXY_ROOT" "$PROXY_RESULTS_DIR" "$PROXY_SLURM_LOGS"

# Isolate vLLM / flashinfer JIT caches on scratch (avoid inheriting unrelated
# XDG_CACHE_HOME from the user's shell, which breaks offline node compiles).
export XDG_CACHE_HOME="$PROXY_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"
# FlashInfer workspace (separate from XDG layout); keep on scratch.
export FLASHINFER_WORKSPACE_BASE="$PROXY_ROOT"
# Ray temp/session dir. MUST be short: ray creates AF_UNIX sockets under
# $RAY_TMPDIR/ray/session_<ts>/sockets/plasma_store, and the full socket path
# cannot exceed 107 bytes. A deep $SCRATCH path overflows that limit and ray
# fails with "AF_UNIX path length cannot exceed 107 bytes", so we use a short
# node-local /tmp path (sockets/logs are tiny; the plasma object store lives in
# /dev/shm, not here).
export RAY_TMPDIR="/tmp/ray_${USER}"
mkdir -p "$RAY_TMPDIR" 2>/dev/null || true

# NOTE: vLLM 0.11+ removed the V0 engine; vLLM 0.23 here runs the V1 engine
# unconditionally. VLLM_USE_V1 is now an unknown/no-op variable (vLLM warns
# "Unknown vLLM environment variable detected: VLLM_USE_V1"), so we do NOT set
# it. Engine-level robustness (enforce_eager, gpu_memory_utilization, capped
# max_model_len) is configured per-run in benchmark/run_benchmarks.sh instead.
# Avoid FlashInfer JIT compilation on offline compute nodes (exit 127: nvcc/ninja
# unavailable). Native PyTorch sampling preserves research decoding params.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Strip CUDA toolkit bins from PATH: stubs make shutil.which("nvcc") succeed but
# the binary is missing on compute nodes, which triggers broken FlashInfer JIT.
if [ -n "${PATH:-}" ]; then
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -vE '/opt/compilers/cuda/|/usr/local/cuda/bin' | paste -sd: -)"
    export PATH
fi

# ----------------------------------------------------------------------------
# Hugging Face / caches
# ----------------------------------------------------------------------------
export HF_HOME="$PROXY_ROOT/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# hf-xet (default in recent huggingface_hub) can fail on some large repos with
# "Unable to parse string as hex hash value"; fall back to plain HTTP downloads.
export HF_HUB_DISABLE_XET=1
export NLTK_DATA="$PROXY_ROOT/nltk_data"
export HF_DATASETS_TRUST_REMOTE_CODE=1
# sacrebleu downloads tokenizer files on first flores200 BLEU call unless cached.
export SACREBLEU="$PROXY_ROOT/sacrebleu"
mkdir -p "$HF_HOME" "$NLTK_DATA" "$SACREBLEU/models"

# ----------------------------------------------------------------------------
# Online/offline switching (Leonardo compute nodes have NO internet access)
# ----------------------------------------------------------------------------
if [ "${PROXY_ONLINE:-0}" = "1" ]; then
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_EVALUATE_OFFLINE
else
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_EVALUATE_OFFLINE=1
fi

# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
# Do NOT load the cluster cuda module: it prepends CUDA stub libraries
# (including a stub libnvidia-ml.so) to LD_LIBRARY_PATH, which breaks NVML
# and vLLM's GPU platform detection ("Device string must not be empty").
# The pip-installed torch/vllm wheels bundle their own CUDA runtime libs.
if command -v module >/dev/null 2>&1; then
    module unload cuda 2>/dev/null || true
fi
unset CUDA_HOME CUDA_PATH
# Belt-and-braces: strip any toolkit CUDA lib dirs (with NVML stubs) that may
# have been inherited through sbatch/srun's exported environment.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -v '/opt/compilers/cuda/' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

# ----------------------------------------------------------------------------
# Conda env helpers (envs are created by cluster/setup_envs.sh as prefix envs)
# ----------------------------------------------------------------------------
export PROXY_MAIN_ENV="$PROXY_ENVS_DIR/proxy_main"
export PROXY_COMET_ENV="$PROXY_ENVS_DIR/proxy_comet"
export PROXY_METRICX_ENV="$PROXY_ENVS_DIR/proxy_metricx"

proxy_activate() {
    # usage: proxy_activate <main|comet|metricx>
    local env_path
    case "$1" in
        main)    env_path="$PROXY_MAIN_ENV" ;;
        comet)   env_path="$PROXY_COMET_ENV" ;;
        metricx) env_path="$PROXY_METRICX_ENV" ;;
        *) echo "proxy_activate: unknown env '$1' (use main|comet|metricx)" >&2; return 1 ;;
    esac
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
    conda activate "$env_path"
}
