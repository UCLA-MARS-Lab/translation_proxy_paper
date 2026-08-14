#!/bin/bash
# Run the 9 multilingual benchmarks with lm-eval (vLLM backend).
#
# Usage:
#   ./run_benchmarks.sh              # run all (non-skipped) models sequentially
#   ./run_benchmarks.sh <model_name> # run a single model from models.yaml
#
# GPU placement is handled by the scheduler (SLURM sets CUDA_VISIBLE_DEVICES);
# the tensor-parallel size for each model is read from models.yaml.

set -euo pipefail

MODEL_FILTER="${1:-}"

# Environment Variables
export HF_DATASETS_TRUST_REMOTE_CODE=1
# VLLM_USE_V1 is inert on vllm>=0.11 (V0 removed); kept for documentation only.
export VLLM_USE_V1=0
# Reduce CUDA allocator fragmentation so the large float32 log_softmax buffer in
# lm-eval's loglikelihood pass can be allocated without OOM (recommended by the
# torch OOM error itself).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Results root: scratch on the cluster, ./results locally.
RESULTS_DIR="${PROXY_RESULTS_DIR:-./results}"

# Benchmarks
AFRI_MMLU="afrimmlu-irokobench"
AFRI_XNLI="afrixnli-irokobench"
BELEBELE="belebele"
GLOBAL_MMLU="global_mmlu_ar,global_mmlu_bn,global_mmlu_de,global_mmlu_en,global_mmlu_fr,global_mmlu_hi,global_mmlu_id,global_mmlu_it,global_mmlu_ja,global_mmlu_ko,global_mmlu_pt,global_mmlu_es,global_mmlu_sw,global_mmlu_yo,global_mmlu_zh"
HELLA_SWAG="hellaswag_multilingual"
TRUTHFUL_QA="truthfulqa_multilingual"
MGSM="mgsm_direct"
MLQA="mlqa_en_ar,mlqa_en_de,mlqa_en_en,mlqa_en_es,mlqa_en_hi,mlqa_en_vi,mlqa_en_zh"
INCLUDE="include_base_44_albanian,include_base_44_arabic,include_base_44_armenian,include_base_44_azerbaijani,include_base_44_basque,include_base_44_belarusian,include_base_44_bengali,include_base_44_bulgarian,include_base_44_chinese,include_base_44_croatian,include_base_44_dutch,include_base_44_estonian,include_base_44_finnish,include_base_44_french,include_base_44_georgian,include_base_44_german,include_base_44_greek,include_base_44_hebrew,include_base_44_hindi,include_base_44_hungarian,include_base_44_indonesian,include_base_44_italian,include_base_44_japanese,include_base_44_kazakh,include_base_44_korean,include_base_44_lithuanian,include_base_44_malay,include_base_44_malayalam,include_base_44_nepali,include_base_44_north macedonian,include_base_44_persian,include_base_44_polish,include_base_44_portuguese,include_base_44_russian,include_base_44_serbian,include_base_44_spanish,include_base_44_tagalog,include_base_44_tamil,include_base_44_telugu,include_base_44_turkish,include_base_44_ukrainian,include_base_44_urdu,include_base_44_uzbek,include_base_44_vietnamese"

# Combine into one giant task list
ALL_TASKS="${AFRI_MMLU},${AFRI_XNLI},${BELEBELE},${GLOBAL_MMLU},${HELLA_SWAG},${TRUTHFUL_QA},${MGSM},${MLQA},${INCLUDE}"

# Allow the caller (e.g. the smoke test) to override the task list and add
# extra lm-eval flags without changing the production defaults.
TASKS="${BENCH_TASKS:-$ALL_TASKS}"
EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

# Used by cluster/prefetch.sh to download all benchmark datasets on the login
# node without duplicating the task list.
if [ "${BENCH_PRINT_TASKS:-0}" = "1" ]; then
    echo "$TASKS"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read model config from models.yaml (mirrors run_translation.py vLLM settings).
mapfile -t MODELS < <(python - "$SCRIPT_DIR/../models.yaml" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    for m in yaml.safe_load(f)["models"]:
        if not m.get("skip", False):
            trc = "true" if m.get("trust_remote_code") else "false"
            mml = m.get("max_model_len", 4096)
            ee = "true" if m.get("enforce_eager") else "false"
            # bench_tp: tensor-parallel size to use for the benchmark suite only.
            # Independent of `tp` (which sizes translation). Defaults to `tp`.
            btp = m.get("bench_tp", m["tp"])
            print(f"{m['name']}|{m['path']}|{m['tp']}|{trc}|{mml}|{ee}|{btp}")
PYEOF
)

if [ -n "$MODEL_FILTER" ]; then
    found=0
    for entry in "${MODELS[@]}"; do
        [ "${entry%%|*}" = "$MODEL_FILTER" ] && found=1
    done
    if [ "$found" -eq 0 ]; then
        echo "ERROR: model '$MODEL_FILTER' not found (or marked skip) in models.yaml" >&2
        exit 1
    fi
fi

echo "Starting Evaluation..."

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path tp_size trust_remote_code max_model_len enforce_eager bench_tp <<< "$entry"

    if [ -n "$MODEL_FILTER" ] && [ "$model_name" != "$MODEL_FILTER" ]; then
        continue
    fi

    BASE_OUTPUT_DIR="${RESULTS_DIR}/raw/${model_name}"
    LOG_FILE="${BASE_OUTPUT_DIR}/raw_log.txt"

    mkdir -p "$BASE_OUTPUT_DIR"

    # GPU count the job actually holds (SLURM sets CUDA_VISIBLE_DEVICES).
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
    else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    [ "${NUM_GPUS:-0}" -ge 1 ] || NUM_GPUS=1

    # Effective tensor-parallel size for the benchmark = bench_tp (per model,
    # defaults to tp), capped at the GPUs we hold. We use a SINGLE persistent
    # vLLM engine spanning these GPUs and NO ray data-parallelism (dp=1).
    #
    # Why dp=1: lm-eval's data_parallel_size>1 path (vllm_causallms._model_generate)
    # is a ray fan-out that (a) constructs a *fresh* LLM inside each ray actor on
    # every phase and tears it down with ray.shutdown() between the generate_until
    # and loglikelihood passes, and (b) collects every RequestOutput back into the
    # driver via a single ray.get(). On this 743-task suite that path repeatedly
    # died three ways: ray raylet/GCS "timed out during startup" on the
    # cross-phase re-init (Olmo-3-7B, granite-4.1-3b, Olmo-3.1-32B, Yi, internlm),
    # "EngineCore failed to start" on the loglikelihood re-load, and host-RAM
    # OOM-kills from N model replicas + the 2.28M-output ray.get (Falcon3-*,
    # Mistral-7B, internlm2_5-7b, Olmo-3-1125-32B). A single persistent engine
    # sharded with tensor parallelism uses all held GPUs, never re-inits between
    # phases, and streams results without the ray collection, so it is reliable.
    eff_tp="${bench_tp:-$tp_size}"
    [ "$eff_tp" -le "$NUM_GPUS" ] || eff_tp="$NUM_GPUS"
    [ "$eff_tp" -ge 1 ] || eff_tp=1
    dp=1

    # enforce_eager + capped max_model_len keep the vLLM V1 engine off the
    # torch.compile/cudagraph path and prevent the KV-cache thrash/deadlock that
    # froze earlier full-context runs.
    #
    # gpu_memory_utilization is deliberately conservative (0.70): lm-eval's
    # loglikelihood pass makes vLLM compute log_softmax over the full vocab in
    # float32, which needs several GiB of headroom on top of the KV cache. At
    # 0.9 large-vocab models (aya/command-r 256k, glm-4, Qwen3.6, ...) OOM'd in
    # sampler.compute_logprobs.
    model_args="pretrained=${model_path},tensor_parallel_size=${eff_tp},dtype=bfloat16,max_model_len=${max_model_len},gpu_memory_utilization=0.70,enforce_eager=True,enable_flashinfer_autotune=False"
    if [ "$trust_remote_code" = "true" ]; then
        model_args+=",trust_remote_code=True"
    fi
    # enforce_eager already forced above for reliability; per-model override in
    # models.yaml is redundant now but kept harmless (idempotent flag).

    echo "=== ${model_name} (bench_tp=${eff_tp}, dp=${dp}, gpus=${NUM_GPUS}, max_model_len=${max_model_len}) ==="

    # Per-model engine overrides. DeepSeek-V2-Lite (MLA) otherwise pulls in the
    # flashinfer rope kernel, which JIT-compiles via the hardcoded
    # /usr/local/cuda/bin/nvcc that env.sh strips on offline nodes (build fails
    # with exit 127). TRITON_MLA keeps attention off the flashinfer JIT path.
    if [ "$model_name" = "DeepSeek-V2-Lite-Chat" ]; then
        export VLLM_ATTENTION_BACKEND=TRITON_MLA
    else
        unset VLLM_ATTENTION_BACKEND
    fi

    # Request-level cache (sqlite). On a resubmit after a timeout/kill, lm_eval
    # skips already-computed requests instead of restarting the full suite.
    CACHE_DIR="${BASE_OUTPUT_DIR}/lm_cache"
    mkdir -p "$CACHE_DIR"

    # shellcheck disable=SC2086
    python "$SCRIPT_DIR/lm_eval_run.py" \
        --model vllm \
        --model_args "$model_args" \
        --tasks "$TASKS" \
        --batch_size auto \
        --seed 42 \
        --use_cache "$CACHE_DIR/cache" \
        --cache_requests true \
        --output_path "$BASE_OUTPUT_DIR" $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: lm_eval failed for ${model_name} (exit ${rc})" >&2
        exit "$rc"
    fi

    shopt -s nullglob
    jsons=("$BASE_OUTPUT_DIR"/*/results_*.json "$BASE_OUTPUT_DIR"/results_*.json)
    shopt -u nullglob
    if [ ${#jsons[@]} -eq 0 ]; then
        echo "ERROR: lm_eval exited 0 but no results_*.json under ${BASE_OUTPUT_DIR}" >&2
        exit 1
    fi
done
