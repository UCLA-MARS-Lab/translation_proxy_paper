
#!/bin/bash

# Check if GPUs were passed as an argument
if [ -z "$1" ]; then
    echo "Error: No GPUs specified."
    echo "Usage: $0 <gpu_ids> (e.g., $0 0,1,2,3)"
    exit 1
fi

# Configuration from arguments
GPUS=$1

# Calculate TP_SIZE 
TP_SIZE=$(echo $GPUS | tr -cd ',' | wc -c)
TP_SIZE=$((TP_SIZE + 1))

# Environment Variables
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export VLLM_USE_V1=0

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

declare -a MODELS=(
    "Qwen3-4B|Qwen/Qwen3-4B"
    "Qwen3-8B|Qwen/Qwen3-8B"
    "Qwen3-14B|Qwen/Qwen3-14B"
    "Qwen3-32B|Qwen/Qwen3-32B"
    "Qwen3-30B-A3B|Qwen/Qwen3-30B-A3B"
    "Qwen2.5-72B-Instruct|Qwen/Qwen2.5-72B-Instruct"
    "Gemma-3-1B-it|google/gemma-3-1b-it"
    "Gemma-3-4B-it|google/gemma-3-4b-it"
    "Gemma-3-12B-it|google/gemma-3-12b-it"
    "Gemma-3-27B-it|google/gemma-3-27b-it"
    "Llama-3.3-70B-Instruct|meta-llama/Llama-3.3-70B-Instruct"
    "DeepSeek-R1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "DeepSeek-R1-Distill-Llama-70B|deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "Phi-4|microsoft/phi-4"
)

echo "Starting Evaluation..."

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    model_path="${entry##*|}"
    BASE_OUTPUT_DIR="./results/raw/${model}"
    LOG_FILE="${BASE_OUTPUT_DIR}/raw_log.txt"

    mkdir -p "$BASE_OUTPUT_DIR"

    lm_eval \
        --model vllm \
        --model_args "pretrained=${model_path},tensor_parallel_size=${TP_SIZE},dtype=bfloat16" \
        --tasks "$ALL_TASKS" \
        --batch_size auto \
        --seed 42 \
        --output_path "$BASE_OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
done
