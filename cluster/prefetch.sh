#!/bin/bash
# Prefetch every asset the experiments need onto scratch.
# Run on a LOGIN NODE (needs internet) — Leonardo compute nodes are offline.
#
# Usage:
#   ./cluster/prefetch.sh all            # everything (full production prefetch)
#   ./cluster/prefetch.sh smoke          # minimal subset for the smoke test
#   ./cluster/prefetch.sh models         # all (non-skipped) LLMs
#   ./cluster/prefetch.sh corpora        # the 3 parallel corpora
#   ./cluster/prefetch.sh metrics        # COMET ckpts, MetricX, evaluate+nltk
#   ./cluster/prefetch.sh benchmarks     # all lm-eval benchmark datasets
#
# Gated repos (Cohere, meta-llama, google/gemma) need a Hugging Face token:
# run `hf auth login` (or export HF_TOKEN) before prefetching models.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_ONLINE=1 source "$SCRIPT_DIR/env.sh"

STAGE="${1:-all}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

activate() {
    conda deactivate 2>/dev/null || true
    conda activate "$1"
}

prefetch_models() {
    echo "===== [prefetch] LLMs (SLURM lrd_all_serial — avoids login-node OOM) ====="
    "$PROXY_REPO_DIR/slurm/submit_download.sh"
}

prefetch_smoke_model() {
    echo "===== [prefetch] Smoke-test LLM (Llama-3.2-1B-Instruct) ====="
    activate "$PROXY_MAIN_ENV"
    # Full snapshot (no excludes): offline loading requires every repo file.
    hf download meta-llama/Llama-3.2-1B-Instruct --max-workers 16
}

prefetch_corpora() {
    echo "===== [prefetch] Parallel corpora (FLORES-200, WMT24++, NTREX) ====="
    activate "$PROXY_MAIN_ENV"
    python "$PROXY_REPO_DIR/download_datasets.py"
}

prefetch_metrics() {
    echo "===== [prefetch] Metric models and evaluate/nltk assets ====="
    # COMET checkpoints + HF evaluate modules + nltk data (proxy_comet env,
    # where evaluate_mt.py runs).
    activate "$PROXY_COMET_ENV"
    python - <<'PYEOF'
import nltk, os

nltk_dir = os.environ["NLTK_DATA"]
for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg, download_dir=nltk_dir)
    # Ensure corpora are extracted (offline compute nodes cannot unzip on demand).
    for sub in ("corpora", "tokenizers"):
        d = os.path.join(nltk_dir, sub)
        if os.path.isdir(d):
            for z in os.listdir(d):
                if z.endswith(".zip"):
                    import zipfile
                    dest = os.path.join(d, z[:-4])
                    if not os.path.isdir(dest):
                        zipfile.ZipFile(os.path.join(d, z)).extractall(d)

import evaluate
evaluate.load("meteor")
evaluate.load("rouge")
print("evaluate modules cached.")

from comet import download_model, load_from_checkpoint
from transformers import XLMRobertaTokenizer

# Encoder tokenizer used by XCOMET-XL / SSA-COMET; must be cached for offline
# compute nodes (COMET loads it via from_pretrained at runtime).
XLMRobertaTokenizer.from_pretrained("facebook/xlm-roberta-xl")

for repo in ["Unbabel/XCOMET-XL", "McGill-NLP/ssa-comet-mtl"]:
    print(f"Downloading and warming {repo} ...")
    ckpt = download_model(repo)
    load_from_checkpoint(ckpt)
print("COMET checkpoints cached and warmed.")

    # sacrebleu flores200 tokenizer (needed for BLEU with tokenize=flores200).
    import sacrebleu
    sacrebleu.corpus_bleu(["test"], [["test"]], tokenize="flores200")
    print(f"sacrebleu flores200 tokenizer cached under {os.environ.get('SACREBLEU')}.")
PYEOF

    # MetricX model + tokenizer (weights are plain HF repos; cache location is
    # HF_HOME so any env's `hf` CLI works).
    activate "$PROXY_METRICX_ENV"
    hf download google/metricx-24-hybrid-xl-v2p6
    hf download google/mt5-xl
}

prefetch_benchmarks() {
    local tasks="$1"
    echo "===== [prefetch] lm-eval benchmark datasets ====="
    activate "$PROXY_MAIN_ENV"
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    python - "$tasks" <<'PYEOF'
import sys

task_list = [t for t in sys.argv[1].split(",") if t]
print(f"Prefetching datasets for {len(task_list)} lm-eval task specs...")

from lm_eval.tasks import TaskManager, get_task_dict

tm = TaskManager()
failed = []
for spec in task_list:
    try:
        # Instantiating the task forces its dataset download into HF_HOME.
        get_task_dict([spec], tm)
        print(f"  [ok] {spec}")
    except Exception as e:
        failed.append(spec)
        print(f"  [FAIL] {spec}: {e}")

if failed:
    print(f"\n{len(failed)} task spec(s) failed to prefetch: {failed}")
    sys.exit(1)
print("All benchmark datasets cached.")
PYEOF
}

all_bench_tasks() {
    BENCH_PRINT_TASKS=1 bash "$PROXY_REPO_DIR/benchmark/run_benchmarks.sh"
}

case "$STAGE" in
    models)     prefetch_models ;;
    corpora)    prefetch_corpora ;;
    metrics)    prefetch_metrics ;;
    benchmarks) prefetch_benchmarks "$(all_bench_tasks)" ;;
    smoke)
        prefetch_smoke_model
        prefetch_corpora
        prefetch_metrics
        prefetch_benchmarks "mgsm_direct"
        ;;
    all)
        prefetch_models
        prefetch_corpora
        prefetch_metrics
        prefetch_benchmarks "$(all_bench_tasks)"
        ;;
    *) echo "Usage: $0 [all|smoke|models|corpora|metrics|benchmarks]" >&2; exit 1 ;;
esac

echo "[prefetch] Stage '$STAGE' complete."
