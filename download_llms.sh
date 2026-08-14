#!/bin/bash

# Download every (non-skipped) LLM in models.yaml into $HF_HOME on scratch.
# WARNING: This will require a large amount of disk space (~2–3 TB for all models).
#
# On Leonardo, prefer the SLURM wrapper (avoids login-node OOM kills):
#   ./slurm/submit_download.sh [model_name ...]
#
# Direct / login-node use (small models only):
#   ./cluster/prefetch.sh models
#
# Optional args: models.yaml "name" values to download (default: all non-skipped).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_ONLINE=1
source "$SCRIPT_DIR/cluster/env.sh"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate "$PROXY_MAIN_ENV"

# Build download list: CLI args > ONLY_MODELS env > all models.
FILTER_ARGS=("$@")
if [ "${#FILTER_ARGS[@]}" -eq 0 ] && [ -n "${ONLY_MODELS:-}" ]; then
    IFS='|' read -r -a FILTER_ARGS <<< "$ONLY_MODELS"
fi

mapfile -t MODELS < <(python - "$SCRIPT_DIR/models.yaml" "${FILTER_ARGS[@]}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    all_models = [m for m in yaml.safe_load(f)["models"] if not m.get("skip", False)]
filter_names = sys.argv[2:]
if filter_names:
    by_name = {m["name"]: m["path"] for m in all_models}
    for name in filter_names:
        if name not in by_name:
            raise SystemExit(f"Unknown or skipped model name: {name}")
    paths = [by_name[n] for n in filter_names]
else:
    paths = [m["path"] for m in all_models]
print("\n".join(paths))
PYEOF
)

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "ERROR: no models to download" >&2
    exit 1
fi

workers_for_path() {
    python - "$SCRIPT_DIR/models.yaml" "$1" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    for m in yaml.safe_load(f)["models"]:
        if m["path"] == sys.argv[2]:
            tp = m.get("tp", 1)
            # tp=4 models: 1 worker to fit 16G SLURM mem budget.
            print({1: 4, 2: 4, 4: 1}.get(tp, 2))
            raise SystemExit
raise SystemExit("model not in models.yaml")
PYEOF
}

echo "Starting download of ${#MODELS[@]} LLM(s)..."

FAILED=()
for model in "${MODELS[@]}"; do
    workers="${DOWNLOAD_MAX_WORKERS:-$(workers_for_path "$model")}"
    echo "Downloading $model (max-workers=$workers)"
    if hf download "$model" --max-workers "$workers"; then
        echo "Finished $model"
    else
        echo "[FAILED] $model"
        FAILED+=("$model")
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "${#FAILED[@]} model(s) FAILED to download:"
    printf '  - %s\n' "${FAILED[@]}"
    echo "Gated repos may need access approval at https://huggingface.co/<repo>."
    exit 1
fi

echo "All models downloaded successfully."
