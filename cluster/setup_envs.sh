#!/bin/bash
# Create the three conda environments on scratch and clone the MetricX repo.
# Run once on a LOGIN NODE (needs internet).
#
# Usage: ./cluster/setup_envs.sh [main|comet|metricx|metricx-repo|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_ONLINE=1 source "$SCRIPT_DIR/env.sh"

TARGET="${1:-all}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

create_env() {
    local yml="$1" prefix="$2"
    if [ -d "$prefix" ] && [ -x "$prefix/bin/python" ]; then
        echo "[setup] Env already exists: $prefix (delete it to rebuild)"
        return 0
    fi
    echo "[setup] Creating env from $yml at $prefix ..."
    conda env create -f "$PROXY_REPO_DIR/$yml" -p "$prefix"
}

mkdir -p "$PROXY_ENVS_DIR"

case "$TARGET" in
    main|all)    create_env proxy_main.yml    "$PROXY_MAIN_ENV" ;;&
    comet|all)   create_env proxy_comet.yml   "$PROXY_COMET_ENV" ;;&
    metricx|all) create_env proxy_metricx.yml "$PROXY_METRICX_ENV" ;;&
    metricx-repo|all)
        if [ ! -d "$METRICX_REPO_DIR/.git" ]; then
            echo "[setup] Cloning google-research/metricx into $METRICX_REPO_DIR ..."
            git clone https://github.com/google-research/metricx.git "$METRICX_REPO_DIR"
        else
            echo "[setup] MetricX repo already present: $METRICX_REPO_DIR"
        fi
        ;;&
    main|comet|metricx|metricx-repo|all) : ;;
    *) echo "Usage: $0 [main|comet|metricx|metricx-repo|all]" >&2; exit 1 ;;
esac

echo "[setup] Done."
