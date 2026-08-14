#!/bin/bash
# Submit the two-stage smoke test on the debug QOS.
# Stage 2 (metrics) only starts if stage 1 (generation) succeeds.
#
# Usage: ./slurm/submit_smoke.sh
# Prereq: ./cluster/prefetch.sh smoke   (run on a login node first)

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"

gen_id="$(sbatch --parsable \
    --job-name=proxy-smoke-generate \
    --output="$PROXY_SLURM_LOGS/%x_%j.out" \
    --export=ALL,STEP=generate,PROXY_REPO="$REPO_DIR" \
    "$SCRIPT_DIR/smoke_test.sbatch")"
echo "[smoke] generate job: $gen_id"

met_id="$(sbatch --parsable \
    --job-name=proxy-smoke-metrics \
    --output="$PROXY_SLURM_LOGS/%x_%j.out" \
    --dependency="afterok:$gen_id" \
    --export=ALL,STEP=metrics,PROXY_REPO="$REPO_DIR" \
    "$SCRIPT_DIR/smoke_test.sbatch")"
echo "[smoke] metrics job:  $met_id (depends on $gen_id)"
