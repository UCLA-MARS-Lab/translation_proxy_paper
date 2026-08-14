# Running proxy_paper_runs on Leonardo (CINECA)

This guide covers the SLURM-based adaptation of the translation-proxy experiment for the Leonardo `boost_usr_prod` partition (4× A100 64GB per node).

## One-time setup (login node, needs internet)

```bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/001/proxy_paper_runs

# 1. Create conda envs + clone MetricX repo on scratch
./cluster/setup_envs.sh all

# 2. Ensure Hugging Face token is in the scratch cache (gated models)
#    cp $SCRATCH/huggingface/token $SCRATCH/proxy_paper/hf_cache/token

# 3. Prefetch assets (choose one)
./cluster/prefetch.sh smoke     # minimal subset for smoke test (~30 min)
./cluster/prefetch.sh all       # full production prefetch (~hours, ~2–3 TB)

# Model downloads use SLURM lrd_all_serial (internet + 300G RAM) to avoid
# login-shell OOM kills on 70B weights. Submit only the failed models:
#   ./slurm/submit_download.sh Llama-3.3-70B-Instruct Llama-4-Scout-17B-16E-Instruct
# Monitor: tail -f $SCRATCH/proxy_paper/slurm_logs/proxy-download_*.out

# Prefetch vs download_llms.sh:
#   ./cluster/prefetch.sh models  — submits ./slurm/submit_download.sh (all models)
#   ./slurm/submit_download.sh    — same, optionally filter by models.yaml name
#   ./download_llms.sh            — underlying loop; run inside the SLURM job
```

Scratch layout (everything heavy):

| Path | Purpose |
|------|---------|
| `$SCRATCH/proxy_paper/envs/` | conda envs (`proxy_main`, `proxy_comet`, `proxy_metricx`) |
| `$SCRATCH/proxy_paper/hf_cache/` | Hugging Face models, datasets, COMET/MetricX weights |
| `$SCRATCH/proxy_paper/results/` | translations, benchmarks, metrics |
| `$SCRATCH/proxy_paper/slurm_logs/` | SLURM stdout/stderr |
| `$SCRATCH/proxy_paper/metricx/` | cloned google-research/metricx repo |

## Smoke test (verify end-to-end on debug QOS)

```bash
./slurm/submit_smoke.sh
# monitor:  squeue -u $USER
# logs:     ls $SCRATCH/proxy_paper/slurm_logs/proxy-smoke-*
```

Smoke test runs two chained jobs (30 min each, 1 GPU):
1. **generate** — translation (1 language × 3 corpora) + bounded lm-eval (`mgsm_direct --limit 8`)
2. **metrics** — BLEU/chrF++/ROUGE-L/METEOR/XCOMET/SSA-COMET + MetricX backfill

Expected outputs after success:
- `$SCRATCH/proxy_paper/results/translations/{flores-200,ntrex,wmt24}/Llama-3.2-1B-Instruct/eng-fra.csv`
- `$SCRATCH/proxy_paper/results/metrics/Llama-3.2-1B-Instruct/{flores-200,ntrex,wmt24}.csv` (all 7 metric columns populated)
- `$SCRATCH/proxy_paper/results/raw/Llama-3.2-1B-Instruct/.../results_*.json`

## Production runs (NOT auto-launched)

Submit after `./cluster/prefetch.sh all` completes:

```bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/001/proxy_paper_runs

# Translation generation (one SLURM job per model, tp-matched GPUs)
./slurm/submit_all.sh translate

# Multilingual benchmarks (one job per model; heavy — up to 24h for 70B)
./slurm/submit_all.sh benchmark

# After translations exist: MT metrics (array job, 1 GPU per model)
./slurm/submit_all.sh eval_mt

# After eval_mt: MetricX backfill (array job, 1 GPU per model)
./slurm/submit_all.sh metricx
```

Submit a single model:

```bash
./slurm/submit_all.sh translate Llama-3.2-1B-Instruct
./slurm/submit_all.sh benchmark  Llama-3.2-1B-Instruct
```

### Benchmark retries (failed Jul 9 batch)

After fixing `benchmark/run_benchmarks.sh` (vLLM context caps, `trust_remote_code`,
`pipefail`, results check), resubmit only the models that failed:

```bash
# cancel zombies if any are still RUNNING with dead engines:
scancel <jobid> ...

./slurm/submit_benchmark_failed.sh              # all 27 failed models
./slurm/submit_benchmark_failed.sh phi-4        # single model from the list
```

**Success criterion:** Slurm `COMPLETED` is not enough. Check for a fresh
`results_*.json` under `$SCRATCH/proxy_paper/results/raw/<model>/`:

```bash
ls $SCRATCH/proxy_paper/results/raw/<model>/*/results_*.json
```

`DeepSeek-V2-Lite-Chat` uses FlashInfer RoPE kernels; if it fails with `nvcc: No
such file or directory`, run a one-model benchmark smoke on a **login node with
CUDA** (or prefetch) so `$SCRATCH/proxy_paper/.cache/flashinfer/` is populated
before resubmitting on compute nodes.

### GPU sizing (from `models.yaml` `tp` field)

| tp | GPUs | CPUs | RAM | Use case |
|----|------|------|-----|----------|
| 1  | 1    | 8    | 120G | ≤16B params |
| 2  | 2    | 16   | 240G | ~27–48B |
| 4  | 4    | 32   | 480G | 70B+ (full node) |

`Llama-4-Maverick` is marked `skip: true` (cannot fit one node). `Llama-4-Scout` is best-effort at tp=4.

### Post-processing

```bash
source cluster/env.sh
proxy_activate main
python benchmark/parse.py   # writes $PROXY_RESULTS_DIR/parsed/<model>.csv
```

Sync parsed results off scratch periodically (Leonardo scratch is purged on a schedule):

```bash
rsync -av $SCRATCH/proxy_paper/results/parsed/ $WORK/proxy_paper_results/parsed/
```

## Key cluster adaptations

- **Offline compute nodes**: all assets prefetched on login node; jobs run with `HF_HUB_OFFLINE=1`.
- **CUDA module unloaded**: loading `module cuda` injects stub NVML libs that break vLLM; pip torch wheels carry their own CUDA runtime.
- **FlashInfer disabled**: `VLLM_USE_FLASHINFER_SAMPLER=0` avoids JIT compilation (no nvcc on compute nodes).
- **Per-model SLURM jobs**: GPU count varies by model, so array jobs are not used for translate/benchmark.
- **Research integrity preserved**: sampling params, prompts, seeds, datasets, and metric formulas are unchanged from the original scripts.

## Account / partition

- Account: `AIH4A_udutech`
- Partition: `boost_usr_prod` (24h walltime)
- Debug QOS: `boost_qos_dbg` (30 min, used by smoke test)
