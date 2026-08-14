#!/usr/bin/env python3
"""Publish proxy-paper translations, benchmarks, and eval_mt scores to the Hub.

Creates GATED datasets under an organization (default: African-Languages-Lab):

    <org>/proxy-mt-translations       raw eng->X model translations (~18 GB)
    <org>/proxy-mt-benchmark-scores   parsed score CSVs + raw lm-eval JSON
    <org>/proxy-mt-eval-scores        BLEU..SSA-COMET CSVs from evaluate_mt.py

Gating is set to "manual" so access must be requested and approved by hand.

Auth: put a WRITE token in <repo>/.env (preferred, used for imsheriff):
    HF_TOKEN=hf_xxx
or export HF_TOKEN / run `hf auth login`. `.env` wins over a cached colleague login.

Usage:
    python publish/push_to_hf.py --which all
    python publish/push_to_hf.py --which translations --dry-run
    python publish/push_to_hf.py --which eval_mt --org African-Languages-Lab
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure we talk to the Hub even if cluster env.sh enabled offline mode.
for _v in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_EVALUATE_OFFLINE"):
    os.environ.pop(_v, None)

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent


def _resolve_results_dir() -> Path:
    """Locate the results dir even when cluster/env.sh was not sourced.

    Order: PROXY_RESULTS_DIR -> $PROXY_ROOT/results -> $SCRATCH/proxy_paper/results
    -> <repo>/results.
    """
    env = os.environ.get("PROXY_RESULTS_DIR")
    if env:
        return Path(env)
    root = os.environ.get("PROXY_ROOT")
    if root:
        return Path(root) / "results"
    scratch = os.environ.get("SCRATCH")
    if scratch and (Path(scratch) / "proxy_paper/results").is_dir():
        return Path(scratch) / "proxy_paper/results"
    return REPO_DIR / "results"


RESULTS_DIR = _resolve_results_dir()

TRANSLATIONS_SRC = RESULTS_DIR / "translations"
PARSED_SRC = RESULTS_DIR / "parsed"
RAW_SRC = RESULTS_DIR / "raw"
METRICS_SRC = RESULTS_DIR / "metrics"

DEFAULT_ORG = "African-Languages-Lab"
TRANSLATIONS_REPO = "proxy-mt-translations"
BENCHMARK_REPO = "proxy-mt-benchmark-scores"
EVAL_MT_REPO = "proxy-mt-eval-scores"

CARD_TRANSLATIONS = REPO_DIR / "publish" / "README_translations.md"
CARD_BENCHMARK = REPO_DIR / "publish" / "README_benchmark.md"
CARD_EVAL_MT = REPO_DIR / "publish" / "README_eval_mt.md"


def log(msg: str) -> None:
    print(f"[push] {msg}", flush=True)


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from <repo>/.env without overriding existing env vars."""
    env_path = REPO_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_token() -> str:
    _load_dotenv()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        # fall back to a token stored via `hf auth login` (may be a colleague account)
        from huggingface_hub import get_token as _gt

        tok = _gt()
    if not tok:
        sys.exit(
            "ERROR: no Hugging Face token found. Put HF_TOKEN=hf_... in "
            f"{REPO_DIR / '.env'} (write scope, org owner), or export HF_TOKEN."
        )
    return tok


def ensure_gated_repo(api: HfApi, repo_id: str, token: str, dry_run: bool) -> None:
    log(f"create dataset repo {repo_id} (exist_ok)")
    if dry_run:
        return
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    log(f"set gating=manual on {repo_id}")
    api.update_repo_settings(
        repo_id=repo_id, repo_type="dataset", gated="manual", token=token
    )


def upload_card(api: HfApi, repo_id: str, card: Path, token: str, dry_run: bool) -> None:
    log(f"upload dataset card -> {repo_id}/README.md")
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add/update dataset card",
    )


def push_translations(api: HfApi, org: str, token: str, dry_run: bool) -> None:
    if not TRANSLATIONS_SRC.is_dir():
        sys.exit(f"ERROR: translations dir not found: {TRANSLATIONS_SRC}")
    repo_id = f"{org}/{TRANSLATIONS_REPO}"
    n_csv = sum(1 for _ in TRANSLATIONS_SRC.rglob("*.csv"))
    log(f"translations source: {TRANSLATIONS_SRC} ({n_csv} CSVs)")
    ensure_gated_repo(api, repo_id, token, dry_run)
    upload_card(api, repo_id, CARD_TRANSLATIONS, token, dry_run)
    log(f"upload_large_folder -> {repo_id} (resumable; safe to re-run)")
    if dry_run:
        return
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(TRANSLATIONS_SRC),
        allow_patterns=["*.csv"],
    )
    log(f"translations done: https://huggingface.co/datasets/{repo_id}")


def stage_benchmark(stage: Path) -> tuple[int, int]:
    """Copy parsed score CSVs and raw results/logs (minus lm_cache) into `stage`."""
    if stage.exists():
        shutil.rmtree(stage)
    scores_dir = stage / "scores"
    scores_dir.mkdir(parents=True)
    n_scores = 0
    for csv in sorted(PARSED_SRC.glob("*.csv")):
        shutil.copy2(csv, scores_dir / csv.name)
        n_scores += 1

    # Only models that actually produced a results_*.json count as completed.
    n_models = 0
    for model_dir in sorted(RAW_SRC.iterdir()):
        if not model_dir.is_dir():
            continue
        result_jsons = [
            p for p in model_dir.rglob("results_*.json") if "lm_cache" not in p.parts
        ]
        if not result_jsons:
            continue
        n_models += 1
        for src in model_dir.rglob("*"):
            if not src.is_file() or "lm_cache" in src.parts:
                continue
            rel = src.relative_to(RAW_SRC)
            dst = stage / "raw" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return n_scores, n_models


def push_benchmark(api: HfApi, org: str, token: str, dry_run: bool) -> None:
    if not PARSED_SRC.is_dir():
        sys.exit(f"ERROR: parsed scores dir not found: {PARSED_SRC}")
    repo_id = f"{org}/{BENCHMARK_REPO}"
    stage = RESULTS_DIR / "hf_stage_benchmark"
    log(f"stage benchmark payload -> {stage}")
    n_scores, n_models = stage_benchmark(stage)
    log(f"staged {n_scores} score CSVs + raw results/logs for {n_models} models")
    ensure_gated_repo(api, repo_id, token, dry_run)
    upload_card(api, repo_id, CARD_BENCHMARK, token, dry_run)
    log(f"upload_folder -> {repo_id}")
    if dry_run:
        return
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage),
        token=token,
        commit_message="Add benchmark scores and raw lm-eval results",
    )
    log(f"benchmark done: https://huggingface.co/datasets/{repo_id}")


def push_eval_mt(api: HfApi, org: str, token: str, dry_run: bool) -> None:
    if not METRICS_SRC.is_dir():
        sys.exit(f"ERROR: eval_mt metrics dir not found: {METRICS_SRC}")
    repo_id = f"{org}/{EVAL_MT_REPO}"
    n_csv = sum(1 for _ in METRICS_SRC.rglob("*.csv"))
    log(f"eval_mt source: {METRICS_SRC} ({n_csv} CSVs)")
    ensure_gated_repo(api, repo_id, token, dry_run)
    upload_card(api, repo_id, CARD_EVAL_MT, token, dry_run)
    log(f"upload_folder -> {repo_id}")
    if dry_run:
        return
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(METRICS_SRC),
        token=token,
        allow_patterns=["**/*.csv"],
        commit_message="Add/update eval_mt metric CSVs (BLEU..SSA-COMET)",
    )
    log(f"eval_mt done: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--which",
        choices=["translations", "benchmark", "eval_mt", "all", "both"],
        default="all",
        help="which dataset(s) to publish (default: all)",
    )
    ap.add_argument("--org", default=DEFAULT_ORG, help=f"HF org (default: {DEFAULT_ORG})")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned actions without creating repos or uploading",
    )
    args = ap.parse_args()

    token = None if args.dry_run else get_token()
    api = HfApi(token=token)
    if not args.dry_run:
        who = api.whoami(token=token)
        log(f"authenticated as {who.get('name')} (orgs: {[o.get('name') for o in who.get('orgs', [])]})")

    which = "all" if args.which == "both" else args.which
    if which in ("eval_mt", "all"):
        push_eval_mt(api, args.org, token, args.dry_run)
    if which in ("benchmark", "all"):
        push_benchmark(api, args.org, token, args.dry_run)
    if which in ("translations", "all"):
        push_translations(api, args.org, token, args.dry_run)

    log("all requested uploads complete." if not args.dry_run else "dry-run complete.")


if __name__ == "__main__":
    main()
