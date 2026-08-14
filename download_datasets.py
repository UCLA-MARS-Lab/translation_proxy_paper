"""Prefetch the 3 parallel corpora exactly as run_translation.py loads them.

Downloads every dataset config/split the translation script will request, so
that translation jobs can run fully offline (HF_HUB_OFFLINE=1) on compute
nodes. Run on a machine with internet access (e.g. a cluster login node).
"""

import os
import sys

from datasets import load_dataset

# languages.py lives in translation/ next to run_translation.py
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "translation")
)
from languages import LANG_MAP

SOURCE_CODE = "eng"
SOURCE_FLORES = LANG_MAP[SOURCE_CODE]["flores_code"]

print("Starting download of 3 parallel corpora (all required configs)...")

failures = []


def fetch(ds_id, cfg, split):
    label = f"{ds_id}" + (f" [{cfg}]" if cfg else "") + f" ({split})"
    try:
        load_dataset(ds_id, cfg, split=split)
        print(f"  [ok] {label}")
    except Exception as e:
        failures.append(label)
        print(f"  [FAIL] {label}: {e}")


# NTREX: one config, test split.
fetch("mteb/NTREX", None, "test")

# FLORES-200 and WMT24++: one config per target language, mirroring the
# (dataset, config, split) tuples built in run_translation.translate_batch.
for lang, info in LANG_MAP.items():
    if lang == SOURCE_CODE:
        continue
    if info.get("flores_code"):
        fetch("facebook/flores", f"{SOURCE_FLORES}-{info['flores_code']}", "devtest")
    if info.get("wmt_code"):
        fetch("google/wmt24pp", f"en-{info['wmt_code']}", "train")

if failures:
    print(f"\n{len(failures)} download(s) failed:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("All datasets downloaded.")
