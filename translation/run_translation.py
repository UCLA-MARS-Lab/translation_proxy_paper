import os
import torch
import pandas as pd
import gc
import datetime
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from languages import LANG_MAP
import traceback
from tqdm import tqdm

# Configuration
SOURCE_CODE = "eng"
SOURCE_FLORES = LANG_MAP[SOURCE_CODE]["flores_code"]


# Model List
MODELS_TO_RUN = [
    # Qwen Family
    {"path": "Qwen/Qwen3-4B", "name": "Qwen3-4B", "family": "qwen", "tp": 2},
    {"path": "Qwen/Qwen3-8B", "name": "Qwen3-8B", "family": "qwen", "tp": 2},
    {"path": "Qwen/Qwen3-14B", "name": "Qwen3-14B", "family": "qwen", "tp": 2},
    {"path": "Qwen/Qwen3-32B", "name": "Qwen3-32B", "family": "qwen", "tp": 2},
    {"path": "Qwen/Qwen3-30B-A3B", "name": "Qwen3-30B-A3B", "family": "qwen", "tp": 2},
    {
        "path": "Qwen/Qwen2.5-72B-Instruct",
        "name": "Qwen2.5-72B-Instruct",
        "family": "qwen",
        "tp": 4,
    },
    # Gemma Family
    {
        "path": "google/gemma-3-1b-it",
        "name": "Gemma-3-1B-it",
        "family": "gemma",
        "tp": 2,
    },
    {
        "path": "google/gemma-3-4b-it",
        "name": "Gemma-3-4B-it",
        "family": "gemma",
        "tp": 2,
    },
    {
        "path": "google/gemma-3-12b-it",
        "name": "Gemma-3-12B-it",
        "family": "gemma",
        "tp": 2,
    },
    {
        "path": "google/gemma-3-27b-it",
        "name": "Gemma-3-27B-it",
        "family": "gemma",
        "tp": 2,
    },
    # Llama Family
    {
        "path": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama-3.3-70B-Instruct",
        "family": "llama",
        "tp": 4,
    },
    # DeepSeek Family
    {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "family": "deepseek",
        "tp": 2,
    },
    {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "name": "DeepSeek-R1-Distill-Llama-70B",
        "family": "deepseek",
        "tp": 4,
    },
    # Mistral / Phi
    {
        "path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "name": "Mistral-Small-24B-Instruct-2501",
        "family": "mistral",
        "tp": 2,
    },
    {"path": "microsoft/phi-4", "name": "phi-4", "family": "phi", "tp": 2},
]


def log(message):
    """Prints to console and appends to a log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def get_sampling_params(family):
    """
    Returns the EXACT sampling parameters you provided.
    """

    if family == "gemma":
        return SamplingParams(
            temperature=1.0,
            top_k=64,
            top_p=0.95,
            max_tokens=1024,
        )
    elif family == "qwen":
        return SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=1024,
        )
    elif family == "deepseek":
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
        )
    elif family == "llama":
        return SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024)
    elif family == "mistral":
        return SamplingParams(temperature=0.15, max_tokens=1024)
    elif family == "phi":
        return SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
    else:
        # Fallback
        return SamplingParams(
            temperature=0.7,
            max_tokens=1024,
        )


def translate_batch(llm, tokenizer, sampling_params, target_lang_code, model_name):
    info = LANG_MAP[target_lang_code]
    target_name = info["name"]

    tasks = []
    # 1. FLORES
    if info.get("flores_code"):
        tasks.append(
            (
                "facebook/flores",
                f"{SOURCE_FLORES}-{info['flores_code']}",
                f"sentence_{SOURCE_FLORES}",
                f"sentence_{info['flores_code']}",
                "flores-200",
            )
        )
    # 2. NTREX
    if info.get("ntrex_code"):
        tasks.append(("mteb/NTREX", None, "eng_Latn", info["ntrex_code"], "ntrex"))
    # 3. WMT
    if info.get("wmt_code"):
        tasks.append(
            ("google/wmt24pp", f"en-{info['wmt_code']}", "source", "target", "wmt24")
        )

    for ds_id, cfg, src_col, tgt_col, folder in tasks:
        # Construct path:
        out_dir = os.path.join("results/translations", folder, model_name)
        out_path = os.path.join(out_dir, f"{SOURCE_CODE}-{target_lang_code}.csv")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(out_path):
            log(f"    Skipping {folder} (Exists): {target_lang_code}")
            continue

        log(f"    Processing {folder}: {target_name} ({target_lang_code})")

        try:
            curr_split = (
                "devtest"
                if "flores" in ds_id
                else ("train" if "wmt24" in ds_id else "test")
            )
            ds = load_dataset(ds_id, cfg, split=curr_split)

            sources = ds[src_col]
            targets = ds[tgt_col]

            if folder == "wmt24":
                sources = sources[1:]
                targets = targets[1:]

            try:
                prompts = [
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"Translate the following sentence into {target_name}. Do not output any other text.\nEnglish: {txt}\n{target_name}:",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for txt in sources
                ]
            except TypeError:
                log("      [WARN] 'enable_thinking' failed. Retrying without it.")
                prompts = [
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"Translate to {target_name}. Output only the translation.\nEnglish: {txt}\n{target_name}:",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for txt in sources
                ]

            log(f"      Generating {len(prompts)} translations...")
            outputs = llm.generate(prompts, sampling_params)
            results = [o.outputs[0].text.strip() for o in outputs]

            pd.DataFrame(
                {"source": sources, "target": targets, "translation": results}
            ).to_csv(out_path, index=False)
            log(f"   [DONE] Saved to {out_path}")

        except Exception as e:
            log(f"   [ERROR] Failed {folder} for {target_lang_code}: {e}")
            with open(LOG_FILE, "a") as f:
                f.write(traceback.format_exc() + "\n")


if __name__ == "__main__":
    # Directory and Logging Setup

    RESULTS_DIR = "./results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOG_FILE = os.path.join(RESULTS_DIR, "translation_log.txt")

    with open(LOG_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"LOG FILE:  {os.path.abspath(RESULTS_DIR)}\n")
        f.write("=" * 60 + "\n")

    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Logging to: {LOG_FILE}")

    for model_cfg in tqdm(MODELS_TO_RUN, desc="Total Progress (Models)", position=0):
        model_path = model_cfg["path"]
        model_name = model_cfg["name"]
        family = model_cfg["family"]
        tp_size = model_cfg["tp"]

        log(f"\n{'=' * 60}")
        log(f"LOADING: {model_name} (TP={tp_size})")
        log(f"{'=' * 60}")

        try:
            # 1. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 2. Get YOUR Exact Params
            sampling_params = get_sampling_params(family)

            # 3. Load vLLM
            llm = LLM(model=model_path, tensor_parallel_size=tp_size, dtype="bfloat16")

            # 4. Run Translations

            langs_to_process = [l for l in LANG_MAP if l != SOURCE_CODE]

            for lang in tqdm(
                langs_to_process,
                desc=f"Languages ({model_name})",
                position=1,
                leave=False,
            ):
                translate_batch(llm, tokenizer, sampling_params, lang, model_name)

            # 5. Cleanup
            log(f"Cleaning up {model_name}...")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"CRITICAL ERROR with {model_name}: {e}")
            with open(LOG_FILE, "a") as f:
                f.write(traceback.format_exc() + "\n")

            try:
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                log("   Emergency cleanup attempted.")
            except:
                pass

    log("\nAll models processed!")
