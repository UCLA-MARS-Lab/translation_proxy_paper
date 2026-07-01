import os
import torch
import pandas as pd
import gc
import datetime
import yaml
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from languages import LANG_MAP
import traceback
from tqdm import tqdm

# Configuration
SOURCE_CODE = "eng"
SOURCE_FLORES = LANG_MAP[SOURCE_CODE]["flores_code"]

# Fixed RNG seed for every sampling call so translation runs are reproducible.
# vLLM ignores it for greedy (temperature=0.0) decoding, where it is a no-op.
SEED = 42


# Model List (single source of truth: models.yaml at the repo root)
_MODELS_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.yaml"
)
with open(_MODELS_YAML) as f:
    MODELS_TO_RUN = yaml.safe_load(f)["models"]


def log(message):
    """Prints to console and appends to a log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


def get_sampling_params(family):
    """
    Returns the decoding parameters used for each model family.

    Methodology (extends the setup of arXiv:2601.11778, which used
    "model-specific sampling parameters to align with the recommended
    configuration for each architecture"):

      * Where a model's developers publish a recommended sampling
        configuration (model card or generation_config.json), we use it.
      * Where the developers recommend greedy decoding, or publish no sampling
        recommendation at all, we default to greedy (deterministic) decoding
        -- temperature=0.0, which vLLM treats as argmax/greedy. This keeps the
        no-recommendation models reproducible and avoids importing unofficial
        third-party numbers.

    Every branch cites its source URL so the choices can be audited and
    reproduced. All sampling calls pass a fixed SEED for run-to-run
    reproducibility. Verified 2026-07 against the models in models.yaml.

    Families with an OFFICIAL recommendation (we follow it):
        gemma, qwen, deepseek, llama, mistral, phi, cohere, olmo
    Families that are GREEDY (vendor recommends greedy, or no rec published):
        granite, openelm, falcon, moonshot
    """

    if family == "gemma":
        # Gemma 3 / Gemma 4: Google-recommended temp 1.0, top_p 0.95, top_k 64.
        # https://huggingface.co/google/gemma-3-12b-it/discussions/25
        return SamplingParams(
            temperature=1.0,
            top_k=64,
            top_p=0.95,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "qwen":
        # Qwen3 / Qwen3.6 non-thinking mode (we set enable_thinking=False):
        # temp 0.7, top_p 0.8, top_k 20, min_p 0. Qwen warns against greedy
        # decoding. (Qwen3.6 optionally adds presence_penalty 1.5 to curb
        # repetition; omitted here so Qwen3 and Qwen3.6 share one preset.)
        # https://huggingface.co/Qwen/Qwen3-8B  (Best Practices)
        return SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "deepseek":
        # DeepSeek-R1 (and its Distill variants): temp 0.5-0.7 (0.6 recommended),
        # top_p 0.95. https://huggingface.co/deepseek-ai/DeepSeek-R1  (Usage
        # Recommendations)
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "llama":
        # Llama 3.x / Llama 4: generation_config.json ships temp 0.6, top_p 0.9
        # (no top_k). Llama 4 uses the same values.
        # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/generation_config.json
        return SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024, seed=SEED)
    elif family == "mistral":
        # Ministral-3-*-Instruct-2512: card recommends a temperature *below 0.1*
        # for daily-driver / production use; no top_p specified. (The original
        # paper listed Mistral at 0.15 for an older Mistral model; models.yaml
        # now contains only Ministral-3-2512, so we follow its newer card.)
        # https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
        return SamplingParams(temperature=0.1, max_tokens=1024, seed=SEED)
    elif family == "phi":
        # phi-4: temp 0.8, top_p 0.95. Commonly-cited Phi-4 default (also used by
        # Phi-4-reasoning); note phi-4's generation_config.json pins no sampling
        # params, so this is the reported default rather than a config value.
        # https://huggingface.co/microsoft/phi-4
        return SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "cohere":
        # Aya Expanse / Command-R: model-card examples use do_sample=True,
        # temperature 0.3. Cohere's API default top_p ("p") is 0.75, top_k off.
        # https://huggingface.co/CohereLabs/aya-expanse-8b
        # https://docs.cohere.com/docs/advanced-generation-hyperparameters
        return SamplingParams(
            temperature=0.3,
            top_p=0.75,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "olmo":
        # OLMo 3 Instruct: generation_config.json ships do_sample=True,
        # temp 0.6, top_p 0.95 (no top_k). The base Olmo-3-1125-32B publishes no
        # sampling params (defaults greedy); we apply the Instruct preset here.
        # https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/generation_config.json
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "granite":
        # IBM Granite 4: IBM recommends greedy decoding for deterministic,
        # reproducible output; no sampling tuple is published. => greedy.
        # https://huggingface.co/ibm-granite/granite-4.1-8b
        # https://www.ibm.com/granite/docs/models/code
        return SamplingParams(temperature=0.0, max_tokens=1024)
    elif family == "openelm":
        # Apple OpenELM: no sampling params published (ships greedy). The card's
        # only generation suggestion is repetition_penalty=1.2. => greedy.
        # https://huggingface.co/apple/OpenELM-3B-Instruct
        return SamplingParams(
            temperature=0.0,
            repetition_penalty=1.2,
            max_tokens=1024,
        )
    elif family == "falcon":
        # Falcon3 Instruct: TII publishes NO recommended sampling params (the
        # commonly-quoted temp 0.2 / top_p 0.95 / top_k 50 values come only from
        # third-party deployment catalogs, not TII). No-recommendation => greedy.
        # https://huggingface.co/tiiuae/Falcon3-10B-Instruct
        return SamplingParams(temperature=0.0, max_tokens=1024)
    elif family == "moonshot":
        # Kimi-Linear / Moonlight: Moonshot publishes no sampling params in the
        # model cards or generation_config.json. No-recommendation => greedy.
        # https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
        return SamplingParams(temperature=0.0, max_tokens=1024)
    else:
        # Fallback for any family not listed above: greedy, matching the
        # "no published recommendation => deterministic decoding" rule so the
        # code never silently samples at an arbitrary temperature.
        return SamplingParams(temperature=0.0, max_tokens=1024)


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
