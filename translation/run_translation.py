import argparse
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
SEED = 42

# Smallest recommended temperature in our set (Ministral-3). Used for models
# whose developers publish no sampling recommendation: instead of exact greedy
# (temperature 0.0) we sample at this minimum non-zero temperature.
MIN_TEMPERATURE = 0.1


# Model List (single source of truth: models.yaml at the repo root)
_MODELS_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.yaml"
)
with open(_MODELS_YAML) as f:
    # Models flagged skip: true (too large for a single cluster node) are
    # excluded everywhere.
    MODELS_TO_RUN = [
        m for m in yaml.safe_load(f)["models"] if not m.get("skip", False)
    ]


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
      * Where the developers publish no sampling recommendation at all (their
        generation_config.json carries only token IDs -- verified 2026-07 for
        every such model, see per-branch URLs), we sample at MIN_TEMPERATURE,
        the smallest temperature in our recommended set (0.1, from Ministral-3).
        This keeps every model in the same sampling regime -- near-deterministic
        but not exact greedy -- and avoids importing unofficial third-party
        numbers.

    Every branch cites its source URL so the choices can be audited and
    reproduced. All calls pass a fixed SEED for run-to-run reproducibility.
    Verified 2026-07 against the models in models.yaml.

    Families with an OFFICIAL recommendation (we follow it):
        gemma, qwen, deepseek, llama, mistral, phi, cohere, olmo, internlm, glm
    Families with NO published recommendation (we use MIN_TEMPERATURE):
        granite, openelm, falcon, moonshot, yi
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
        # Mistral AI instruct models (Ministral-3-2512, Mistral-7B-v0.3, NeMo,
        # Mixtral): Ministral-3 card recommends temp below 0.1; classic Mistral
        # models publish no sampling tuple in generation_config.json.
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
        # IBM Granite 4: generation_config.json carries only token IDs (no
        # sampling params); IBM otherwise recommends greedy. No published
        # sampling tuple => MIN_TEMPERATURE.
        # https://huggingface.co/ibm-granite/granite-4.1-8b/raw/main/generation_config.json
        return SamplingParams(temperature=MIN_TEMPERATURE, max_tokens=1024, seed=SEED)
    elif family == "openelm":
        # Apple OpenELM: generation_config.json carries only token IDs (no
        # sampling params). The card's one generation suggestion is
        # repetition_penalty=1.2, which we keep. No sampling tuple =>
        # MIN_TEMPERATURE.
        # https://huggingface.co/apple/OpenELM-3B-Instruct/raw/main/generation_config.json
        return SamplingParams(
            temperature=MIN_TEMPERATURE,
            repetition_penalty=1.2,
            max_tokens=1024,
            seed=SEED,
        )
    elif family == "falcon":
        # Falcon3 Instruct: generation_config.json carries only token IDs (no
        # sampling params); TII publishes no recommendation (the commonly-quoted
        # temp 0.2 / top_p 0.95 / top_k 50 values come only from third-party
        # deployment catalogs, not TII). No sampling tuple => MIN_TEMPERATURE.
        # https://huggingface.co/tiiuae/Falcon3-10B-Instruct/raw/main/generation_config.json
        return SamplingParams(temperature=MIN_TEMPERATURE, max_tokens=1024, seed=SEED)
    elif family == "yi":
        # Yi-1.5: generation_config.json carries only token IDs (no sampling
        # params). No published sampling tuple => MIN_TEMPERATURE.
        # https://huggingface.co/01-ai/Yi-1.5-9B-Chat/raw/main/generation_config.json
        return SamplingParams(temperature=MIN_TEMPERATURE, max_tokens=1024, seed=SEED)
    elif family == "internlm":
        # InternLM2.5: generation_config.json carries only token IDs; official
        # llama.cpp deployment example uses temp 0.8, top_p 0.8.
        # https://huggingface.co/internlm/internlm2_5-7b-chat-gguf
        return SamplingParams(temperature=0.8, top_p=0.8, max_tokens=1024, seed=SEED)
    elif family == "glm":
        # GLM-4-9B-Chat: generation_config.json ships do_sample=True,
        # temp 0.8, top_p 0.8.
        # https://huggingface.co/THUDM/glm-4-9b-chat/raw/main/generation_config.json
        return SamplingParams(temperature=0.8, top_p=0.8, max_tokens=1024, seed=SEED)
    elif family == "moonshot":
        # Kimi-Linear / Moonlight: generation_config.json carries only token IDs
        # (no sampling params); Moonshot publishes no recommendation. No sampling
        # tuple => MIN_TEMPERATURE.
        # https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/generation_config.json
        return SamplingParams(temperature=MIN_TEMPERATURE, max_tokens=1024, seed=SEED)
    else:
        # Fallback for any family not listed above: MIN_TEMPERATURE, matching the
        # "no published recommendation => smallest recommended temperature" rule
        # so the code never silently samples at an arbitrary temperature.
        return SamplingParams(temperature=MIN_TEMPERATURE, max_tokens=1024, seed=SEED)


def format_translation_prompts(tokenizer, sources, target_name, prompt_mode):
    """Build per-sentence prompts for translation.

    Most instruct models ship a chat template; tiny-aya-base is a pretrained
    base model and Cohere's card documents plain completion instead:
    https://huggingface.co/CohereLabs/tiny-aya-base
    """
    if prompt_mode == "completion":
        return [
            (
                f"Translate the following sentence into {target_name}. "
                f"Output only the translation.\n"
                f"English: {txt}\n{target_name}:"
            )
            for txt in sources
        ]

    try:
        return [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Translate the following sentence into {target_name}. "
                            f"Do not output any other text.\n"
                            f"English: {txt}\n{target_name}:"
                        ),
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
        return [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Translate to {target_name}. Output only the translation.\n"
                            f"English: {txt}\n{target_name}:"
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for txt in sources
        ]


def translate_batch(llm, tokenizer, sampling_params, target_lang_code, model_name, prompt_mode):
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
        out_dir = os.path.join(RESULTS_DIR, "translations", folder, model_name)
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
                prompts = format_translation_prompts(
                    tokenizer, sources, target_name, prompt_mode
                )
            except Exception as e:
                raise RuntimeError(f"prompt formatting failed: {e}") from e

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
    parser = argparse.ArgumentParser(
        description="Generate translations with vLLM for the models in models.yaml."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run only the model with this 'name' from models.yaml "
        "(default: run all models sequentially).",
    )
    parser.add_argument(
        "--langs",
        default=None,
        help="Comma-separated subset of target language codes to translate "
        "(default: all languages in LANG_MAP). Intended for smoke tests.",
    )
    args = parser.parse_args()

    models_to_run = MODELS_TO_RUN
    if args.model:
        models_to_run = [m for m in MODELS_TO_RUN if m["name"] == args.model]
        if not models_to_run:
            raise SystemExit(
                f"Model '{args.model}' not found (or marked skip) in models.yaml"
            )

    lang_subset = None
    if args.langs:
        lang_subset = [l.strip() for l in args.langs.split(",") if l.strip()]
        unknown = [l for l in lang_subset if l not in LANG_MAP]
        if unknown:
            raise SystemExit(f"Unknown language codes: {unknown}")

    # Directory and Logging Setup
    # PROXY_RESULTS_DIR points at scratch on the cluster; falls back to the
    # original ./results for single-machine use.
    RESULTS_DIR = os.environ.get("PROXY_RESULTS_DIR", "./results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-model log file so parallel SLURM jobs don't clobber each other.
    log_suffix = f"_{args.model}" if args.model else ""
    LOG_FILE = os.path.join(RESULTS_DIR, f"translation_log{log_suffix}.txt")

    with open(LOG_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"LOG FILE:  {os.path.abspath(RESULTS_DIR)}\n")
        f.write("=" * 60 + "\n")

    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Logging to: {LOG_FILE}")

    for model_cfg in tqdm(models_to_run, desc="Total Progress (Models)", position=0):
        model_path = model_cfg["path"]
        model_name = model_cfg["name"]
        family = model_cfg["family"]
        tp_size = model_cfg["tp"]
        trust_remote_code = model_cfg.get("trust_remote_code", False)
        prompt_mode = model_cfg.get("prompt_mode", "chat")
        # Translation prompts are short (<<4k tokens); cap vLLM KV-cache reservation
        # so models with 131k–1M native context do not OOM on Leonardo A100s.
        max_model_len = model_cfg.get("max_model_len", 4096)
        # Disable CUDA graphs for models that crash during cudagraph capture
        # (e.g. DeepSeek-V2-Lite-Chat: cudaErrorStreamCaptureInvalidated) or that
        # deadlock mid-generation (Yi-1.5-34B-Chat EngineDeadError at tp=2).
        enforce_eager = model_cfg.get("enforce_eager", False)

        log(f"\n{'=' * 60}")
        log(f"LOADING: {model_name} (TP={tp_size}, max_model_len={max_model_len})")
        log(f"{'=' * 60}")

        # DeepSeek-V2-Lite (MLA) otherwise pulls in the flashinfer rope kernel,
        # which JIT-compiles via the hardcoded /usr/local/cuda/bin/nvcc that
        # env.sh strips on offline nodes. TRITON_MLA keeps attention off the
        # flashinfer JIT path.
        if model_name == "DeepSeek-V2-Lite-Chat":
            os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_MLA"
        else:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)

        try:
            # 1. Load Tokenizer (slow backend avoids fast-tokenizer conversion
            # failures on Moonlight / OpenELM architectures).
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )

            # 2. Get YOUR Exact Params
            sampling_params = get_sampling_params(family)

            # 3. Load vLLM
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                dtype="bfloat16",
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.90,
            )

            # 4. Run Translations

            langs_to_process = [l for l in LANG_MAP if l != SOURCE_CODE]
            if lang_subset is not None:
                langs_to_process = [l for l in langs_to_process if l in lang_subset]

            for lang in tqdm(
                langs_to_process,
                desc=f"Languages ({model_name})",
                position=1,
                leave=False,
            ):
                translate_batch(
                    llm,
                    tokenizer,
                    sampling_params,
                    lang,
                    model_name,
                    prompt_mode,
                )

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
