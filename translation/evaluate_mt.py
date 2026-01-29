import os
import json
import pandas as pd
import torch
import sacrebleu
import evaluate
from comet import download_model, load_from_checkpoint
import datetime
import traceback
from tqdm import tqdm
import re


# Configuration
MODELS_TO_EVALUATE = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Gemma-3-1B-it": "google/gemma-3-1b-it",
    "Gemma-3-4B-it": "google/gemma-3-4b-it",
    "Gemma-3-12B-it": "google/gemma-3-12b-it",
    "Gemma-3-27B-it": "google/gemma-3-27b-it",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Mistral-Small-24B-Instruct-2501": "mistralai/Mistral-Small-24B-Instruct-2501",
    "Phi-4": "microsoft/phi-4",
}


DATASET_FOLDERS = ["flores-200", "wmt24", "ntrex"]
METRICS_SAVE_DIR = "./results/metrics"


def log(message):
    """Prints to console and appends to a log file."""
    tqdm.write(message)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")


def clean_translation_text(text):
    """
    Cleans reasoning traces using regex
    """
    if not isinstance(text, str):
        return str(text)

    return re.sub(r"(?s).*?</think>", "", text).strip()


# Metric Calculation Function
def evaluate_metrics(
    csv_path,
    csv_save_path,
    src_lang,
    tgt_lang,
    meteor_scorer,
    rouge_scorer,
    xcomet_scorer,
    ssa_comet_scorer,
):
    """Calculates metrics and appends a single row to the model's dataset CSV."""
    df = pd.read_csv(csv_path)

    # get references, translations and sources
    references = df["target"].astype(str).tolist()
    raw_translations = df["translation"].astype(str).tolist()
    sources = df["source"].astype(str).tolist()

    translations = [clean_translation_text(t) for t in raw_translations]

    # Language-Agnostic Metrics
    bleu = round(
        sacrebleu.corpus_bleu(translations, [references], tokenize="flores200").score, 2
    )
    chrf = round(sacrebleu.corpus_chrf(translations, [references]).score, 2)
    rouge_l = round(
        rouge_scorer.compute(predictions=translations, references=references)["rougeL"],
        4,
    )

    print("    - Running METEOR...")
    meteor = round(
        meteor_scorer.compute(predictions=translations, references=references)[
            "meteor"
        ],
        4,
    )

    # COMET Data format
    comet_data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, translations, references)
    ]

    print("    - Running XCOMET...")
    xcomet_res = xcomet_scorer.predict(comet_data, batch_size=128)
    xcomet_score = round(xcomet_res.system_score, 4)

    print("    - Running SSA-COMET...")
    ssa_res = ssa_comet_scorer.predict(comet_data, batch_size=128)
    ssa_score = round(ssa_res.system_score, 4)

    # Prepare Row
    new_row = {
        "translation-pair": f"{src_lang}-{tgt_lang}",
        "bleu": bleu,
        "chrf++": chrf,
        "rouge-l": rouge_l,
        "meteor": meteor,
        "xcomet": xcomet_score,
        "ssa-comet": ssa_score,
        "metricx": "",
    }

    # Append to CSV
    results_df = pd.DataFrame([new_row])
    file_exists = os.path.isfile(csv_save_path)

    results_df.to_csv(
        csv_save_path, mode="a", index=False, header=not file_exists, encoding="utf-8"
    )


# 3. Main Execution
if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    LOG_DIR = "./logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f"eval_log_{current_time}.log")

    with open(LOG_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVAL RUN STARTED: {current_time}\n")
        f.write(f"LOG FILE:  {os.path.abspath(LOG_FILE)}\n")
        f.write("=" * 60 + "\n")

    print(f"Logging to: {LOG_FILE}")

    log("Preloading evaluation models...")
    meteor_scorer = evaluate.load("meteor")
    rouge_scorer = evaluate.load("rouge")

    log("- Loading XCOMET (XL)...")
    xcomet_path = download_model("Unbabel/XCOMET-XL")
    xcomet_scorer = load_from_checkpoint(xcomet_path)

    log("- Loading SSA-COMET...")
    ssa_path = download_model("McGill-NLP/ssa-comet-mtl")
    ssa_comet_scorer = load_from_checkpoint(ssa_path)

    # Loop through each Dataset type
    for ds_folder in DATASET_FOLDERS:
        log("\n" + "#" * 40)
        log(f"DATASET: {ds_folder}")
        log("#" * 40)

        save_filename = f"{ds_folder}.csv"

        # Loop through each Model
        model_list = list(MODELS_TO_EVALUATE.keys())
        for model_name in tqdm(model_list, desc=f"Models ({ds_folder})", position=0):
            translations_dir = f"results/translations/{ds_folder}/{model_name}/"

            model_metrics_dir = os.path.join(METRICS_SAVE_DIR, model_name)
            os.makedirs(model_metrics_dir, exist_ok=True)
            csv_save_path = os.path.join(model_metrics_dir, save_filename)

            if not os.path.isdir(translations_dir):
                log(
                    f"[WARN] Skipping {model_name} for {ds_folder}: Directory not found."
                )
                continue

            csv_files = [f for f in os.listdir(translations_dir) if f.endswith(".csv")]

            if not csv_files:
                log(f"[WARN] No CSV files found for {model_name} in {translations_dir}")
                continue

            for csv_file in tqdm(
                sorted(csv_files), desc=f"Pairs ({model_name})", position=1, leave=False
            ):
                try:
                    # Extract lang pair (eng-afr) from filename
                    pair_name = csv_file.replace(".csv", "")
                    src_lang, tgt_lang = pair_name.split("-")

                    # Check if row already exists
                    if os.path.exists(csv_save_path):
                        existing = pd.read_csv(csv_save_path)
                        if pair_name in existing["translation-pair"].values:
                            print(f"  {pair_name} already evaluated. Skipping.")
                            continue

                    with open(LOG_FILE, "a") as f:
                        f.write(f"  Processing {pair_name} on {model_name}...\n")

                    full_csv_path = os.path.join(translations_dir, csv_file)

                    evaluate_metrics(
                        full_csv_path,
                        csv_save_path,
                        src_lang,
                        tgt_lang,
                        meteor_scorer,
                        rouge_scorer,
                        xcomet_scorer,
                        ssa_comet_scorer,
                    )

                except Exception as e:
                    log(f"  [ERROR] Failed to process {csv_file}: {e}")

    log("\nAll evaluations complete!")
