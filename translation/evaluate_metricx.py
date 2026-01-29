import os
import json
import pandas as pd
import subprocess
import tempfile
import numpy as np
import re

DATASET_FOLDERS = ["flores-200", "wmt24", "ntrex"]
METRICS_SAVE_DIR = "./results/metrics"

MODELS_TO_CHECK = {
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
    "Phi-4": "microsoft/phi-4",
}


METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
METRICX_TOKENIZER = "google/mt5-xl"
METRICX_MAX_LENGTH = 1536
METRICX_BATCH_SIZE = 1

METRICX_REPO_DIR = "./metricx"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Helper Functions
def clean_translation_text(text):
    if not isinstance(text, str):
        return str(text)

    return re.sub(r"(?s).*?</think>", "", text).strip()


def csv_to_jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)
    # Ensure strings
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["translation"] = df["translation"].astype(str)

    df["translation"] = df["translation"].apply(clean_translation_text)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            data = {
                "source": row["source"],
                "reference": row["target"],
                "hypothesis": row["translation"],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def run_metricx(input_jsonl, output_jsonl):
    cmd = [
        "python",
        "-m",
        "metricx24.predict",
        "--tokenizer",
        METRICX_TOKENIZER,
        "--model_name_or_path",
        METRICX_MODEL,
        "--max_input_length",
        str(METRICX_MAX_LENGTH),
        "--batch_size",
        str(METRICX_BATCH_SIZE),
        "--input_file",
        input_jsonl,
        "--output_file",
        output_jsonl,
    ]

    if not os.path.isdir(METRICX_REPO_DIR):
        raise FileNotFoundError(f"MetricX repo dir not found: {METRICX_REPO_DIR}")

    # Capture output to print real error if it fails
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, cwd=METRICX_REPO_DIR
    )
    return result


def extract_metricx_scores(output_jsonl):
    scores = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            scores.append(json.loads(line)["prediction"])
    return scores


def calculate_metricx_corpus_score(scores):
    if not scores:
        return None
    return sum(scores) / len(scores)


# Main Logic

if __name__ == "__main__":
    print(f"Starting MetricX Backfill... (Repo: {METRICX_REPO_DIR})")

    for ds_folder in DATASET_FOLDERS:
        print(f"\n{'=' * 40}\nDATASET: {ds_folder}\n{'=' * 40}")
        save_filename = "wmt.csv" if ds_folder == "wmt24" else f"{ds_folder}.csv"

        for model_name in MODELS_TO_CHECK.keys():
            model_metrics_dir = os.path.join(METRICS_SAVE_DIR, model_name)
            metrics_csv_path = os.path.join(model_metrics_dir, save_filename)
            translations_dir = f"./results/translations/{ds_folder}/{model_name}/"

            if not os.path.exists(metrics_csv_path):
                print(f"  - Metrics file not found for {model_name}. Skipping.")
                continue

            # Load existing metrics
            df = pd.read_csv(metrics_csv_path)

            # Create 'metricx' column if it doesn't exist
            if "metricx" not in df.columns:
                df["metricx"] = np.nan

            # Filter rows that need MetricX (NaN or empty string)
            # We iterate by index to update the DataFrame in place
            rows_to_update = df[
                pd.to_numeric(df["metricx"], errors="coerce").isna()
            ].index

            if len(rows_to_update) == 0:
                print(f"    {model_name} is fully up to date.")
                continue

            print(
                f"    Processing {len(rows_to_update)} missing rows for {model_name}..."
            )

            for idx in rows_to_update:
                row = df.loc[idx]
                pair_name = row["translation-pair"]
                src_lang, tgt_lang = pair_name.split("-")

                # Locate source translation file
                source_csv_path = os.path.join(translations_dir, f"{pair_name}.csv")

                if not os.path.exists(source_csv_path):
                    print(
                        f"    [WARN] Translation file not found: {source_csv_path}. Skipping."
                    )
                    continue

                print(f"      Calculating MetricX for {pair_name}...")

                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        input_jsonl = os.path.abspath(
                            os.path.join(tmpdir, "input.jsonl")
                        )
                        output_jsonl = os.path.abspath(
                            os.path.join(tmpdir, "output.jsonl")
                        )

                        csv_to_jsonl(source_csv_path, input_jsonl)
                        run_metricx(input_jsonl, output_jsonl)

                        scores = extract_metricx_scores(output_jsonl)
                        corpus_score = calculate_metricx_corpus_score(scores)

                        if corpus_score is not None:
                            # INVERT SCORE (Higher is Better)
                            final_score = round(25 - corpus_score, 4)
                            df.at[idx, "metricx"] = final_score

                            # Save immediately (checkpointing)
                            df.to_csv(metrics_csv_path, index=False)
                            # print(f"      Saved score: {final_score}")

                except subprocess.CalledProcessError as e:
                    print("      [ERROR] MetricX Crashed!")
                    print(f"      STDOUT: {e.stdout}")
                    print(
                        f"      STDERR: {e.stderr}"
                    )  # This will show you exactly WHY it failed
                except Exception as e:
                    print(f"      [ERROR] Python Error: {e}")

    print("\nBackfill complete!")
