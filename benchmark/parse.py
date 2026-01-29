import json
import re
import pandas as pd
import os
import glob

# Configuration
BASE_INPUT_DIR = "./results/raw/"
BASE_OUTPUT_DIR = "./results/parsed/"

MODELS_TO_PARSE = {
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

# Normalization Maps
iroko_bench_unnorm = ["swa", "orm"]
iroko_bench_to_norm = {"swa": "swh", "orm": "gaz"}

global_mmlu_lang_code = {
    "bn": "ben",
    "zh": "zho",
    "fr": "fra",
    "de": "deu",
    "hi": "hin",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "ar": "arb",
    "pt": "por",
    "es": "spa",
    "sw": "swh",
    "yo": "yor",
}

okapi_lang_map = {
    "ar": "arb",
    "bn": "ben",
    "ca": "cat",
    "da": "dan",
    "de": "deu",
    "es": "spa",
    "eu": "eus",
    "fr": "fra",
    "gu": "guj",
    "hi": "hin",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "id": "ind",
    "it": "ita",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "npi",
    "nl": "nld",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sr": "srp",
    "sk": "slk",
    "sv": "swe",
    "ta": "tam",
    "te": "tel",
    "uk": "ukr",
    "vi": "vie",
    "zh": "zho",
}

include_lang_map = {
    "albanian": "als",
    "arabic": "arb",
    "armenian": "hye",
    "azerbaijani": "azj",
    "basque": "eus",
    "belarusian": "bel",
    "bengali": "ben",
    "bulgarian": "bul",
    "chinese": "zho",
    "croatian": "hrv",
    "dutch": "nld",
    "estonian": "est",
    "finnish": "fin",
    "french": "fra",
    "georgian": "kat",
    "german": "deu",
    "greek": "ell",
    "hebrew": "heb",
    "hindi": "hin",
    "hungarian": "hun",
    "indonesian": "ind",
    "italian": "ita",
    "japanese": "jpn",
    "kazakh": "kaz",
    "korean": "kor",
    "lithuanian": "lit",
    "malay": "zsm",
    "malayalam": "mal",
    "nepali": "npi",
    "north macedonian": "mkd",
    "persian": "pes",
    "polish": "pol",
    "portuguese": "por",
    "russian": "rus",
    "serbian": "srp",
    "spanish": "spa",
    "tagalog": "tgl",
    "tamil": "tam",
    "telugu": "tel",
    "turkish": "tur",
    "ukrainian": "ukr",
    "urdu": "urd",
    "uzbek": "uzn",
    "vietnamese": "vie",
}

# Shared 2-letter ISO Map
iso_lang_map = {
    "ar": "arb",
    "bn": "ben",
    "ca": "cat",
    "da": "dan",
    "de": "deu",
    "es": "spa",
    "eu": "eus",
    "fr": "fra",
    "gu": "guj",
    "hi": "hin",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "kn": "kan",
    "ko": "kor",
    "ml": "mal",
    "mr": "mar",
    "ne": "npi",
    "nl": "nld",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sr": "srp",
    "sk": "slk",
    "sv": "swe",
    "sw": "swh",
    "ta": "tam",
    "te": "tel",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "vi": "vie",
    "yo": "yor",
    "zh": "zho",
}

# Extraction Functions


def process_afrimmlu(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^afrimmlu_direct_([a-z]+)_prompt_1$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("acc,none")
            if lang_code == "eng":
                continue
            if lang_code in iroko_bench_unnorm:
                lang_code = iroko_bench_to_norm[lang_code]
            if acc is not None:
                extracted_data.append({"lang_code": lang_code, "afrimmlu": acc})
    return extracted_data


def process_afrixnli(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^afrixnli_([a-z]+)_prompt_2$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("acc,none")
            if lang_code == "eng":
                continue
            if lang_code in iroko_bench_unnorm:
                lang_code = iroko_bench_to_norm[lang_code]
            if acc is not None:
                extracted_data.append({"lang_code": lang_code, "afrixnli": acc})
    return extracted_data


def process_belebele(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^belebele_([a-z]+)_([A-Za-z]+)$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            script_code = match.group(2)
            acc = metrics.get("acc,none")
            if lang_code == "eng":
                continue
            if lang_code == "zho" and script_code == "Hant":
                lang_code = "zho_trad"
            if acc is not None:
                extracted_data.append({"lang_code": lang_code, "belebele": acc})
    return extracted_data


def process_global_mmlu(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^global_mmlu_([a-z]{2})$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("acc,none")
            if lang_code == "en":
                continue
            if acc is not None and lang_code in global_mmlu_lang_code:
                norm_code = global_mmlu_lang_code[lang_code]
                extracted_data.append({"lang_code": norm_code, "global_mmlu": acc})
    return extracted_data


def process_hellaswag(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^hellaswag_([a-z]{2})$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("acc_norm,none") or metrics.get("acc,none")
            if lang_code == "en":
                continue
            if acc is not None:
                norm_code = okapi_lang_map.get(lang_code, lang_code)
                extracted_data.append({"lang_code": norm_code, "hellaswag": acc})
    return extracted_data


def process_truthfulqa(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^truthfulqa_([a-z]{2})_mc2$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("acc,none")
            if lang_code == "en":
                continue
            if acc is not None:
                norm_code = okapi_lang_map.get(lang_code, lang_code)
                extracted_data.append({"lang_code": norm_code, "truthfulqa": acc})
    return extracted_data


def process_mgsm(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^mgsm_direct_([a-z]{2})$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("exact_match,flexible-extract")
            if lang_code == "en":
                continue
            if acc is not None:
                norm_code = iso_lang_map.get(lang_code, lang_code)
                extracted_data.append({"lang_code": norm_code, "mgsm": acc})
    return extracted_data


def process_mlqa(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^mlqa_en_([a-z]{2})$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            lang_code = match.group(1)
            acc = metrics.get("f1,none")
            if lang_code == "en":
                continue
            if acc is not None:
                norm_code = iso_lang_map.get(lang_code, lang_code)
                extracted_data.append({"lang_code": norm_code, "mlqa": acc})
    return extracted_data


def process_include(raw_data):
    results = raw_data.get("results", {})
    extracted_data = []
    pattern = re.compile(r"^include_base_44_([a-z ]+)$")

    for task_name, metrics in results.items():
        match = pattern.match(task_name)
        if match:
            full_name = match.group(1)
            acc = metrics.get("acc,none")
            if acc is not None and full_name in include_lang_map:
                norm_code = include_lang_map[full_name]
                extracted_data.append({"lang_code": norm_code, "include": acc})
    return extracted_data


def find_latest_json(base_dir, model_key, hf_path):
    # Sanitize HF path: replace '/' with '__'
    sanitized_hf = hf_path.replace("/", "__")

    # Construct search path: Base / ModelKey / SanitizedHF / results_*.json
    # Example: .../Qwen3-32B/Qwen__Qwen3-32B/results_*.json
    search_pattern = os.path.join(base_dir, model_key, sanitized_hf, "results_*.json")

    files = glob.glob(search_pattern)

    if not files:
        return None

    # If multiple result files exist, pick the latest one based on modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def parse_single_model(model_key, json_path, output_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        all_results = list()
        all_results.extend(process_afrimmlu(data))
        all_results.extend(process_afrixnli(data))
        all_results.extend(process_belebele(data))
        all_results.extend(process_global_mmlu(data))
        all_results.extend(process_hellaswag(data))
        all_results.extend(process_truthfulqa(data))
        all_results.extend(process_mgsm(data))
        all_results.extend(process_mlqa(data))
        all_results.extend(process_include(data))

        if not all_results:
            print(f"  [WARN] No metrics found in {os.path.basename(json_path)}")
            return

        df = pd.DataFrame(all_results)
        df_final = df.groupby("lang_code").first().reset_index()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"  [OK] Saved {model_key} -> {output_path}")

    except Exception as e:
        print(f"  [ERR] Failed to process {model_key}: {e}")


def main():
    print(f"Starting batch processing for {len(MODELS_TO_PARSE)} models...\n")

    for model_key, hf_path in MODELS_TO_PARSE.items():
        # 1. Find the input JSON
        json_file = find_latest_json(BASE_INPUT_DIR, model_key, hf_path)

        if not json_file:
            print(
                f"[SKIP] Could not find result file for: {model_key} (Path checked: {hf_path})"
            )
            continue

        # 2. Define output path
        output_csv = os.path.join(BASE_OUTPUT_DIR, f"{model_key}.csv")

        # 3. Process
        parse_single_model(model_key, json_file, output_csv)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
