## README: Translation as a Scalable Proxy for Multilingual Evaluation

This repository contains code and instructions needed to replicate the experiments presnted in our paper: **"Translation as a Scalable Proxy for Multilingual Evaluation"**.

## Abstract

> The rapid proliferation of LLMs has created a critical evaluation paradox: while LLMs claim multilingual proficiency, comprehensive non-machine-translated benchmarks exist for fewer than 30 languages, leaving >98% of the world's 7,000 languages in an empirical void. Traditional benchmark construction faces scaling challenges such as cost, scarcity of domain experts, and data contamination. We evaluate the validity of a simpler alternative: _can translation quality alone indicate a model's broader multilingual capabilities?_ Through systematic evaluation of 14 models (1B-72B parameters) across 9 diverse benchmarks and 7 translation metrics, we find that translation performance is a good indicator of downstream task success (e.g., Phi-4, median Pearson _r_: MetricX = 0.89, xCOMET = 0.91, SSA-COMET = 0.87). These results suggest that the representational abilities supporting faithful translation overlap with those required for multilingual understanding. Translation quality, thus emerges as a strong, inexpensive first-pass proxy of multilingual performance, enabling a _translation-first screening_ with targeted follow-up for specific tasks.

---

## Setup and Installation

### Prerequisites

- **Python 3.9+** and **Conda** (or Miniconda)
- **NVIDIA GPU with CUDA 12.x** for LLM inference (e.g., `vllm`).
- **Hugging Face Hub CLI and Datasets library**
- **High Storage Capacity**: Replicating this study requires downloading all 14 LLMs and 3 parallel corpora.

### Environment Setup

Two separate environments are required to replicate all experiments due to dependency conflicts, particularly with the `MetricX` evaluation framework.

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Create the Main Environment (`proxy_main`):**
    This enviornment is used for all LLM evaluation (`lm-eval`), machine translation, and most MT metric calculations (BLEU, xCOMET, etc.).

    ```bash
    conda env create -f proxy_main.yml
    ```

3.  **Create the MetricX Environment (`proxy_metricx`):**
    This environment is required **only** for the calculation of the **MetricX** score due to its specific dependencies.

    ```bash
    conda env create -f proxy_metricx.yml
    ```

    Additionally, MetricX is run from its local script, not as an installed package. Clone the repository into your project's root directory:

    ```bash
    git clone https://github.com/google-research/metricx.git
    ```

## Download Models and Datasets

This is the most storage-intensive step. Activate the main enviornment before you download anything.

We provide scripts to download all required assets from Hugging Face. If you do not have the stoage capabilities, we recommend, downloading the number of models you can, and continuing the experiment from there.

### 1. Download LLMs (14 Models)

Before you run this script, make sure you request permission from the huggingface repos, and set your huggingface token, and hugging face path.
Run the script:

```bash
./download_llms.sh
```

### 2. Download Parallel Corpora (3 Datasets)

This script uses the datasets library to download the three parallel corpora.

```
python download_datasets.py
```

## Experiment 1: Multilingual Benchmarking

This section details how to replicate the multilingual LLM evaluation results. This process is managed by a script that iterates through all models.

### 1. Data Preparation

The benchmarks are defined within the scripts called by benchmark.sh. The 9 primary benchmarks are:

- Global MMMLU
- Belebele
- MGSM
- AfriMMLU
- AfriXNLI
- HellaSwag
- TruthfulQA
- MLQA
- INCLUDE

The required datasets for these benchmarks are public and will be downloaded automatically by the lm-eval scripts on their first run.

### 2. Execution

The main entry point for running all benchmarks across all 14 models is the `run_benchmarks.sh` script. It accepts a GPU list as a parameter for vLLM inference. Results are saved to individual model directories within the ./results/ folder

```bash
conda activate proxy_main

# Make the script executable
chmod +x ./benchmarks/run_benchmarks.sh

# Run benchmarks for all 14 models on specific GPUs
./benchmarks/run_benchmarks.sh "0,1,2,3"
```

**Benchmark Retrieval**

This step extracts the raw scores from the experiment's output. What it does: Finds the latest results\_\*.json file for each model outputted by `lm-eval`.

- Normalizes language codes to ISO 639-3 (e.g., ar → arb).
- Extracts specific metrics (acc, f1, etc.) and merges them into a single table.
- Saves the results as a standardized CSV in ./results/parsed/.

```bash
python ./benchmarks/parse.py
```

## Experiment 2: Machine Language Translation and Metrics

This experiment is a two-stage process:

1. Generation: Use the 14 LLMs to generate translations for the 3 parallel corpora.
2. Evaluation: Calculate the 7 translation metrics (MetricX, XCOMET, etc.) on the generated translations.

### 1. Translation Generation

This step uses the provided Python script (e.g., run_translations.py) which leverages vllm for high-throughput inference. The script will iterate through all 14 models and generate translations for all language pairs in the Flores-200, WMT24++, and NTREX datasets

```bash
python ./translation/run_translation.py
```

### 2. Metric Evaluation

After translations are generated, run the metric evaluation scripts on the output CSV files.

The scripts (e.g., evaluate_mt.py) should be pointed to the translations/ directory to read the source, target, and (model-generated) translation columns.

#### A. Standard Metrics (COMET, BLEU, etc.)

Run the evaluation for XCOMET, SSA-COMET, chrF++, METEOR, ROUGE-L, and BLEU in the main environment (mars_lab).

```bash
python ./translation/evaluate_mt.py
```

#### B. MetricX Calculation

Deactivate the main environment and activate the dedicated proxy_metricx environment. This script will find the csv files created in the previous step, calculate the missing MetricX score, and update the csv files in place.

```bash
conda deactivate
conda activate proxy_metricx

python ./translation/evaluate_metricx.py
```

# Expected Results and Analysis

Following the execution of the experiments, the ./results/ directory will contain the raw data required to reproduce the paper's correlation analysis.

### 1. Multilingual Benchmark Scores

After running parse.py, you will find standardized CSV files in `./results/parsed/` for each model (e.g., Phi-4.csv).

- Format: Each file contains a lang_code (ISO 639-3) column followed by columns for each benchmark (afrimmlu, belebele, mgsm, etc.).
- Consistency: The script automatically handles the mapping of inconsistent codes (e.g., swa → swh) to ensure language-aligned comparisons across different datasets.

### 2. Translation Quality Metrics

The MT experiment produces a multi-layered results structure:

- **Raw Translations**: Located in folders named after the source dataset (e.g., ./flores-200/[model_name]/), containing translation pairs for every language.
- **Aggregated Metrics**: The evaluate_mt.py script creates a metrics/ directory. For each model, a CSV (e.g., metrics/Phi-4/flores-200.csv) is generated containing scores for BLEU, chrF++, METEOR, ROUGE-L, xCOMET, and SSA-COMET.
- **MetricX Integration**: After running the backfill script (evaluate_metricx.py), the metricx column in these CSVs will be updated. Note that the script inverts the raw MetricX error score (calculating 25 - score) so that a higher value indicates better quality, aligning it with the other 6 metrics.

# Citation

If you use this code or the results in your research, please cite our paper:

```
@article{issaka2026translation,
    title={Translation as a Scalable Proxy for Multilingual Evaluation},
    author={Issaka, Sheriff and Rosas Gonzalez, Erick and Liu, Lieqi and Agyei, Evans Kofi and Bandarkar, Lucas and Peng, Nanyun and Adelani, David Ifeoluwa and Guzmán, Francisco and Gabriel, Saadia},
    journal={Preprint},
    year={2026},
    url={https://translation-as-multilingual-proxy.github.io/}
}
```
