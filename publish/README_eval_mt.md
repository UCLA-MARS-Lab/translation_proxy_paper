---
pretty_name: Proxy-MT Eval Scores
license: other
task_categories:
  - translation
tags:
  - machine-translation
  - evaluation
  - bleu
  - comet
  - flores-200
  - ntrex
  - wmt24
  - african-languages
size_categories:
  - n<1K
extra_gated_prompt: >-
  You agree to use these evaluation results for research purposes and to cite the
  source. Access is granted on request and reviewed manually.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use this dataset for non-commercial research only: checkbox
---

# Proxy-MT Eval Scores

Corpus-level MT metrics for **50 open-weight LLMs** on the translations in
[`proxy-mt-translations`](https://huggingface.co/datasets/African-Languages-Lab/proxy-mt-translations).
Computed by `evaluate_mt.py` (BLEU, chrF++, ROUGE-L, METEOR, XCOMET-XL, SSA-COMET).
MetricX is backfilled separately and may still be empty in this snapshot.

## Layout

```
<model>/flores-200.csv
<model>/ntrex.csv
<model>/wmt24.csv
```

Each CSV has one row per `eng-<lang>` pair:

| column | description |
|--------|-------------|
| `translation-pair` | e.g. `eng-yor` |
| `bleu` | sacrebleu corpus BLEU (`tokenize=flores200`) |
| `chrf++` | sacrebleu corpus chrF |
| `rouge-l` | HuggingFace `evaluate` ROUGE-L |
| `meteor` | HuggingFace `evaluate` METEOR |
| `xcomet` | Unbabel XCOMET-XL system score |
| `ssa-comet` | McGill-NLP SSA-COMET-MTL system score |
| `metricx` | MetricX-24-XL (higher-is-better, `25 - raw`); may be blank |

`meteor == -99` marks four degenerate outputs where NLTK METEOR hit a recursion
limit (DeepSeek-R1-Distill-Qwen-7B `eng-heb`/`eng-snd`, Falcon3-7B `eng-mal`,
Olmo-3-7B `eng-spa`). Other metrics on those rows are valid.

## Models (50)

aya-expanse-32b, aya-expanse-8b, tiny-aya-base, c4ai-command-r-08-2024,
DeepSeek-R1-0528-Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-1.5B,
DeepSeek-R1-Distill-Qwen-14B, DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Qwen-7B,
deepseek-llm-7b-chat, Falcon3-10B-Instruct, Falcon3-3B-Instruct, Falcon3-7B-Instruct,
gemma-3-12b-it, gemma-3-1b-it, gemma-3-27b-it, gemma-3-4b-it, gemma-4-26B-A4B-it,
gemma-4-31B-it, gemma-4-E2B-it, gemma-4-E4B-it, granite-4.1-30b, granite-4.1-8b,
granite-4.1-3b, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct,
Ministral-3-14B-Instruct-2512, Ministral-3-3B-Instruct-2512, Ministral-3-8B-Instruct-2512,
Mistral-7B-Instruct-v0.3, Mistral-Nemo-Instruct-2407, Mixtral-8x7B-Instruct-v0.1,
Olmo-3-7B-Instruct, Olmo-3.1-32B-Instruct, Olmo-3-1125-32B, Yi-1.5-9B-Chat,
Yi-1.5-34B-Chat, internlm2_5-7b-chat, internlm2_5-20b-chat, glm-4-9b-chat, phi-4,
Qwen3-1.7B, Qwen3-14B, Qwen3-30B-A3B, Qwen3-4B, Qwen3-8B, Qwen3.6-27B, Qwen3.6-35B-A3B.

## Access

This is a **gated** dataset — access is reviewed and approved manually. Fill in the
request form to be granted access.
