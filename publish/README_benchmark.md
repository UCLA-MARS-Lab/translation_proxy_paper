---
pretty_name: Proxy-MT Benchmark Scores
license: other
task_categories:
  - text-generation
  - multiple-choice
  - question-answering
tags:
  - evaluation
  - lm-eval-harness
  - african-languages
  - multilingual
  - benchmark
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

# Proxy-MT Benchmark Scores

Multilingual benchmark results for **50 open-weight LLMs**, evaluated with the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via a vLLM
backend. Covers reasoning, comprehension, and knowledge tasks with an emphasis on
African and other lower-resource languages.

## Layout

```
scores/<model>.csv          # parsed per-language scores (tidy, ready to plot)
raw/<model>/.../results_*.json   # raw lm-eval-harness result files
raw/<model>/raw_log.txt          # full evaluation console log (provenance)
```

`lm_cache/` request caches are intentionally excluded.

### `scores/<model>.csv`

One row per language code with a column per benchmark metric, e.g.:

```
lang_code,mgsm,belebele,global_mmlu,hellaswag,truthfulqa,mlqa,afrixnli,include
ben,0.125,...
```

## Benchmarks parsed

AfriXNLI, Belebele, Global-MMLU, HellaSwag, TruthfulQA, MGSM, MLQA, INCLUDE.

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

> Note: set the `license` field above to the license you intend to release under.
