---
pretty_name: Proxy-MT Translations
license: other
task_categories:
  - translation
language:
  - af
  - am
  - ar
  - bm
  - bn
  - bg
  - ca
  - ceb
  - cs
  - cy
  - da
  - de
  - el
  - et
  - eu
  - ee
  - fi
  - fr
  - gu
  - ha
  - he
  - hi
  - hr
  - hu
  - hy
  - ig
  - id
  - is
  - it
  - ja
  - jv
  - kn
  - ka
  - kk
  - km
  - rw
  - ky
  - ko
  - lo
  - ln
  - lg
  - luo
  - lv
  - ml
  - mr
  - mk
  - mt
  - mi
  - my
  - nl
  - "no"
  - ny
  - or
  - pa
  - fa
  - pl
  - pt
  - ro
  - ru
  - sn
  - si
  - sk
  - sl
  - sd
  - so
  - st
  - es
  - sr
  - ss
  - su
  - sv
  - sw
  - ta
  - te
  - tg
  - tl
  - th
  - ti
  - tn
  - ts
  - tr
  - tw
  - uk
  - ur
  - uz
  - vi
  - war
  - wo
  - xh
  - yo
  - zh
  - zu
tags:
  - machine-translation
  - african-languages
  - flores-200
  - ntrex
  - wmt24
  - low-resource
size_categories:
  - 10M<n<100M
extra_gated_prompt: >-
  You agree to use these model-generated translations for research purposes and
  to cite the source. Access is granted on request and reviewed manually.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use this dataset for non-commercial research only: checkbox
---

# Proxy-MT Translations

English→X machine translations generated with [vLLM](https://github.com/vllm-project/vllm)
across **50 open-weight LLMs** on three evaluation benchmarks. This dataset holds the
**raw model outputs** (one CSV per model × dataset × target language); metric scores
(BLEU / chrF / COMET / MetricX) live in `proxy-mt-eval-scores`.

## Layout

```
flores-200/<model>/eng-<lang>.csv   # 119 target languages
ntrex/<model>/eng-<lang>.csv        #  87 target languages
wmt24/<model>/eng-<lang>.csv        #  51 target languages
```

Each CSV has three columns:

| column | description |
|--------|-------------|
| `source` | English source sentence |
| `target` | reference translation from the benchmark |
| `translation` | model-generated translation |

## Source benchmarks

- **FLORES-200** (`facebook/flores`, `devtest` split)
- **NTREX** (`mteb/NTREX`, `test` split)
- **WMT24++** (`google/wmt24pp`, `train` split)

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
