---
tags:
- techwithsergiu
- gguf
- qwen3_5_text
library_name: gguf
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-{SIZE}/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- techwithsergiu/Qwen3.5-text-{SIZE}
---

# Qwen3.5-text-{SIZE}-GGUF

<img width="400px" src="https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png">

GGUF quants of [techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}) —
the text-only bf16 derivative of [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}).

The visual tower has been removed before conversion. All text-backbone weights are
**identical** to the original — no retraining, no weight changes, no quality loss for
text tasks.

## Quants

| File | Type | Size | Notes |
|---|---|---|---|
| `Qwen3.5-text-{SIZE}-Q8_0.gguf` | Q8_0 | ~53% of f16 | near-lossless — for high-quality inference |
| `Qwen3.5-text-{SIZE}-Q6_K.gguf` | Q6_K | ~41% of f16 | excellent quality, good balance with f16 |
| `Qwen3.5-text-{SIZE}-Q5_K_M.gguf` | Q5_K_M | ~37% of f16 | very good quality, smaller than Q6 |
| `Qwen3.5-text-{SIZE}-Q4_K_M.gguf` | Q4_K_M | ~31% of f16 | ✅ recommended — best size/quality balance |
| `Qwen3.5-text-{SIZE}-Q4_K_S.gguf` | Q4_K_S | ~30% of f16 | optional — slightly smaller, slightly lower quality |

## Model family

```mermaid
graph LR
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · VLM"]
    BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
    TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
    TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only"]
    GGUF["Qwen3.5-text-{size}-GGUF<br/>Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q4_K_S"]

    SRC -->|"base_model"| BNBVLM
    SRC -->|"base_model"| TEXTF16
    TEXTF16 -->|"base_model"| TEXTBNB
    TEXTF16 -->|"base_model"| GGUF
```

| Model | Type | Base model |
|---|---|---|
| [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}) | f16 · VLM · source | — |
| [techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit) | BNB NF4 · VLM | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}) | bf16 · text-only | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit) | BNB NF4 · text-only | Qwen3.5-text-{SIZE} |
| **[techwithsergiu/Qwen3.5-text-{SIZE}-GGUF](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-GGUF)** | GGUF quants | Qwen3.5-text-{SIZE} |

The GGUF repo is derived from the text-only f16 model — same weights, different container
format. `base_model` points to the f16 text variant to keep the VLM and text lineages
distinct on the Hub.

## Inference

### llama.cpp

```bash
./llama.cpp/build/bin/llama-cli \
    -m Qwen3.5-text-{SIZE}-Q4_K_M.gguf \
    -p "What is the capital of Romania?" \
    -n 256
```

### LM Studio

Load any `.gguf` file from this repo directly in [LM Studio](https://lmstudio.ai).
Recommended quant: `Q4_K_M`.

### Thinking mode

Qwen3.5 supports an optional chain-of-thought `<think>` block before the answer.
Thinking is **enabled by default** in llama.cpp.

**Note:** `--chat-template-kwargs '{"enable_thinking":...}'` is deprecated — do not use.
**Known issue:** `--reasoning off` is accepted but does not actually disable thinking.
**Workaround:** use `--reasoning-budget 0` — this reliably disables the `<think>` block.
Track the bug at [llama.cpp issues](https://github.com/ggml-org/llama.cpp/issues).

```bash
# Thinking OFF — direct answer (workaround: --reasoning-budget 0)
./llama.cpp/build/bin/llama-cli \
    -m Qwen3.5-text-{SIZE}-Q4_K_M.gguf \
    --reasoning-budget 0 \
    -p "What is the capital of Romania?" \
    -n 256

# Thinking ON — default, no flag needed
./llama.cpp/build/bin/llama-cli \
    -m Qwen3.5-text-{SIZE}-Q4_K_M.gguf \
    -p "What is 17 × 34?" \
    -n 1024
```

## Pipeline diagram

```mermaid
flowchart TD
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · source"]

    subgraph PATH_A ["Path A — BNB text-only  (training target)"]
        BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
        TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only"]
        V1{{"verified"}}
        V3{{"verified"}}
        BNBVLM -->|"qwen35-strip --mode bnb"| TEXTBNB
        BNBVLM -->|"qwen35-verify-qwen35"| V1
        TEXTBNB -->|"qwen35-verify"| V3
    end

    subgraph PATH_B ["Path B — f16 text-only + GGUF  (inference / merge base)"]
        TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
        V2{{"verified"}}
        GGUUF16["Qwen3.5-text-{size}.gguf<br/>GGUF f16"]
        Q4["Q4_K_M"]
        Q5KM["Q5_K_M"]
        Q6K["Q6_K"]
        Q8["Q8_0"]
        TEXTF16 -->|"qwen35-verify"| V2
        TEXTF16 -->|"convert_hf_to_gguf.py"| GGUUF16
        GGUUF16 -->|"llama-quantize"| Q4
        GGUUF16 -->|"llama-quantize"| Q5KM
        GGUUF16 -->|"llama-quantize"| Q6K
        GGUUF16 -->|"llama-quantize"| Q8
    end

    HUB[("HuggingFace Hub")]

    SRC -->|"qwen35-convert"| BNBVLM
    SRC -->|"qwen35-strip --mode f16"| TEXTF16
    V1 -->|"qwen35-upload"| HUB
    V3 -->|"qwen35-upload"| HUB
    V2 -->|"qwen35-upload"| HUB
    Q4   -->|"qwen35-upload"| HUB
    Q5KM -->|"qwen35-upload"| HUB
    Q6K  -->|"qwen35-upload"| HUB
    Q8   -->|"qwen35-upload"| HUB
```

## From fine-tuned adapter to GGUF

If you have a LoRA adapter trained with
[qwen-qlora-train](https://techwithsergiu.github.io/qwen-qlora-train),
merge it first, then convert to GGUF:

```bash
# 1. Merge adapter into f16 weights
qlora-merge \
  --base  Qwen/Qwen3.5-{SIZE} \
  --adapter adapters/<run_name> \
  --output merged/qwen35-text-{SIZE}-sft-f16

# 2. Convert merged model to GGUF  (requires llama.cpp)
python llama.cpp/convert_hf_to_gguf.py merged/qwen35-text-{SIZE}-sft-f16 \
    --outtype f16 \
    --outfile merged/qwen35-text-{SIZE}-sft-F16.gguf

# 3. Quantize
./llama.cpp/build/bin/llama-quantize \
    merged/qwen35-text-{SIZE}-sft-F16.gguf \
    merged/qwen35-text-{SIZE}-sft-Q4_K_M.gguf \
    Q4_K_M
```

Full post-training workflow is documented in
[qwen-qlora-train → Post-merge workflow](https://techwithsergiu.github.io/qwen-qlora-train/post-merge-workflow.html).

## Conversion

Converted using [qwen35-toolkit](https://techwithsergiu.github.io/qwen35-toolkit) —
a Python toolkit for BNB quantization, visual tower removal, verification and
HF Hub publishing of Qwen3.5 models.

---

## Acknowledgements

Based on [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE})
by the Qwen Team. If you use this model in research, please cite the original:

```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
```
