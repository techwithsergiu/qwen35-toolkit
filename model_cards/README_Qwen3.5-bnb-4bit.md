---
tags:
- techwithsergiu
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-{SIZE}/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.5-{SIZE}
---

# Qwen3.5-{SIZE}-bnb-4bit

<img width="400px" src="https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png">

BNB NF4 4-bit quantization of [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}).

Retains the full visual tower — this is a **VLM-capable** model (image + text input).
Primary use-case: Unsloth LoRA fine-tuning when you need image understanding in the
fine-tuned result.

> If you only need text fine-tuning, use
> [techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit)
> instead — same backbone, visual tower removed, lighter VRAM footprint.

## What was changed

- Quantized with `bitsandbytes` NF4 double-quant (`bnb_4bit_quant_type=nf4`, `bnb_4bit_compute_dtype=bfloat16`)
- Visual tower layers kept at **bf16** (`llm_int8_skip_modules`) — required for correct image inference
- `lm_head.weight` kept at **bf16** for output quality

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
| **[techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit)** | BNB NF4 · VLM | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}) | bf16 · text-only | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit) | BNB NF4 · text-only | Qwen3.5-text-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-GGUF](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-GGUF) | GGUF quants | Qwen3.5-text-{SIZE} |

The visual tower is a bf16 overhead that scales with model size (~0.19 GB for 0.8B, ~0.62 GB for 2B/4B, ~0.85 GB for 9B).
BNB-quantized models are roughly 40% of the original f16 size (exact ratio varies by size).

## Fine-tuning

**Text-only LoRA fine-tuning** — use the text-only BNB variant as training base:
[techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit)

Training pipeline (QLoRA · Unsloth · TRL):
[github.com/techwithsergiu/qwen-qlora-train](https://techwithsergiu.github.io/qwen-qlora-train)

**VLM (image + text) fine-tuning** — refer to the official Unsloth guide:
[unsloth.ai/docs/models/qwen3.5/fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

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
