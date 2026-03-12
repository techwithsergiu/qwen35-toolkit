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
    A["Qwen/Qwen3.5-{SIZE}<br/>f16 · VLM"] --> B["Qwen3.5-{SIZE}-bnb-4bit<br/>BNB NF4 · VLM"]
    A --> C["Qwen3.5-text-{SIZE}<br/>bf16 · text-only"]
    C --> D["Qwen3.5-text-{SIZE}-bnb-4bit<br/>BNB NF4 · text-only"]
    C --> E["Qwen3.5-text-{SIZE}-GGUF<br/>Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q4_K_S"]
    style B fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style E fill:#fce7f3,stroke:#db2777,color:#831843
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

For VLM (image + text) fine-tuning with Unsloth, refer to the official guide:
[unsloth.ai/docs/models/qwen3.5/fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

## Pipeline diagram

```mermaid
flowchart TD
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · source"]

    subgraph BRANCH_A ["Branch A — VLM"]
        BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
    end

    subgraph BRANCH_B ["Branch B — Text-only BNB"]
        TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
        TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only"]
        TEXTF16 -->|"qwen35-strip --mode bnb"| TEXTBNB
    end

    subgraph BRANCH_C ["Branch C — GGUF"]
        GGUUF16["Qwen3.5-text-{size}.gguf<br/>GGUF f16"]
        Q4["Q4_K_M ✅ main"]
        Q5KM["Q5_K_M very good quality"]
        Q6K["Q6_K excellent quality"]
        Q8["Q8_0 near-lossless"]
        GGUUF16 -->|"llama-quantize"| Q4
        GGUUF16 -->|"llama-quantize"| Q5KM
        GGUUF16 -->|"llama-quantize"| Q6K
        GGUUF16 -->|"llama-quantize"| Q8
    end

    SRC -->|"qwen35-convert"| BNBVLM
    SRC -->|"qwen35-strip --mode f16"| TEXTF16
    TEXTF16 -->|"convert_hf_to_gguf.py"| GGUUF16

    BNBVLM -->|"qwen35-verify-qwen35"| V1{{"✅ verified"}}
    TEXTF16 -->|"qwen35-verify"| V2{{"✅ verified"}}
    TEXTBNB -->|"qwen35-verify"| V3{{"✅ verified"}}

    V1 -->|"qwen35-upload"| HUB[("HuggingFace Hub")]
    V2 -->|"qwen35-upload"| HUB
    V3 -->|"qwen35-upload"| HUB
    Q4  -->|"qwen35-upload"| HUB
    Q5KM -->|"qwen35-upload"| HUB
    Q6K  -->|"qwen35-upload"| HUB
    Q8  -->|"qwen35-upload"| HUB

    style TEXTBNB fill:#dcfce7,stroke:#16a34a
    style Q4  fill:#fce7f3,stroke:#db2777
    style Q5KM fill:#fce7f3,stroke:#db2777
    style HUB     fill:#f3e8ff,stroke:#9333ea
```

## Conversion

Converted using [qwen35-toolkit](https://github.com/techwithsergiu/qwen35-toolkit) —
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
