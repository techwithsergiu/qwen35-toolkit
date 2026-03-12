---
tags:
- techwithsergiu
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-{SIZE}/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- techwithsergiu/Qwen3.5-text-{SIZE}
---

# Qwen3.5-text-{SIZE}-bnb-4bit

<img width="400px" src="https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png">

BNB NF4 4-bit quantization of [techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}) —
a text-only derivative of [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}).

**No visual tower** — text input only. This is the recommended base for Unsloth LoRA
text fine-tuning: smaller VRAM footprint, no visual-dependency complexity, cleaner
adapter targeting.

Inference has been verified. LoRA fine-tuning docs are pending — see Fine-tuning section below.

## What was changed from the original Qwen3.5-{SIZE}

- Visual tower removed (same as `Qwen3.5-text-{SIZE}`)
- Text backbone quantized to BNB NF4 double-quant (`bnb_4bit_quant_type=nf4`, `bnb_4bit_compute_dtype=bfloat16`)
- `lm_head.weight` kept at **bf16** for output quality / stability

## Model family

```mermaid
graph LR
    A["Qwen/Qwen3.5-{SIZE}<br/>f16 · VLM"] --> B["Qwen3.5-{SIZE}-bnb-4bit<br/>BNB NF4 · VLM"]
    A --> C["Qwen3.5-text-{SIZE}<br/>bf16 · text-only"]
    C --> D["Qwen3.5-text-{SIZE}-bnb-4bit<br/>BNB NF4 · text-only"]
    C --> E["Qwen3.5-text-{SIZE}-GGUF<br/>Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q4_K_S"]
    style D fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style E fill:#fce7f3,stroke:#db2777,color:#831843
```

| Model | Type | Base model |
|---|---|---|
| [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}) | f16 · VLM · source | — |
| [techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit) | BNB NF4 · VLM | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}) | bf16 · text-only | Qwen/Qwen3.5-{SIZE} |
| **[techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit)** | BNB NF4 · text-only | Qwen3.5-text-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-GGUF](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-GGUF) | GGUF quants | Qwen3.5-text-{SIZE} |

The visual tower scales with model size (~0.19 GB for 0.8B, ~0.62 GB for 2B/4B, ~0.85 GB for 9B).
BNB text-only models are roughly 34% of the original f16 size (4B example: 9.32 GB → 3.12 GB).

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "What is the capital of Romania?"}]

# Thinking OFF — direct answer
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print(response)

# Thinking ON — chain-of-thought before the answer
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print(response)
```

## Fine-tuning

> **TBD** — LoRA training with this model has not been documented yet.
> The model has been verified for inference (text generation, thinking ON/OFF).
> The expectation is that standard Unsloth LoRA training applies — this is a
> text-only BNB 4-bit model architecturally identical to models Unsloth supports —
> but this has not been tested yet and there is no official Qwen3.5 text-only
> training guide to reference.
>
> For VLM (image + text) fine-tuning of the full model, see:
> [unsloth.ai/docs/models/qwen3.5/fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

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
