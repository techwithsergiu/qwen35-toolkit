---
tags:
- techwithsergiu
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-{SIZE}/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.5-{SIZE}
---

# Qwen3.5-text-{SIZE}

<img width="400px" src="https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png">

Text-only bf16 derivative of [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}).

The visual tower (vision encoder, image merger, video preprocessor) has been removed.
All text-backbone weights are **identical** to the original — no retraining, no weight
changes, no quality loss for text tasks.

Primary use-case: intermediate model for GGUF conversion or CPU-side f16 merge after
LoRA training. For direct fine-tuning use
[techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit).

## What was changed

- Visual tower removed: `visual`, `image_newline`, `patch_embed`, and related keys
  stripped from safetensors shards
- `config.json` updated: `architectures` → `Qwen3_5ForCausalLM`, `vision_config` removed
- `tokenizer_config.json` and `chat_template.jinja`: image/video branches stripped from
  the Jinja2 chat template — prevents tokenizer errors when no image is provided
- Vision-specific sidecar files omitted (`preprocessor_config.json`, `processor_config.json`,
  `video_preprocessor_config.json`)
- All text weights remain at **bf16**

## Model family

```mermaid
graph LR
    A["Qwen/Qwen3.5-{SIZE}<br/>f16 · VLM"] --> B["Qwen3.5-{SIZE}-bnb-4bit<br/>BNB NF4 · VLM"]
    A --> C["Qwen3.5-text-{SIZE}<br/>bf16 · text-only"]
    C --> D["Qwen3.5-text-{SIZE}-bnb-4bit<br/>BNB NF4 · text-only"]
    C --> E["Qwen3.5-text-{SIZE}-GGUF<br/>Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q4_K_S"]
    style C fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style E fill:#fce7f3,stroke:#db2777,color:#831843
```

| Model | Type | Base model |
|---|---|---|
| [Qwen/Qwen3.5-{SIZE}](https://huggingface.co/Qwen/Qwen3.5-{SIZE}) | f16 · VLM · source | — |
| [techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-{SIZE}-bnb-4bit) | BNB NF4 · VLM | Qwen/Qwen3.5-{SIZE} |
| **[techwithsergiu/Qwen3.5-text-{SIZE}](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE})** | bf16 · text-only | Qwen/Qwen3.5-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit) | BNB NF4 · text-only | Qwen3.5-text-{SIZE} |
| [techwithsergiu/Qwen3.5-text-{SIZE}-GGUF](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-GGUF) | GGUF quants | Qwen3.5-text-{SIZE} |

Removing the visual tower saves ~0.19 GB (0.8B), ~0.62 GB (2B / 4B), or ~0.85 GB (9B).
The relative saving is larger for smaller models.

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "techwithsergiu/Qwen3.5-text-{SIZE}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
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

This model is an **intermediate artifact** — not a direct training target.
For fine-tuning, use [techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-text-{SIZE}-bnb-4bit)
which is the BNB-quantized version of this model.

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
