---
title: GGUF conversion
---


# GGUF conversion

Converts a text-only bf16 model to GGUF format using **llama.cpp** (external
tool, not part of this toolkit). Requires Path B of the pipeline — the source
must be a `Qwen3_5ForCausalLM` text-only model, not the VLM variant.

> **Why text-only:** llama.cpp does not support the Qwen3.5 visual tower.
> Using the VLM variant will fail or produce incorrect output.

## Setup

```bash
git clone https://github.com/ggml-org/llama.cpp

# Build with CUDA support
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

No build needed for `convert_hf_to_gguf.py` — it is a Python-only script.

## Step 1 — HF → GGUF f16

```bash
python llama.cpp/convert_hf_to_gguf.py <model_path> \
    --outtype f16 \
    --outfile <output_dir>/<name>-F16.gguf
```

Example (0.8B):

```bash
python llama.cpp/convert_hf_to_gguf.py ./Qwen3.5-text-0.8B \
    --outtype f16 \
    --outfile ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf
```

## Step 2 — GGUF f16 → quantized

```bash
./llama.cpp/build/bin/llama-quantize <input.gguf> <output.gguf> <TYPE>
```

### Quant types

| Type   | Size vs f16 | Notes |
|--------|-------------|-------|
| Q8_0   | ~53%        | Near-lossless — for high-quality inference |
| Q6_K   | ~41%        | Excellent quality, good balance with f16 |
| Q5_K_M | ~37%        | Very good quality, smaller than Q6 |
| Q4_K_M | ~31%        | ✅ Recommended — best size/quality balance |
| Q4_K_S | ~30%        | Optional — slightly smaller, slightly lower quality |

### Examples (0.8B)

```bash
./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q8_0.gguf \
    Q8_0

./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q6_K.gguf \
    Q6_K

./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q5_K_M.gguf \
    Q5_K_M

./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q4_K_M.gguf \
    Q4_K_M

./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q4_K_S.gguf \
    Q4_K_S
```

## Running GGUF models with llama.cpp

The published GGUF models are quantized to Q4_K_M and available at:
https://huggingface.co/collections/techwithsergiu/qwen35-text-gguf

```bash
./build/bin/llama-cli \
  --hf-repo <your-hf-username>/Qwen3.5-text-0.8B-GGUF \
  --hf-file Qwen3.5-text-0.8B-Q4_K_M.gguf \
  -p "Write a Python function that reverses a string." \
  -n 512
```
