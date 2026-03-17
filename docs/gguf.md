---
title: GGUF conversion
---

# GGUF conversion

> [!NOTE]
> Validated path uses text-only Qwen3.5 checkpoints.
> Do not use VLM checkpoints with visual tower for GGUF conversion.

## Prerequisites

- Text-only model from Path B (for example `Qwen3.5-text-0.8B`).
- `llama.cpp` cloned locally.
- Python environment with conversion script dependencies.

## Steps

### Step 1 — Build llama.cpp tools
Build `llama-quantize` binary for your environment.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

Success criteria: `./llama.cpp/build/bin/llama-quantize --help` runs.

### Step 2 — Convert HF model to GGUF f16
Convert text-only model directory into f16 GGUF file.

```bash
python llama.cpp/convert_hf_to_gguf.py <model_path> \
  --outtype f16 \
  --outfile <output_dir>/<name>-F16.gguf
```

Success criteria: `<name>-F16.gguf` exists in output directory.

### Step 3 — Quantize GGUF
Create quantized variants from f16 GGUF.

```bash
./llama.cpp/build/bin/llama-quantize <input.gguf> <output.gguf> <TYPE>
```

Success criteria: quantized `.gguf` file is produced for selected type.

## Quantization options

| Type | Size vs f16 | Notes |
|------|-------------|-------|
| `Q8_0` | ~53% | Near-lossless quality |
| `Q6_K` | ~41% | High quality balance |
| `Q5_K_M` | ~37% | Smaller with strong quality |
| `Q4_K_M` | ~31% | Recommended default |
| `Q4_K_S` | ~30% | Slightly smaller, lower quality |

## Example commands (0.8B)

```bash
python llama.cpp/convert_hf_to_gguf.py ./Qwen3.5-text-0.8B \
  --outtype f16 \
  --outfile ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf
```

```bash
./llama.cpp/build/bin/llama-quantize \
  ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
  ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q4_K_M.gguf \
  Q4_K_M
```

## Expected result

- You have one f16 GGUF file.
- You have one or more quantized GGUF files (commonly `Q4_K_M`).
- Files are ready for llama.cpp-based inference or Hub upload.

## Common failures

- Conversion fails on VLM checkpoint -> use text-only checkpoint first (`qwen35-strip --mode f16`).
- `llama-quantize` not found -> build llama.cpp binaries first.
- Unexpected quality/size tradeoff -> regenerate with different quant type.

## Related

- [Quickstart](quickstart.md)
- [Conversion pipeline](conversion-pipeline.md)
- [Upload](upload.md)
