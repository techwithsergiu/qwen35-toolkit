---
title: qwen35-convert
---

# qwen35-convert

## Purpose

Convert a Qwen3.5 f16/bf16 VLM checkpoint into BNB NF4 4-bit format.
The command is Qwen3.5-specific and uses `Qwen3_5ForConditionalGeneration`.

## When to use

- You need a BNB 4-bit VLM checkpoint as input to strip/verify workflows.
- You want lower VRAM quantization with `--low-vram`.
- You need reproducible skip-module behavior for visual + embedding layers.

## Syntax

```text
qwen35-convert --model <repo_or_path> --output <dir> [options]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/Qwen3.5-0.8B` | HF Hub repo id or local path of source model |
| `--output` | `./Qwen3.5-0.8B-bnb-4bit` | Output directory for quantized model |
| `--low-vram` | off | Quantize one layer at a time; peak VRAM ~200–400 MB |
| `--verbose` | off | Print per-layer replace/skip decisions |
| `--hf-token` | env `HF_TOKEN` | HF token for private repos; falls back to env var |

## Conversion flow

Avoids the `Params4bit/_is_hf_initialized` version mismatch bug:

```text
1. Load:
   - Load source model to CPU RAM as bf16/f16 (no accelerate dispatch).
2. Restore (if needed):
   - Replace legacy pre-quantized skip layers back to full-precision nn.Linear.
3. Replace:
   - Swap eligible nn.Linear layers with BNB Linear4bit on CPU.
4. Quantize:
   - Trigger real NF4 quantization on first `.cuda()` call.
   - default: full model to GPU at once (~5 GB VRAM for 4B).
   - `--low-vram`: layer-by-layer quantization (~200-400 MB peak VRAM).
5. Save:
   - Save quantized weights and tokenizer.
6. Finalize:
   - Rebuild config/tokenizer config.
   - Copy optional side-car files.
```

Why it matters: conversion is deterministic and keeps visual/lm_head skip behavior stable across reloads.

> [!WARNING]
> Do not skip the `.cuda()` quantization trigger step.
> Saving before CUDA keeps full bf16 payload (~18 GB instead of ~4 GB).

## Hardware requirements

| Resource | Default mode | `--low-vram` |
|----------|-------------|--------------|
| RAM      | ~18 GB      | ~18 GB       |
| VRAM     | ~5 GB       | ~200–400 MB peak |

## Skip modules and config behavior

The conversion has two skip mechanisms: one for in-process replacement, one for reload behavior.

In-process replacement (`replace_with_bnb_linear4bit`):
- Match uses short-name fragments against full dotted layer paths.
- Matching fragments:

```
embed_tokens, embedding, lm_head, visual, vision_tower,
merger, multi_modal_projector, modality_projection, router
```

Notes:
- Extra fragments (`embedding`, `vision_tower`, etc.) are harmless no-ops on Qwen3.5.
- They are kept for compatibility if the converter is adapted to other model families.

Reload behavior (`config.json` / `quantization_config`):
- Uses exact model-tree paths so transformers applies skip rules on load.
- Written value:

```json
"llm_int8_skip_modules": [
    "lm_head",
    "model.visual",
    "model.language_model.embed_tokens"
]
```

Rules:
- Short names (`visual`, etc.) are in-process only and are not written to config.
- Transformers matches by prefix; `model.visual` covers the full visual tower.

## Output examples

### Verbose output

```text
⏳ Replacing linear layers with BNB Linear4bit …
   └── skipped      model.visual
   └── skipped      model.language_model.embed_tokens
   └── skipped      lm_head
   186 layers replaced
```

Useful to confirm visual layers are correctly skipped (not quantized).

### Example output

```text
GPU : NVIDIA GeForce RTX 3070 Laptop GPU  |  VRAM : 7.7 GB free / 7.7 GB total  [before load]
⏳ Loading model into CPU RAM (bf16, no quantization yet) …
Loading weights: 100%|...| 473/473 [...]
✅ Model loaded into RAM.

⏳ Replacing linear layers with BNB Linear4bit …
   186 layers replaced

⏳ Moving to GPU (this triggers BNB nf4 quantization per layer) …
✅ Model quantized and on GPU.

GPU : NVIDIA GeForce RTX 3070 Laptop GPU  |  VRAM : 6.7 GB free / 7.7 GB total  [after quantize]
💾 Saving to './test/Qwen3.5-0.8B-bnb-4bit' …
✅ Done. Saved to: ./test/Qwen3.5-0.8B-bnb-4bit
```

Interpretation: conversion completed end-to-end (CPU load -> BNB replacement -> GPU quantization -> save).

Stable fields: stage order, replaced-layer count line, final done marker.
Variable fields: VRAM values, timings, model path, progress bar rates.

## Examples

```bash
# Minimal example
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit
```

```bash
# Low VRAM and verbose diagnostics
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit --verbose
qwen35-convert --model unsloth/Qwen3.5-9B --output ./Qwen3.5-9B-bnb-4bit --low-vram
```

## Edge cases / limitations

- Qwen3.5-only loader path; other model families require a different converter.
- Source models already quantized by config may need skip-layer restoration (handled in flow step 2).
- Private HF repos require `--hf-token` or `HF_TOKEN`.

## Related

- [Quickstart](quickstart.md)
- [Strip](strip.md)
- [Verify](verify.md)
