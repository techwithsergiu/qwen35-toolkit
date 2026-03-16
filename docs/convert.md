---
title: qwen35-convert
---


# qwen35-convert

⚠️ Qwen3.5-specific — uses `Qwen3_5ForConditionalGeneration`.  
For other model families use a generic `AutoModelForCausalLM` variant.

## Conversion strategy

Avoids the `Params4bit/_is_hf_initialized` version mismatch bug:

```text
1.  Load full model into CPU RAM — bf16, no GPU involved, no accelerate dispatch.
1b. Dequantize skip-module layers — if the source model was already quantized
    (e.g. some unsloth Hub models load with BNB via their config.json), any
    Linear4bit layers in skip-module subtrees (visual, embed_tokens) are
    restored to plain bf16 nn.Linear BEFORE the replacement pass runs.
    Without this step, those layers stay as Linear4bit and cause visual-tower
    quantization bugs. Reports how many layers were restored.
2.  Replace every eligible nn.Linear → Linear4bit in-place (still on CPU).
3.  Quantize — BNB nf4 compression is triggered by the first .cuda() call per
    parameter. Two modes:
      default    : model.cuda() — full model to GPU at once.
                   Fast; requires ~5 GB VRAM for a 4B model.
      --low-vram : quantize_layer_by_layer() — moves one layer at a time,
                   immediately returns quantized weights to CPU RAM.
                   Peak VRAM ≈ one layer (~200–400 MB). Model saved from CPU.
4.  Save quantized model + tokenizer.
5.  Rebuild config.json from source (architectures, vision_config,
    quantization_config). Patch tokenizer_config.json (chat_template).
6.  Copy optional side-car files (merges.txt, vocab.json,
    preprocessor_config.json, …) — missing files silently skipped.
    Note: preprocessor_config.json is included here because the output
    is still a VLM model (visual tower intact). It is intentionally
    excluded by qwen35-strip when producing a text-only model.
```

> **Important:** do not skip step 3. `Params4bit` on CPU only wraps the
> original bf16 data — no compression has occurred yet. Calling
> `save_pretrained()` before `.cuda()` writes full bf16 weights (~18 GB instead
> of ~4 GB).

## Hardware requirements

| Resource | Default mode | `--low-vram` |
|----------|-------------|--------------|
| RAM      | ~18 GB      | ~18 GB       |
| VRAM     | ~5 GB       | ~200–400 MB peak |

## Layers skipped from quantization (left at full precision)

During the in-process walk `replace_with_bnb_linear4bit` skips any layer whose
full dotted path contains one of these short-name fragments:

```
embed_tokens, embedding, lm_head, visual, vision_tower,
merger, multi_modal_projector, modality_projection, router
```

Extra entries (`embedding`, `vision_tower`, etc.) are harmless no-ops for
Qwen3.5 but kept for compatibility if the script is adapted to other models.

The `quantization_config` written to `config.json` uses **exact model-tree
paths** so transformers respects them on reload:

```json
"llm_int8_skip_modules": [
    "lm_head",
    "model.visual",
    "model.language_model.embed_tokens"
]
```

Short names (`"visual"` etc.) are used only during the in-process conversion
walk — they are not written to config. Transformers matches by prefix, so
`"model.visual"` covers the entire visual tower.

## `--verbose` output

```text
⏳ Replacing linear layers with BNB Linear4bit …
   └── skipped      model.visual
   └── skipped      model.language_model.embed_tokens
   └── skipped      lm_head
   186 layers replaced
```

Useful to confirm visual layers are correctly skipped (not quantized).

## CLI reference

```
qwen35-convert --model <repo_or_path> --output <dir> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/Qwen3.5-0.8B` | HF Hub repo id or local path of the source f16 model |
| `--output` | `./Qwen3.5-0.8B-bnb-4bit` | Directory where the quantized model will be saved |
| `--low-vram` | off | Quantize one layer at a time; peak VRAM ~200–400 MB |
| `--verbose` | off | Print every layer replacement and skip decision |
| `--hf-token` | env `HF_TOKEN` | HF access token for private repos; falls back to `HF_TOKEN` env var |

## Examples

```bash
# Standard quantization
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit
# repeat for 2B, 4B; use --low-vram for 9B

# Verbose — confirm skip decisions
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit --verbose
```
