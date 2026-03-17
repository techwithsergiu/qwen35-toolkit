---
title: qwen35-verify
---

# qwen35-verify / qwen35-verify-qwen35

## Purpose

Verify model structure and runtime behavior for Qwen3.5 workflows.
Both commands accept local paths and HF Hub repo ids, with automatic mode/device detection.

## When to use

- After conversion/strip, before upload or training.
- When you need sanity checks for BNB vs f16 checkpoints.
- When you need VLM-specific checks (thinking ON/OFF + image pipeline).

## Syntax

```text
qwen35-verify          --model <repo_or_path> [--hf-token hf_...]
qwen35-verify-qwen35   --model <repo_or_path> [--hf-token hf_...]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF repo id or local path |
| `--hf-token` | env `HF_TOKEN` | HF access token; falls back to env var |

## Command selection

| Command | Module | Use for |
|---|---|---|
| `qwen35-verify` | `qwen35_toolkit.verify` | Generic verification for `AutoModelForCausalLM`-compatible models (BNB/f16 auto-detected) |
| `qwen35-verify-qwen35` | `qwen35_toolkit.verify_qwen35` | Qwen3.5-specific checks: thinking ON/OFF + image inference test |

Source detection behavior:
- Paths starting with `./`, `../`, `/`, `~` or existing directories are treated as local.
- Any other string containing `/` is treated as HF Hub repo id.

## Verification flow

### Generic flow (`verify.py`)

```text
1. Config:
   - BNB: validate `quantization_config` and required keys.
   - f16: report `architectures`, `model_type`, `torch_dtype`.
2. Load:
   - Load model and tokenizer.
   - Pick device strategy.
   - Move model with `_move_to_cuda()` (OOM-safe fallback to CPU).
3. Precision:
   - BNB: count `Linear4bit` layers and check visual precision.
   - f16: report linear count and predominant weight dtype.
4. Image:
   - Skipped in generic verifier (use `qwen35-verify-qwen35` for VLM image test).
5. Inference:
   - Run text tasks and report latency/speed metrics.
```

### Qwen3.5-specific flow (`verify_qwen35.py`)

```text
1. Config:
   - Same config checks as generic flow.
2. Load:
   - Auto-select loader by architecture + mode.
3. Precision:
   - Same precision checks as generic flow.
4. Image:
   - VLM only: run end-to-end image pipeline test.
5. Inference branches:
   - thinking OFF
   - thinking ON
```

### Loader matrix (`verify_qwen35.py`)

```text
1. vlm + bnb:
   - Qwen3_5ForConditionalGeneration
   - `restore_visual_to_fp` fallback enabled
2. vlm + f16:
   - Qwen3_5ForConditionalGeneration
   - `torch_dtype=auto`
3. text + bnb:
   - AutoModelForCausalLM
4. text + f16:
   - AutoModelForCausalLM
   - `torch_dtype=auto`
```

## Automatic device strategy

`_pick_device()` uses on-disk file sizes (no model load) to choose strategy before loading.

Decision rule:
- Compare full model size (`total`) and text-only estimate (`total - visual`) to VRAM budget.
- Threshold is `<= 95% VRAM`.

| Strategy | Threshold | Behavior |
|---|---|---|
| `cuda_direct` | total <= 95% VRAM | full model to GPU |
| `cuda_drop` | (total - visual) <= 95% VRAM | image test on CPU -> drop visual -> text inference on GPU |
| `cpu` | (total - visual) > 95% VRAM | CPU-only execution |

`_move_to_cuda()` is an OOM guard around `model.cuda()`:
- catches `OutOfMemoryError`,
- clears cache,
- falls back to CPU,
- continues inference instead of aborting.

Accounting note:
- Full-size accounting intentionally includes non-quantized components (`lm_head`, embeddings, visual tower).
- This avoids underestimating memory on larger checkpoints.

### Strategy mechanics

`cuda_direct`:
- load on CPU,
- move full model to GPU,
- run image test on GPU (if VLM),
- optionally drop visual tower and keep text inference on GPU.

`cuda_drop`:
- load on CPU,
- run image test on CPU,
- drop visual tower on CPU,
- move remaining text stack to GPU for text inference.

`cpu`:
- keep full flow on CPU,
- run image test (VLM) on CPU,
- drop visual tower,
- run text inference on CPU.

## Implementation notes

### Visual tower handling (BNB)

Primary mechanism:
- `llm_int8_skip_modules` contains exact paths (for example `model.visual`) so transformers does not re-quantize visual layers.

Fallback mechanism:
- `restore_visual_to_fp()` reloads original bf16 visual weights from safetensors for older/broken checkpoints.

### `lm_head.weight` precision

`lm_head.weight` stays full precision for generation stability.
On larger models this tensor is a significant memory component, so full-size accounting matters in device selection.

### Image test

Qwen3.5-specific verifier uses an embedded red/blue image prompt.
Expected left-half color answer: `red` (case-insensitive).

### Module tree

```text
qwen35_toolkit/verify.py (generic core)
  is_hf_repo()               -> source type detection
  resolve_source()           -> source label/normalization
  check_config()             -> config validation/reporting
  model_size_gb()            -> on-disk size accounting
  detect_mode()              -> bnb/f16
  detect_arch()              -> vlm/text
  _pick_device()             -> cuda_direct/cuda_drop/cpu
  _move_to_cuda()            -> OOM-safe cuda() move
  restore_visual_to_fp()     -> visual precision fallback
  drop_visual_tower()        -> free visual branch for text-only path
  count_quantized_layers()   -> quantization ratio diagnostics
  run_inference_tasks()      -> task probes + tok/s
  verify()                   -> generic orchestrator

qwen35_toolkit/verify_qwen35.py (Qwen3.5 specializations)
  load_qwen35()              -> arch+mode loader selection
  verify_qwen35()            -> image + thinking on/off orchestrator
```

## Console output examples

### Example output (`qwen35-verify-qwen35` on VLM checkpoint)

```text
./test/Qwen3.5-0.8B-bnb-4bit  [Qwen3.5 · local · vlm · bnb]
📦 Model weight size on disk: 0.90 GB

── 1. Config ──
✅ config.json OK  |  skip_modules: ['lm_head', 'model.visual', 'model.language_model.embed_tokens']

── 2. Load ──
✅ Model loaded  (Qwen3_5ForConditionalGeneration)  |  strategy: cuda_direct

── 3. Precision ──
✅ 186/237 linear layers quantized (78%)
✅ Visual tower: 50 layer(s) at full precision (bfloat16)

── 4. Inference [IMAGE] ──
✅ [Image] response: 'red'  (expected: 'red', case-insensitive)

── 5a. Inference [THINKING OFF] ──
✅ [Math] ...

── 5b. Inference [THINKING ON] ──
✅ [Math] ...

✅ ALL CHECKS PASSED
```

Interpretation: VLM checkpoint passes config/load/precision/image and both reasoning branches.

Stable fields: stage names, pass markers, image expected value, strategy label.
Variable fields: tokens/time/tok-s values, full model responses, VRAM usage.

### Example output (`qwen35-verify-qwen35` on text-only checkpoint)

```text
./test/Qwen3.5-text-0.8B-bnb-4bit  [Qwen3.5 · local · text · bnb]
📦 Model weight size on disk: 0.71 GB

── 1. Config ──
✅ config.json OK  |  skip_modules: ['lm_head', 'model.language_model.embed_tokens']

── 2. Load ──
✅ Model loaded  (Qwen3_5ForConditionalGeneration)  |  strategy: cuda_direct

── 3. Precision ──
✅ 186/187 linear layers quantized (99%)
ℹ️  No visual tower detected (text-only model)

── 4. Inference [IMAGE] ──
ℹ️  Skipped (text-only model)

✅ ALL CHECKS PASSED
```

Interpretation: text-only checkpoint skips image stage and still passes full verification.

Stable fields: text-only detection, image-skip behavior, pass markers.
Variable fields: file size, quantized ratio, runtime metrics.

## Examples

```bash
# Generic verify (text-only)
qwen35-verify --model ./Qwen3.5-text-0.8B-bnb-4bit
```

```bash
# Generic verify from HF Hub
qwen35-verify --model <your-hf-username>/Qwen3.5-text-0.8B-bnb-4bit
```

```bash
# Qwen3.5-specific verify (VLM)
qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit
```

```bash
# 9B on small VRAM (expects auto fallback strategy)
qwen35-verify-qwen35 --model ./Qwen3.5-9B-bnb-4bit
```

## Edge cases / limitations

> [!WARNING]
> Mixing visual weights from one model size with text weights from another size is unsupported and can produce invalid outputs.

- Generic verifier skips image test; use `qwen35-verify-qwen35` for VLM image validation.
- Private HF repos require `--hf-token` or `HF_TOKEN`.

## Related

- [Quickstart](quickstart.md)
- [Convert](convert.md)
- [Strip](strip.md)
