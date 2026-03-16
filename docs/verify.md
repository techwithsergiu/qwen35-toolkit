---
title: qwen35-verify
---

# qwen35-verify / qwen35-verify-qwen35

Both commands accept a **local path or a HF Hub repo id**.  
Mode (BNB vs f16) and device strategy are **both auto-detected** — no flags needed.

**Source detection:** paths starting with `./`, `../`, `/`, `~` or pointing to an
existing directory are treated as local. Anything else containing a `/` is treated
as a HF Hub repo id.

## Two commands — when to use which

| Command | Module | Use for |
|---|---|---|
| `qwen35-verify` | `qwen35_toolkit.verify` | Universal: any `AutoModelForCausalLM`-compatible model (BNB or f16, auto-detected) |
| `qwen35-verify-qwen35` | `qwen35_toolkit.verify_qwen35` | Qwen3.5-specific: thinking ON/OFF + image inference test, four arch×mode combinations |

`verify.py` is the generic base; `verify_qwen35.py` imports it and overrides only
what is Qwen3.5-specific.

## Module / function tree

```
qwen35_toolkit/verify.py     ← generic, uses AutoModelForCausalLM
  │  is_hf_repo()                detects local path vs HF repo id
  │  resolve_source()            returns (is_remote, label) for display
  │  check_config()              fetches + validates config.json
  │  model_size_gb()             actual on-disk size (all weights)
  │  detect_mode()               reads config.json → "bnb" | "f16"
  │  detect_arch()               reads config.json → "vlm" | "text"
  │  _pick_device()              estimates total size vs VRAM → strategy
  │  _move_to_cuda()             OOM-safe model.cuda() with CPU fallback
  │  restore_visual_to_fp()      restores visual layers to exact bf16
  │  drop_visual_tower()         removes visual submodule, frees RAM/VRAM
  │  count_quantized_layers()    counts Linear4bit vs total nn.Linear
  │  run_inference_tasks()       runs prompt list, reports tok/s
  │  verify()                    full pipeline: steps 1–5
  │
  └── qwen35_toolkit/verify_qwen35.py  ← Qwen3.5-specific
        load_qwen35()              detects arch+mode, picks loader:
                                     vlm+bnb  → Qwen3_5ForConditionalGeneration
                                     vlm+f16  → Qwen3_5ForConditionalGeneration
                                     text+bnb → AutoModelForCausalLM
                                     text+f16 → AutoModelForCausalLM
        verify_qwen35()            adds steps 4, 5a, 5b (see below)
```

## Verification steps

### `verify.py` (generic)

```text
── 1. Config ─────── BNB: validates quantization_config, checks for expected keys
                     f16: reports architectures / model_type / torch_dtype
── 2. Load ───────── loads model + tokenizer; picks strategy, moves to device
                     _move_to_cuda() used as runtime OOM guard in all paths
── 3. Precision ──── BNB: counts Linear4bit layers, checks visual tower dtype
                     f16: reports layer count and predominant weight dtype
── 4. Image ──────── skipped (use qwen35-verify-qwen35 for VLM)
── 5. Inference ──── model generates a coherent response for each test task
```

### `verify_qwen35.py` (Qwen3.5-specific)

```text
── 1. Config ─────── BNB: validates quantization_config
                     f16: reports architectures / model_type / torch_dtype
── 2. Load ───────── arch + mode auto-detected; loader chosen accordingly:
                       vlm+bnb  → Qwen3_5ForConditionalGeneration + restore_visual_to_fp
                       vlm+f16  → Qwen3_5ForConditionalGeneration + torch_dtype=auto
                       text+bnb → AutoModelForCausalLM
                       text+f16 → AutoModelForCausalLM + torch_dtype=auto
── 3. Precision ──── BNB: counts Linear4bit layers, checks visual tower dtype
                     f16: reports linear layer count + weight dtype
── 4. Image ──────── VLM only: visual pipeline e2e test (see below)
                     device: GPU (cuda_direct) or CPU (cuda_drop / cpu)
                     text: skipped
── 5a. Inference ─── THINKING OFF — math task, checks response correctness
── 5b. Inference ─── THINKING ON  — same task with <think> block enabled
```

## Automatic device strategy

`_pick_device()` reads actual on-disk file sizes (no model load) and picks a
strategy. Works identically for BNB and f16.

Size accounting uses the full file size — this captures backbone, visual tower,
`lm_head`, `embed_tokens`, and all other non-quantized weights. Summing only
the quantized backbone would undercount (e.g. `lm_head.weight` alone is ~1 GB
in the 9B model).

| Strategy      | Threshold                     | Description                                 |
| ------------- | ----------------------------- | ------------------------------------------- |
| `cuda_direct` | total ≤ 95% VRAM              | Full model fits → move everything to GPU    |
| `cuda_drop`   | (total − visual) ≤ 95% VRAM  | Image test on CPU → drop visual → GPU       |
| `cpu`         | (total − visual) > 95% VRAM  | Stay on CPU throughout                      |

```text
cuda_direct:
    Load to CPU (always CPU-first for BNB)
    _move_to_cuda()  — full model to GPU, visual included
    Image test on GPU  ✅ fast
    drop_visual_tower() → free VRAM
    Text inference on GPU

cuda_drop:
    Load to CPU
    Image test on CPU  (visual intact, all on same device)
    drop_visual_tower() → free RAM
    _move_to_cuda()  — backbone + lm_head + embeddings fit in VRAM
    Text inference on GPU

cpu:
    Load to CPU
    Image test on CPU
    drop_visual_tower() → free RAM
    Text inference on CPU  ← BNB does not require CUDA
```

### `_move_to_cuda()` — runtime OOM guard

Wraps every `model.cuda()` call. If `OutOfMemoryError` is raised (due to
fragmented VRAM, driver overhead, or an imprecise file-size estimate), it clears
the cache and falls back to CPU. Inference always runs — no step is ever skipped
due to VRAM constraints.

## Visual tower handling (BNB models)

The converter writes `llm_int8_skip_modules: ["lm_head", "model.visual"]` with
exact model-tree paths into `config.json`. Transformers matches by prefix, so
`"model.visual"` covers the entire visual tower and prevents re-quantization on
load. This is the primary mechanism.

`restore_visual_to_fp()` acts as a **fallback** for models converted before
this fix or loaded from older Hub checkpoints. It reloads exact original bf16
weights directly from safetensors on disk (no dequantize rounding error). Step 3
reports whether restore was needed.

> **Note:** This assumes the remaining text backbone is architecturally compatible
> with the standalone text config of the same model size. Mixing visual weights
> from one size with text weights from another is unsupported and will produce
> garbage output without an obvious error.

## `lm_head.weight` precision note

`lm_head.weight` is intentionally left at full precision (listed in
`llm_int8_skip_modules` for stability — quantizing the output projection degrades
generation noticeably). It maps hidden states → vocabulary logits. In the 9B model
`lm_head.weight` alone is ~1 GB of fp16, which is why full-file-size accounting in
`_pick_device()` matters.

## Image inference test (step 4)

A 448×224 PNG with the left half red and right half blue is embedded as base64
in the script (no file I/O, no network). The model is asked:

> `"What color is the left half of this image? Answer with one word."`

Expected answer: `"red"` (case-insensitive).

This verifies the full visual pipeline end-to-end:
`image tokens → visual encoder → merger → language model → response`

## Extending to a new model family

Create `verify_<family>.py`, import helpers from `verify_model`, override only the
loader and model-specific generation kwargs.

## CLI reference

```
qwen35-verify          --model <repo_or_path> [--hf-token hf_...]
qwen35-verify-qwen35   --model <repo_or_path> [--hf-token hf_...]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF Hub repo id or local path |
| `--hf-token` | env `HF_TOKEN` | HF access token; falls back to `HF_TOKEN` env var |

## Examples

```bash
# Generic verify (text-only models)
qwen35-verify --model ./Qwen3.5-text-0.8B-bnb-4bit
# repeat for 2B, 4B, 9B

# From HF Hub (cached — re-runs skip download)
qwen35-verify --model <your-hf-username>/Qwen3.5-text-0.8B-bnb-4bit

# Qwen3.5-specific: thinking ON/OFF + image inference (VLM models)
qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit
# repeat for 2B, 4B, 9B

# 9B on 7.7 GB card — auto-detects cuda_drop:
# image test on CPU → drop visual tower → backbone moves to GPU
qwen35-verify-qwen35 --model ./Qwen3.5-9B-bnb-4bit

# From HF Hub
qwen35-verify-qwen35 --model <your-hf-username>/Qwen3.5-0.8B-bnb-4bit
```
