---
title: qwen35-strip
---

# qwen35-strip

## Purpose

Remove visual tower weights from Qwen3.5 VLM checkpoints at file level.
The strip operation edits safetensors/index/config artifacts without loading the full model into memory.

## When to use

- You need text-only checkpoints for training or export workflows.
- You want to remove vision branches before verification/inference.
- You need either BNB text-only output or f16/bf16 text-only output.

## Syntax

```text
qwen35-strip --model <repo_or_path> --output <dir> [options]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF Hub repo id or local path |
| `--output` | — | Destination directory |
| `--mode` | `bnb` | `bnb` or `f16` |
| `--hf-token` | env `HF_TOKEN` | HF token; falls back to env var |

## Modes

| `--mode`        | Input          | Output              | Use for              |
| --------------- | -------------- | ------------------- | -------------------- |
| `bnb` (default) | BNB 4-bit VLM  | text-only BNB 4-bit | training target      |
| `f16`           | f16 / bf16 VLM | text-only bf16      | GGUF / merge base    |

## Strip flow

```text
1. Strip tensors:
   - Remove visual-prefix tensors from safetensors shards.
   - Rebuild shard index for remaining text tensors.
2. Copy side-car files:
   - Keep text-side files.
   - Exclude vision-specific side-cars.
3. Rebuild config (`text_config` as root):
   - Do not inherit VLM vision dimensions.
   - Set architectures to Qwen3_5ForCausalLM.
   - `bnb` mode: keep `quantization_config`, remove `model.visual` from skip modules.
   - `f16` mode: remove `quantization_config`.
4. Patch templates:
   - Remove image/video branches in `tokenizer_config.json`.
   - Remove image/video branches in `chat_template.jinja`.
5. Verify:
   - Run structural checks.
   - Run inference checks.
```

Why it matters: output is a clean text-only checkpoint (config + weights + template) that behaves correctly in downstream verify/train/export flows.

Removed prefixes:

```text
model.visual, model.vision_tower, model.vision_model,
model.multi_modal_projector, model.modality_projection,
model.merger, model.router, model.mm_projector, model.vision_proj
```

Typical size removed:
- 0.8B: ~0.19 GB
- 2B/4B: ~0.62 GB
- 9B: ~0.85 GB

## Automatic device selection (verify step)

`_pick_device()` reads actual on-disk file sizes of the **stripped** output and
picks a strategy.

Decision rule:
- same threshold logic as verifier (`<= 95% VRAM`),
- but on stripped checkpoints visual branch is already absent,
- so `cuda_direct` and `cuda_drop` converge to equivalent behavior.

| Strategy | Threshold | What happens |
| --- | --- | --- |
| `cuda_direct` | total ≤ 95% VRAM | CPU load → move to GPU |
| `cuda_drop` | (total − visual) ≤ 95% VRAM | CPU load → move to GPU (same as cuda_direct post-strip) |
| `cpu` | (total − visual) > 95% VRAM | stay on CPU throughout |

`_move_to_cuda()` catches runtime `OutOfMemoryError` and falls back to CPU —
inference always runs regardless of VRAM constraints.

## HF Hub source

When `--model` is a HF repo id, `snapshot_download` uses the default HF cache
(`~/.cache/huggingface/hub`). Re-runs are instant.

## Examples

```bash
# Minimal example (BNB)
qwen35-strip --model ./Qwen3.5-0.8B-bnb-4bit --output ./Qwen3.5-text-0.8B-bnb-4bit
```

```bash
# f16/bf16 source from HF Hub
qwen35-strip --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-text-0.8B --mode f16
```

## Output examples

### Example output

```text
strip_visual_qwen35.py  |  mode: bnb
./test/Qwen3.5-0.8B-bnb-4bit  [local]

📦 Model weight size on disk: 0.90 GB
── 1. Strip visual weights ──
📋 model.safetensors  (dropped 153 visual keys)
🗑️  Removed (estimate): ~0.19 GB

── 3. Rebuild config.json ──
✅ config.json rebuilt  (architectures: ['Qwen3_5ForCausalLM']  model_type: qwen3_5_text)

── Verification ──
✅ architectures: ['Qwen3_5ForCausalLM']
✅ vision_config absent
✅ quantization_config present
✅ No visual keys in safetensors shards
✅ Quantized layers: 186 / 187 linear
✅ Inference OK: '2+2=' → '4, 2+2=4'

✅ Done  →  test/Qwen3.5-text-0.8B-bnb-4bit
```

Interpretation: strip removed visual tensors, rebuilt text config, and passed structural + inference verification.

Stable fields: strip phases, visual-key removal signal, verification checkpoints.
Variable fields: file sizes, dropped-key counts, load speed, VRAM/runtime values.

## Edge cases / limitations

- The command targets Qwen3.5 VLM -> text-only transformation.
- `f16` mode outputs text-only bf16 format for downstream GGUF/merge flows.
- Private HF sources require `--hf-token` or `HF_TOKEN`.

## Related

- [Quickstart](quickstart.md)
- [Verify](verify.md)
- [GGUF](gguf.md)
