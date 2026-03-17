---
title: qwen35-strip
---


# qwen35-strip

Removes the visual tower from a Qwen3.5 model at the **file level** — no model
is loaded into memory during stripping. Operates directly on safetensors shards.

## Modes

| `--mode`        | Input          | Output              | Use for              |
| --------------- | -------------- | ------------------- | -------------------- |
| `bnb` (default) | BNB 4-bit VLM  | text-only BNB 4-bit | training target      |
| `f16`           | f16 / bf16 VLM | text-only bf16      | GGUF / merge base    |

## Pipeline (5 steps)

```text
1. Strip visual weights   — drop prefixes from safetensors shards (header scan
                            + key filter; no tensor data loaded unnecessarily).
                            Also rebuilds model.safetensors.index.json for
                            multi-shard models (recomputes total_size).
                            Removed prefixes:
                              model.visual, model.vision_tower, model.vision_model,
                              model.multi_modal_projector, model.modality_projection,
                              model.merger, model.router, model.mm_projector,
                              model.vision_proj
                            Typical sizes removed:
                              0.8B → ~0.19 GB  |  2B / 4B → ~0.62 GB  |  9B → ~0.85 GB

2. Copy side-car files    — tokenizer.json, tokenizer_config.json, vocab.json,
                            merges.txt, special_tokens_map.json, generation_config.json.
                            Vision-specific sidecars (preprocessor_config.json,
                            processor_config.json, video_preprocessor_config.json)
                            are intentionally excluded from the text-only output.

3. Rebuild config.json    — starts from config["text_config"] as root to avoid
                            inheriting vision dimensions (hidden_size, num_layers,
                            etc.) from the VLM root config.
                            Sets architectures: ["Qwen3_5ForCausalLM"]
                            Carries over global token ids (bos, eos, pad) that
                            live at the VLM root, not text_config.
                            BNB mode: carries quantization_config, removes
                              model.visual from llm_int8_skip_modules (visual
                              is gone — no need to skip it on load anymore).
                            f16 mode: drops quantization_config entirely.
                            Safety pass strips residual VISION_CONFIG_KEYS
                            (vision_config, image_token_id, video_token_id, …).

4. Patch chat template    — strips image/video branches from the Jinja2 template
                            in both tokenizer_config.json and chat_template.jinja
                            (if present). Patches: image_count/video_count set
                            blocks, {% if 'image' in item %} and
                            {% elif 'video' in item %} branches removed,
                            {% elif 'text' ... %} rewritten to {% if 'text' ... %}.

5. Verify                 — structural check (config + safetensors key scan)
                            + inference check with automatic device selection.
```

## Automatic device selection (verify step)

`_pick_device()` reads actual on-disk file sizes of the **stripped** output and
picks a strategy. Since the visual tower is already removed, both `cuda_direct`
and `cuda_drop` behave identically — there is no image test and no visual tower
to drop.

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

## CLI reference

```
qwen35-strip --model <repo_or_path> --output <dir> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF Hub repo id or local path |
| `--output` | — | Destination directory |
| `--mode` | `bnb` | `bnb` or `f16` |
| `--hf-token` | env `HF_TOKEN` | HF access token; falls back to `HF_TOKEN` env var |

## Examples

```bash
# BNB 4-bit VLM → text-only BNB 4-bit
qwen35-strip --model ./Qwen3.5-0.8B-bnb-4bit --output ./Qwen3.5-text-0.8B-bnb-4bit
# repeat for 2B, 4B, 9B

# f16 VLM → text-only bf16 (from HF Hub)
qwen35-strip --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-text-0.8B --mode f16
# repeat for 2B, 4B, 9B; add --hf-token hf_... for private repos
```
