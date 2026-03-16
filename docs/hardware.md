---
title: Hardware
---

# Hardware

All 16 published models were produced on a single consumer GPU — an RTX 3070
Laptop with 7.7 GB VRAM. This page documents the actual numbers from that process
and what they mean for running the toolkit on your hardware.

## VRAM and RAM breakdown

| Operation | VRAM | RAM |
|---|---|---|
| BNB quantization (4B, default) | ~5 GB | ~18 GB |
| BNB quantization (any size, `--low-vram`) | ~200–400 MB peak | ~18 GB |
| f16 model load for strip / verify | none | ~18 GB |
| GGUF conversion (f16 → GGUF) | none | ~18 GB |
| GGUF Q4_K_M inference (4B) | ~5 GB | — |

## What the numbers mean

**RAM is the real constraint, not VRAM.** The full f16 model must fit in CPU RAM
before any GPU work begins — this is true for quantization, strip, and GGUF
conversion. 18 GB is the practical minimum for the 4B model; the 9B model
requires proportionally more.

**VRAM is only needed for quantization and inference.** Strip and GGUF conversion
are entirely CPU/RAM operations. If your GPU doesn't have enough VRAM for the
default quantization path, `--low-vram` brings the peak down to a single layer
(~200–400 MB) at the cost of longer runtime.

**`--low-vram` is how the 9B model was quantized on a 7.7 GB card.** The flag
quantizes one layer at a time — each layer is moved to GPU, compressed to NF4,
then immediately returned to CPU. Only one layer occupies VRAM at any point.

## Minimum requirements by operation

| Operation | Minimum VRAM | Minimum RAM |
|---|---|---|
| Quantize 0.8B / 2B / 4B | ~5 GB | ~18 GB |
| Quantize 9B | ~200–400 MB peak (with `--low-vram`) | ~36 GB |
| Strip / GGUF convert (any size) | none | model size × 2 |
| Verify 4B text-only | ~3 GB | ~8 GB |

## Running on different hardware

**Less than 8 GB VRAM:** use `--low-vram` for quantization — tested on the RTX 3070
Laptop (7.7 GB effective). Strip and GGUF conversion require no VRAM at all.

**No GPU:** BNB quantization requires CUDA. Strip, verify (CPU fallback), and GGUF
conversion run on CPU only. `_move_to_cuda()` in verify always falls back to CPU if
VRAM is insufficient — no step is ever skipped.

**More than 8 GB VRAM:** the default path (without `--low-vram`) will work for all
model sizes up to 9B.

See [Published Models](models.md) for the full list of what was produced on this hardware.
