---
title: Hardware
---

# Hardware

## Purpose

Provide practical RAM/VRAM requirements for the toolkit workflow.
Numbers are based on runs performed on an RTX 3070 Laptop (7.7 GB VRAM).

## When to use

- Before selecting model size and conversion path.
- Before deciding whether to use `--low-vram`.
- When checking whether an operation is CPU-only or GPU-required.

## Resource requirements

| Operation | VRAM | RAM |
|---|---|---|
| BNB quantization (4B, default) | ~5 GB | ~18 GB |
| BNB quantization (`--low-vram`) | ~200-400 MB peak | ~18 GB |
| f16 model load for strip/verify | none | ~18 GB |
| GGUF conversion (f16 -> GGUF) | none | ~18 GB |
| GGUF Q4_K_M inference (4B) | ~5 GB | — |

## Minimum guidance by operation

| Operation | Minimum VRAM | Minimum RAM |
|---|---|---|
| Quantize 0.8B / 2B / 4B | ~5 GB | ~18 GB |
| Quantize 9B | ~200-400 MB peak (`--low-vram`) | ~36 GB |
| Strip / GGUF convert | none | approximately model size x 2 |
| Verify 4B text-only | ~3 GB | ~8 GB |

## Interpretation notes

- RAM is often the primary constraint because full-precision source weights are staged in CPU memory.
- VRAM is mainly needed for quantization and GPU inference.
- `--low-vram` trades runtime for reduced VRAM peak.

## Typical hardware scenarios

### Less than 8 GB VRAM

- Use `--low-vram` for quantization.
- Strip and GGUF conversion remain CPU/RAM operations.

### No GPU

- BNB quantization requires CUDA.
- Strip, verify (CPU fallback), and GGUF conversion can run on CPU.

### More than 8 GB VRAM

- Default quantization path is usually simpler/faster.
- Larger model sizes still require adequate RAM.

## Edge cases / limitations

> [!WARNING]
> Even with low VRAM usage, insufficient RAM can block quantization/convert flows.
> Check RAM first when planning 9B operations.

- Verify can fall back to CPU when CUDA move fails.
- Published numbers are empirical and depend on driver/runtime overhead.

## Related

- [Quickstart](quickstart.md)
- [Convert](convert.md)
- [Models](models.md)
