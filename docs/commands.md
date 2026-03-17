---
title: Commands
---

# Commands

## Purpose

This page is the command index for the toolkit CLI.
Use it to find the right command, then open the linked reference page.

## When to use

- You know the task (convert/strip/verify/upload) and need the right command.
- You need a starting point before reading command-specific details.

## Command map

| CLI command | Module | Purpose | Docs |
|---|---|---|---|
| `qwen35-convert` | `qwen35_toolkit.convert` | f16 → BNB nf4 ⚠️ Qwen3.5 only | [Convert](convert.md) |
| `qwen35-strip` | `qwen35_toolkit.strip` | Remove visual tower → text-only | [Strip](strip.md) |
| `qwen35-verify` | `qwen35_toolkit.verify` | Universal sanity-check — BNB 4-bit and f16/bf16, auto-detected | [Verify](verify.md) |
| `qwen35-verify-qwen35` | `qwen35_toolkit.verify_qwen35` | Qwen3.5 check — thinking ON/OFF + image | [Verify](verify.md) |
| `qwen35-upload` | `qwen35_toolkit.tools.upload` | Sync with HF Hub — push and pull | [Upload](upload.md) |
| `qwen35-inspect` | `qwen35_toolkit.tools.inspect_safetensors` | Print weight prefixes and sizes | [Tools](tools.md) |
| `qwen35-render-mermaid` | `qwen35_toolkit.tools.render_mermaid` | Pre-render Mermaid diagrams → PNG for PDF export | [Tools](tools.md) |
| — | — | GGUF conversion + quantization (llama.cpp, external) | [GGUF](gguf.md) |

## Notes

- All commands support `--help` for full usage.
- `qwen35-convert` is Qwen3.5-specific.

## Related

- [Quickstart](quickstart.md)
- [Conversion pipeline](conversion-pipeline.md)
- [Verify](verify.md)
