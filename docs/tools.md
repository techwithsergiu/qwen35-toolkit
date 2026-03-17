---
title: Utility tools
---

# Utility tools

## Purpose

Reference for auxiliary toolkit commands:
- `qwen35-inspect` for safetensors prefix/size inspection,
- `qwen35-render-mermaid` for Mermaid-to-PNG markdown rendering.

## When to use

- You want to validate which weight groups exist in a model directory.
- You need Mermaid diagrams rendered for PDF/export workflows.

## Command map

| Command | Purpose | Docs section |
|---|---|---|
| `qwen35-inspect` | Inspect safetensors prefixes and cumulative sizes | [qwen35-inspect](#qwen35-inspect) |
| `qwen35-render-mermaid` | Render Mermaid blocks to PNG + markdown image refs | [qwen35-render-mermaid](#qwen35-render-mermaid) |

## qwen35-inspect

### Purpose

Read safetensors shard headers (without loading tensor payloads), then print unique top-level prefixes and cumulative sizes.

### Syntax

```text
qwen35-inspect <local_model_dir>
```

### Mechanics

```text
1. Read shard headers:
   - Parse safetensors metadata only (no tensor payload load).
2. Group keys:
   - Aggregate by top-level prefix.
3. Report:
   - Print cumulative size per prefix and total size.
```

### Examples

```bash
qwen35-inspect ./Qwen3.5-0.8B-bnb-4bit
qwen35-inspect ./Qwen3.5-text-0.8B-bnb-4bit
```

### Output examples

### Example output (VLM checkpoint)

```text
Found 1 shard(s)

PREFIX                                            SIZE
─────────────────────────────────────────────────────
model.language_model                           0.71 GB
model.visual                                   0.19 GB

TOTAL                                          0.90 GB
```

Interpretation: VLM BNB checkpoint contains both text and visual branches.

Stable fields: prefix naming and table layout.
Variable fields: exact sizes and shard count.

### Example output (text-only checkpoint)

```text
Found 1 shard(s)

PREFIX                                            SIZE
─────────────────────────────────────────────────────
model.language_model                           0.71 GB

TOTAL                                          0.71 GB
```

Interpretation: text-only BNB checkpoint contains only language-model weights.

Stable fields: prefix naming semantics and table columns.
Variable fields: sizes and shard count.

### Edge cases / limitations

- Local paths only (HF repo ids are not downloaded automatically).
- Output reflects safetensors headers and key structure, not runtime inference behavior.

## qwen35-render-mermaid

### Purpose

Render Mermaid code blocks in markdown to PNG files and produce an export-friendly markdown file with image links.

### Syntax

```text
qwen35-render-mermaid --input <doc>.md [options]
```

### Mechanics

```text
1. Parse markdown:
   - Find fenced `mermaid` blocks.
2. Render diagrams:
   - Generate PNG files via Mermaid CLI.
3. Rewrite output markdown:
   - Replace mermaid blocks with image references.
4. Save artifacts:
   - Output markdown + diagrams directory.
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Input markdown file |
| `--output` | `<input_stem>_pdf.md` | Output markdown file |
| `--diagrams-dir` | `<output_dir>/diagrams/` | Directory for rendered PNG files |
| `--scale` | `2.0` | PNG render scale (2.0 default, 3.0 for print) |

### Setup (one-time)

```bash
npm install -g @mermaid-js/mermaid-cli
npx puppeteer browsers install chrome-headless-shell
```

### Examples

```bash
# Default output
qwen35-render-mermaid --input README.md

# Custom output file
qwen35-render-mermaid --input README.md --output README_pdf.md

# Higher image scale
qwen35-render-mermaid --input README.md --scale 3.0
```

```bash
# Batch usage example (BNB 4-bit VLM cards)
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-0.8B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-2B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-4B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-9B-bnb-4bit/README.md
```

### Edge cases / limitations

- Requires Mermaid CLI + headless browser runtime.
- Designed for markdown files containing fenced `mermaid` blocks.

## Related

- [Models](models.md)
- [Conversion pipeline](conversion-pipeline.md)
- [Quickstart](quickstart.md)
