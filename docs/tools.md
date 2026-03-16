---
title: Utility tools
---

# Utility tools

## qwen35-inspect

Reads safetensors shard headers (no weights loaded — only the JSON header of
each shard, a few KB per file) and prints all unique top-2-level weight prefixes
with their cumulative sizes. Runs instantly on any model size, including 70B.

**Useful for:**
- Confirming which components are present (`language_model`, `visual`, `lm_head`, …)
- Estimating visual tower size before stripping
- Verifying that a `qwen35-strip` run removed the expected prefixes

### Usage

```bash
qwen35-inspect <local_model_dir>

# Examples
qwen35-inspect ./Qwen3.5-0.8B-bnb-4bit
qwen35-inspect ./Qwen3.5-text-0.8B-bnb-4bit
```

> Local paths only — HF Hub repo ids are not supported (no download).

### Example output

```text
Found 5 shard(s)

PREFIX                                            SIZE
─────────────────────────────────────────────────────
model.language_model                          3.54 GB
model.visual                                  0.89 GB
lm_head                                      90.0 MB
model.visual.merger                          12.3 MB

TOTAL                                         4.54 GB
```

---

## qwen35-render-mermaid

Pre-renders every ` ```mermaid ` block in a Markdown file to PNG using the
Mermaid CLI (`mmdc`), then outputs a new `.md` file with blocks replaced by
`![]()` image references. Needed for PDF export from VS Code Office Viewer,
which does not render Mermaid natively.

### Setup (one-time)

```bash
npm install -g @mermaid-js/mermaid-cli
npx puppeteer browsers install chrome-headless-shell
```

### Usage

```bash
qwen35-render-mermaid --input <doc>.md
# → <doc>_pdf.md
# → diagrams/diagram_01.png, diagram_02.png  …  (mmdc_config.json auto-deleted)

# Custom output path
qwen35-render-mermaid --input README.md --output my_doc_pdf.md

# Higher scale for print quality (default: 2.0)
qwen35-render-mermaid --input README.md --scale 3.0
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Input Markdown file |
| `--output` | `<input_stem>_pdf.md` | Output Markdown file |
| `--diagrams-dir` | `<output_dir>/diagrams/` | Directory for PNG files |
| `--scale` | `2.0` | PNG scale factor (2.0 = retina, 3.0 = print quality) |

### Batch usage (all model card families)

```bash
# BNB 4-bit VLM
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-0.8B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-2B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-4B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-9B-bnb-4bit/README.md

# Text bf16
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-0.8B/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-2B/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-4B/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-9B/README.md

# Text BNB 4-bit
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-0.8B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-2B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-4B-bnb-4bit/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-9B-bnb-4bit/README.md

# GGUF
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-0.8B-GGUF/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-2B-GGUF/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-4B-GGUF/README.md
qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-9B-GGUF/README.md
```
