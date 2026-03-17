---
title: Quickstart
---

# Quickstart

> [!NOTE]
> Recommended starting point: `0.8B` first.
> On smaller GPUs (for example RTX 3070 7.7 GB), 8B/9B steps often require fallback modes (`--low-vram`, CPU fallback in verify).

## Prerequisites

See [Setup](setup.md) for full install details.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .   # or: pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
hf auth login
```

## Steps

### Path A — BNB text-only (training target)

### Step 1 — Quantize source model to BNB 4-bit
Create BNB VLM checkpoint.

```bash
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit
# repeat for 2B, 4B; use --low-vram for 9B
```

Success criteria: output directory contains BNB model files.

### Step 2 — Strip visual tower
Create text-only BNB checkpoint.

```bash
qwen35-strip --model ./Qwen3.5-0.8B-bnb-4bit --output ./Qwen3.5-text-0.8B-bnb-4bit
# repeat for 2B, 4B, 9B
```

Success criteria: stripped model has no visual tower weights.

### Step 3 — Verify both checkpoints
Run Qwen3.5-specific verification for VLM and generic verification for text-only.

```bash
qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit
qwen35-verify        --model ./Qwen3.5-text-0.8B-bnb-4bit
```

Success criteria: verification steps complete without blocking errors.

### Step 4 — Upload to Hub
Push both model variants.

```bash
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit      --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit
qwen35-upload --local ./Qwen3.5-text-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-text-0.8B-bnb-4bit
```

Success criteria: remote repo contains expected files and metadata.

### Path B — f16 text-only + GGUF (inference / merge base)

### Step 1 — Strip visual tower from f16 source
Create text-only f16/bf16 model.

```bash
qwen35-strip --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-text-0.8B --mode f16
# repeat for 2B, 4B, 9B
```

Success criteria: text-only model directory created.

### Step 2 — Verify text-only model
Ensure converted text model is valid before GGUF export.

```bash
qwen35-verify --model ./Qwen3.5-text-0.8B
```

Success criteria: verification passes.

### Step 3 — Convert to GGUF f16
Generate F16 GGUF file using llama.cpp converter.

```bash
python llama.cpp/convert_hf_to_gguf.py ./Qwen3.5-text-0.8B \
  --outtype f16 \
  --outfile ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf
```

Success criteria: `Qwen3.5-text-0.8B-F16.gguf` exists.

### Step 4 — Quantize GGUF
Create inference-friendly quant file.

```bash
./llama.cpp/build/bin/llama-quantize \
  ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
  ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q4_K_M.gguf \
  Q4_K_M
```

Success criteria: quantized GGUF file exists (for example `Q4_K_M`).

### Step 5 — Upload GGUF artifacts
Push GGUF directory.

```bash
qwen35-upload --local ./Qwen3.5-text-0.8B-GGUF --repo <your-hf-username>/Qwen3.5-text-0.8B-GGUF
```

Success criteria: GGUF files available on Hub repo.

## Expected result

- Path A: BNB VLM + BNB text-only models verified and published.
- Path B: text-only f16 model converted to GGUF and published.

## Common failures

- OOM during quantization -> rerun with `--low-vram`.
- Verify fails on mixed artifacts -> ensure strip output and source sizes match.
- GGUF conversion fails on VLM input -> use text-only model from `qwen35-strip --mode f16`.
- Hub auth errors -> rerun `hf auth login` or pass `--hf-token`.

## Related

- [Setup](setup.md)
- [Conversion pipeline](conversion-pipeline.md)
- [GGUF conversion](gguf.md)
