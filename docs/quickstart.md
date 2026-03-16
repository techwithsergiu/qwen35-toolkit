---
title: Quickstart
---

# Quickstart

> **Start with 0.8B** to validate the pipeline on your hardware before moving to larger models.
> `unsloth/Qwen3.5-0.8B` is a community-maintained fp16 mirror of `Qwen/Qwen3.5-0.8B`.

## Prerequisites

See [Setup](setup.md) for full installation instructions.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .   # or: pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
hf auth login
```

---

## Path A — BNB text-only (training target)

### Step 1 — Quantize

```bash
qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit
# repeat for 2B, 4B; use --low-vram for 9B
```

See [Convert](convert.md) for internals, skip modules, and `--verbose`.

### Step 2 — Strip visual tower

```bash
qwen35-strip --model ./Qwen3.5-0.8B-bnb-4bit --output ./Qwen3.5-text-0.8B-bnb-4bit
# repeat for 2B, 4B, 9B
```

See [Strip](strip.md) for dropped prefixes, config rebuild, and chat template patch.

### Step 3 — Verify

```bash
qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit    # VLM: thinking + image
qwen35-verify        --model ./Qwen3.5-text-0.8B-bnb-4bit # text-only
```

See [Verify](verify.md) for device strategy, image test, and function tree.

### Step 4 — Upload

```bash
qwen35-upload --local ./Qwen3.5-0.8B-bnb-4bit      --repo <your-hf-username>/Qwen3.5-0.8B-bnb-4bit
qwen35-upload --local ./Qwen3.5-text-0.8B-bnb-4bit --repo <your-hf-username>/Qwen3.5-text-0.8B-bnb-4bit
```

See [Upload](upload.md) for more details.

---

## Path B — f16 text-only + GGUF (inference / merge base)

### Step 1 — Strip visual tower (f16)

```bash
qwen35-strip --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-text-0.8B --mode f16
# repeat for 2B, 4B, 9B
```

### Step 2 — Verify

```bash
qwen35-verify --model ./Qwen3.5-text-0.8B
```

### Step 3 — Convert to GGUF

Requires llama.cpp — see [GGUF](gguf.md) for build instructions.

```bash
python llama.cpp/convert_hf_to_gguf.py ./Qwen3.5-text-0.8B \
    --outtype f16 \
    --outfile ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf
```

### Step 4 — Quantize GGUF

```bash
./llama.cpp/build/bin/llama-quantize \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-F16.gguf \
    ./Qwen3.5-text-0.8B-GGUF/Qwen3.5-text-0.8B-Q4_K_M.gguf \
    Q4_K_M
```

All quant types (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S) in [gguf.md](gguf.md).

### Step 5 — Upload

```bash
qwen35-upload --local ./Qwen3.5-text-0.8B-GGUF --repo <your-hf-username>/Qwen3.5-text-0.8B-GGUF
```

---

## Optional — Inspect safetensors

Useful before and after strip to confirm which components are present:

```bash
qwen35-inspect ./Qwen3.5-0.8B-bnb-4bit
qwen35-inspect ./Qwen3.5-text-0.8B-bnb-4bit
```

See [Tools](tools.md) for more details.
