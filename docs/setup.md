---
title: Setup
---

# Setup

## Prerequisites

- Arch Linux (or any Linux with NVIDIA driver)
- Python 3.11
- CUDA via driver (`nvidia-smi` works → CUDA is fine)

```bash
yay -S python311
python3.11 -m venv venv
source venv/bin/activate
```

## Install

```bash
# 1. PyTorch with CUDA (must be installed before this package)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2a. Editable install (local clone)
pip install -e .

# 2b. Install directly from GitHub
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
```

> For LoRA fine-tuning of prepared models, see
> [qwen-qlora-train](https://github.com/techwithsergiu/qwen-qlora-train).

## Authentication

`huggingface_hub` is installed automatically with this package. Run once before using any command:

```bash
hf auth login
```

Alternatively, pass `--hf-token hf_...` per command or set `HF_TOKEN` as an environment variable.
