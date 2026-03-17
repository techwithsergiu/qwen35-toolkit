---
title: Setup
---

# Setup

> [!NOTE]
> This setup is for the toolkit repository only.
> For training, use [qwen-qlora-train](https://techwithsergiu.github.io/qwen-qlora-train).

## Prerequisites

- Arch Linux (or any Linux with NVIDIA driver)
- Python 3.11
- CUDA via driver (`nvidia-smi` works → CUDA is fine)

## Steps

### Step 1 — Create and activate a Python environment
Create an isolated environment before installing dependencies.

```bash
yay -S python311
python3.11 -m venv venv
source venv/bin/activate
```

Success criteria: `python --version` points to your venv and Python 3.11.

### Step 2 — Install toolkit dependencies and package
Install PyTorch first, then install the toolkit.

```bash
# 1) PyTorch with CUDA (install before toolkit)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2a) Editable install (local clone)
pip install -e .

# 2b) Install directly from GitHub
pip install git+https://github.com/techwithsergiu/qwen35-toolkit.git
```

Success criteria: `qwen35-convert --help` prints CLI usage.

### Step 3 — Authenticate with Hugging Face
Authenticate once for Hub download/upload operations.

```bash
hf auth login
```

Success criteria: protected model/repo access works without extra prompts.

## Expected result

- Toolkit commands are available in shell (`qwen35-*`).
- You can run local commands and access HF Hub.

## Common failures

- `nvidia-smi` not found -> install/fix NVIDIA driver first.
- `qwen35-...: command not found` -> reactivate venv and reinstall package.
- HF 401/403 errors -> rerun `hf auth login` or pass `--hf-token`.

Alternatively, pass `--hf-token hf_...` per command or set `HF_TOKEN` as an environment variable.

## Related

- [Quickstart](quickstart.md)
- [Commands](commands.md)
- [Hardware](hardware.md)
