---
layout: home

hero:
  name: Qwen35 toolkit
  text: Convert, strip, verify Qwen3.5 models
  tagline: >
    Qwen3.5 models are often distributed as full-precision VLM checkpoints.
    This toolkit prepares text-only variants for training and inference:
    BNB conversion, visual tower strip, verification, and upload.
  actions:
    - theme: brand
      text: Get started
      link: /quickstart
    - theme: alt
      text: Published models
      link: /models
    - theme: alt
      text: GitHub →
      link: https://github.com/techwithsergiu/qwen35-toolkit

features:
  - icon: ⚡
    title: BNB 4-bit quantization
    details: >
      Converts full-precision f16 models to BNB NF4 4-bit on CPU,
      then quantizes layer-by-layer or all at once. A 4B model fits
      in ~5 GB VRAM. Use --low-vram for 9B on a 7.7 GB card.
    link: /convert
    linkText: Learn more

  - icon: ✂️
    title: Visual tower strip
    details: >
      Removes the visual tower at the file level — no model load,
      no rounding error. Saves ~0.2–0.9 GB and produces a clean
      text-only model ready for LoRA training. VLM fine-tuning
      is not the goal.
    link: /strip
    linkText: Learn more

  - icon: ✅
    title: Verification before publishing
    details: >
      Checks quantization correctness, layer precision, and runs
      inference — thinking ON/OFF and image test for VLM models.
      Auto-detects BNB vs f16, local path vs HF Hub, device strategy.
    link: /verify
    linkText: Learn more

  - icon: 🔄
    title: HuggingFace Hub sync
    details: >
      Six sync modes — push, pull, diff, patch, and dry-runs.
      SHA256-level comparison via LFS pointers means large shards
      are never re-uploaded unless they actually changed.
    link: /upload
    linkText: Learn more

  - icon: 🛠️
    title: GGUF pipeline
    details: >
      Strip → f16 text-only → GGUF f16 → quantized (Q4_K_M recommended).
      Designed for llama.cpp workflows with explicit conversion and
      quantization steps.
    link: /gguf
    linkText: Learn more

  - icon: 🖥️
    title: Hardware-aware workflow
    details: >
      The docs include tested constraints and fallback paths for limited
      VRAM setups (for example RTX 3070 7.7 GB). Check the hardware guide
      before choosing model size.
    link: /hardware
    linkText: Learn more
---

## Part of a two-repo ecosystem

| Repo | Purpose |
|------|---------|
| **qwen35-toolkit** (this repo) | Model prep — BNB quantization, visual tower strip, verify, upload |
| [qwen-qlora-train](https://techwithsergiu.github.io/qwen-qlora-train) | LoRA training, adapter inference, CPU merge |
