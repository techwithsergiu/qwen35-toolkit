---
title: Conversion pipeline
---

# Conversion pipeline

Two independent paths from the original Qwen3.5 f16 source.

**Path A** — BNB text-only (training target): quantize → strip → verify → upload.  
**Path B** — f16 text-only + GGUF (inference / merge base): strip → verify → convert → quantize → upload.

```mermaid
flowchart TD
    SRC["Qwen/Qwen3.5-{size}<br/>f16 · source"]

    subgraph PATH_A ["Path A — BNB text-only  (training target)"]
        BNBVLM["Qwen3.5-{size}-bnb-4bit<br/>BNB NF4 · VLM"]
        TEXTBNB["Qwen3.5-text-{size}-bnb-4bit<br/>BNB NF4 · text-only"]
        V1{{"✅ verified"}}
        V3{{"✅ verified"}}
        BNBVLM -->|"qwen35-strip --mode bnb"| TEXTBNB
        BNBVLM -->|"qwen35-verify-qwen35"| V1
        TEXTBNB -->|"qwen35-verify"| V3
    end

    subgraph PATH_B ["Path B — f16 text-only + GGUF  (inference / merge base)"]
        TEXTF16["Qwen3.5-text-{size}<br/>bf16 · text-only"]
        V2{{"✅ verified"}}
        GGUUF16["Qwen3.5-text-{size}.gguf<br/>GGUF f16"]
        Q4["Q4_K_M ✅ main"]
        Q5KM["Q5_K_M very good quality"]
        Q6K["Q6_K excellent quality"]
        Q8["Q8_0 near-lossless"]
        TEXTF16 -->|"qwen35-verify"| V2
        TEXTF16 -->|"convert_hf_to_gguf.py"| GGUUF16
        GGUUF16 -->|"llama-quantize"| Q4
        GGUUF16 -->|"llama-quantize"| Q5KM
        GGUUF16 -->|"llama-quantize"| Q6K
        GGUUF16 -->|"llama-quantize"| Q8
    end

    HUB[("HuggingFace Hub")]

    SRC -->|"qwen35-convert"| BNBVLM
    SRC -->|"qwen35-strip --mode f16"| TEXTF16
    V1 -->|"qwen35-upload"| HUB
    V3 -->|"qwen35-upload"| HUB
    V2 -->|"qwen35-upload"| HUB
    Q4   -->|"qwen35-upload"| HUB
    Q5KM -->|"qwen35-upload"| HUB
    Q6K  -->|"qwen35-upload"| HUB
    Q8   -->|"qwen35-upload"| HUB

    style TEXTBNB fill:#dcfce7,stroke:#16a34a
    style Q4     fill:#fce7f3,stroke:#db2777
    style Q5KM   fill:#fce7f3,stroke:#db2777
    style Q6K    fill:#fce7f3,stroke:#db2777
    style Q8     fill:#fce7f3,stroke:#db2777
    style HUB    fill:#f3e8ff,stroke:#9333ea
```

## Commands per step

| Step | Command | Doc |
|------|---------|-----|
| Quantize f16 → BNB 4-bit | `qwen35-convert` | [Convert](convert.md) |
| Strip visual tower (BNB) | `qwen35-strip --mode bnb` | [Strip](strip.md) |
| Strip visual tower (f16) | `qwen35-strip --mode f16` | [Strip](strip.md) |
| Verify VLM model | `qwen35-verify-qwen35` | [Verify](verify.md) |
| Verify text-only model | `qwen35-verify` | [Verify](verify.md) |
| Convert to GGUF | `convert_hf_to_gguf.py` (llama.cpp) | [GGUF](gguf.md) |
| Quantize GGUF | `llama-quantize` (llama.cpp) | [GGUF](gguf.md) |
| Upload to Hub | `qwen35-upload` | [Upload](upload.md) |
