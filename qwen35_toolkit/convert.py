"""
convert.py
────────────────────────────
Converts a Qwen3.5 model from full-precision (f16/bf16) to BNB 4-bit (nf4).
Output is compatible with Unsloth LoRA fine-tuning.

⚠️  Qwen3.5-specific: uses Qwen3_5ForConditionalGeneration.
    For other model families use a generic AutoModelForCausalLM variant.

Strategy (avoids the Params4bit/_is_hf_initialized version mismatch bug):
  1. Load the full model into CPU RAM — no GPU involved, no accelerate dispatch.
  2. Replace every eligible nn.Linear with a BNB Linear4bit in-place (still RAM).
  3. Quantize: BNB nf4 compression is triggered by the first .cuda() call per
     parameter. Two modes available via --low-vram:
       default  : model.cuda() — moves everything to GPU at once.
                  Fast, but requires ~5 GB VRAM for a 4B model.
       --low-vram: quantize_layer_by_layer() — moves one layer at a time,
                  immediately returns the quantized weights to CPU RAM.
                  Peak VRAM ≈ one layer (~200–400 MB). Saves to CPU.
  4. Save the quantized model.
  5. Copy optional tokenizer/processor side-car files that save_pretrained skips.

Hardware requirements:
  RAM  : ~18 GB  (full bf16 model must fit in RAM before quantization)
  VRAM : ~5 GB   (default mode) or ~400 MB (--low-vram mode)

Usage:
  qwen35-convert --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-0.8B-bnb-4bit
  qwen35-convert --model unsloth/Qwen3.5-2B --output ./Qwen3.5-2B-bnb-4bit
  qwen35-convert --model unsloth/Qwen3.5-4B --output ./Qwen3.5-4B-bnb-4bit
  qwen35-convert --model unsloth/Qwen3.5-9B --output ./Qwen3.5-9B-bnb-4bit --low-vram
"""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn
import bitsandbytes.functional as bnb_F
from bitsandbytes.nn import Linear4bit, Params4bit
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "unsloth/Qwen3.5-0.8B"
DEFAULT_OUTPUT = "./Qwen3.5-0.8B-bnb-4bit"

# Modules whose names contain any of these strings are left at full precision.
# Matches Unsloth's convention — avoids quantizing embedding/head layers.
# Modules to keep at full precision (bf16). Any layer whose full dotted
# path contains one of these strings will not be quantized.
#
# Qwen3.5 structure:
#   model.language_model.embed_tokens  → "embed_tokens"
#   lm_head                            → "lm_head"
#   model.visual.*                     → "visual"
#   model.visual.merger.*              → "merger"  (also covered by "visual")
#
# Extra entries ("embedding", "vision_tower", "multi_modal_projector",
# "modality_projection", "router") are harmless no-ops for Qwen3.5 text
# but kept for compatibility if this script is adapted to other models.
# llm_int8_skip_modules must use exact module paths as they appear in the
# model tree, not just leaf names. transformers matches by prefix, so
# "model.visual" covers model.visual.* and "lm_head" covers lm_head.
# Kept minimal — only fields transformers/Unsloth actually reads on load.
SKIP_MODULES_CONFIG: list[str] = [
    "lm_head",
    "model.visual",
    "model.language_model.embed_tokens",
]

# SKIP_MODULES_CONVERT uses short names for the in-process replacement walk
# (replace_with_bnb_linear4bit checks if any skip keyword appears in full_name).
SKIP_MODULES: set[str] = {
    "embed_tokens",
    "embedding",
    "lm_head",
    "visual",
    "vision_tower",
    "merger",
    "multi_modal_projector",
    "modality_projection",
    "router",
}

# Optional side-car files that save_pretrained does not copy automatically.
# Missing files are silently skipped so the script stays compatible with
# text-only models.
OPTIONAL_SIDECAR_FILES: list[str] = [
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_vram(label: str = "") -> None:
    """Print current VRAM usage. No-op when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    free  = (props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    tag   = f"  [{label}]" if label else ""
    print(f"   GPU : {props.name}  |  VRAM : {free:.1f} GB free / {total:.1f} GB total{tag}")


# ── Layer-by-layer quantization (low VRAM mode) ───────────────────────────────

def quantize_layer_by_layer(model: nn.Module) -> None:
    """
    Quantize every Params4bit weight one layer at a time.

    Each weight is moved to CUDA (which triggers BNB nf4 quantization), then
    immediately moved back to CPU RAM. Only one layer occupies VRAM at a time,
    so peak VRAM ≈ the largest single layer (~200–400 MB for Qwen3.5).

    The resulting model lives entirely on CPU and can be passed directly to
    save_pretrained() — BNB serialises the quantized uint8 data regardless
    of device.

    Use this instead of model.cuda() when VRAM is too small to hold the full
    quantized model (e.g. 9B on a 7.7 GB card).
    """
    layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, Linear4bit)
    ]
    total = len(layers)
    for i, (_, mod) in enumerate(layers, 1):
        # Move weight to CUDA → BNB quantizes it in-place.
        mod.weight = mod.weight.cuda()
        # Move the now-quantized uint8 tensor + quant_state back to CPU.
        mod.weight = mod.weight.cpu()
        if mod.bias is not None:
            mod.bias = mod.bias.cuda().cpu()
        if i % 50 == 0 or i == total:
            print(f"   quantized {i}/{total} layers …")
    torch.cuda.empty_cache()


# ── Layer replacement ──────────────────────────────────────────────────────────

def replace_with_bnb_linear4bit(
    model: nn.Module,
    skip: set[str],
    prefix: str = "",
    verbose: bool = False,
) -> int:
    """
    Recursively walk the module tree and swap every eligible nn.Linear with a
    BNB Linear4bit layer.

    Weights remain on CPU; BNB quantizes them on-the-fly when .cuda() is called.

    Args:
        model:   Root module to walk.
        skip:    Set of name fragments — any layer whose full path contains one
                 of these strings is left unmodified.
        prefix:  Internal use only (tracks the current name path).
        verbose: If True, print every skip decision.

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        # Early-exit: if this node's path contains any skip keyword,
        # bail out of the ENTIRE subtree — do not recurse further.
        if any(s in full_name for s in skip):
            if verbose:
                print(f"   └── skipped      {full_name}")
            continue

        if isinstance(child, nn.Linear):
            new_layer = Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            new_layer.weight = Params4bit(
                child.weight.data,
                requires_grad=False,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(model, name, new_layer)
            replaced += 1
        else:
            replaced += replace_with_bnb_linear4bit(child, skip, full_name, verbose)
    return replaced


def dequantize_skip_modules(model: nn.Module, skip: set[str]) -> int:
    """
    Walk the model and convert any Linear4bit layers that belong to
    skip-modules back to regular nn.Linear (bf16).

    This is needed when the source model was already quantized (e.g. some
    unsloth Hub models load with BNB quantization applied automatically via
    their config.json). Our replace_with_bnb_linear4bit skips those subtrees,
    but the pre-existing Linear4bit layers remain — causing visual tower
    quantization bugs.

    Runs BEFORE replace_with_bnb_linear4bit so the subsequent replacement
    only operates on clean nn.Linear layers.

    Returns:
        Number of layers restored to full precision.
    """
    restored = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, Linear4bit):
            continue
        # Only touch layers that belong to a skip subtree.
        if not any(s in name for s in skip):
            continue

        # Dequantize: reconstruct the bf16 weight from the stored quant state.
        try:
            weight_fp = bnb_F.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
                quant_type="nf4",
            ).to(torch.bfloat16)
        except Exception:
            # Fallback: the weight may already be bf16 if loaded on CPU.
            weight_fp = module.weight.data.to(torch.bfloat16)

        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
        )
        new_linear.weight = nn.Parameter(weight_fp, requires_grad=False)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(
                module.bias.data.to(torch.bfloat16), requires_grad=False
            )

        # Navigate to the parent module and replace the child.
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear)
        restored += 1

    return restored


# ── Side-car file copy ─────────────────────────────────────────────────────────

def copy_sidecar_files(model_name: str, output_dir: str) -> None:
    """
    Download optional tokenizer/processor files from the HF Hub and copy them
    into output_dir. Files absent from the source repo are silently skipped.

    Args:
        model_name: HF Hub repo id or local path used as the source.
        output_dir: Destination directory (the converted model folder).
    """
    print("⏳ Copying optional side-car files …")
    copied = skipped = 0

    for filename in OPTIONAL_SIDECAR_FILES:
        src_local = Path(model_name) / filename
        dst       = Path(output_dir) / filename

        if src_local.exists():
            shutil.copy2(src_local, dst)
            print(f"   ✅ {filename}  (local copy)")
            copied += 1
            continue

        try:
            local_path = hf_hub_download(repo_id=model_name, filename=filename)
            shutil.copy2(local_path, dst)
            print(f"   ✅ {filename}  (downloaded)")
            copied += 1
        except Exception:
            print(f"   –  {filename}  (not found in repo, skipping)")
            skipped += 1

    print(f"   side-car: {copied} copied, {skipped} skipped\n")


# ── Config patches ─────────────────────────────────────────────────────────────

def patch_tokenizer_config(output_dir: str, tokenizer) -> None:
    """Write chat_template into tokenizer_config.json if save_pretrained omitted it."""
    path = Path(output_dir) / "tokenizer_config.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["chat_template"] = tokenizer.chat_template
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("   ✅ chat_template written to tokenizer_config.json")


# ── Config rebuild ─────────────────────────────────────────────────────────────

def _load_source_config(model_name: str) -> dict:
    """
    Read config.json from the source model — local path or HF Hub.
    Used as the base for the rebuilt output config.
    """
    local = Path(model_name) / "config.json"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    # HF Hub — hf_hub_download caches locally.
    src = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(src, encoding="utf-8") as f:
        return json.load(f)


def build_bnb_config(src_cfg: dict) -> dict:
    """
    Build a clean BNB 4-bit config.json from the original source config.

    Starts from the original config (keeping architectures, vision_config,
    text_config, etc. intact) and injects a clean quantization_config.
    Removes any pre-existing quantization_config that might have come
    from the source being a previously quantized checkpoint.

    llm_int8_skip_modules uses exact model-tree paths so transformers
    matches by prefix on load — "model.visual" covers model.visual.*
    and prevents re-quantization of the visual tower.
    """
    out = dict(src_cfg)

    # Remove any stale quantization_config from the source
    # (e.g. if source was already a BNB checkpoint).
    out.pop("quantization_config", None)

    out["quantization_config"] = {
        "quant_method":                     "bitsandbytes",
        "load_in_4bit":                     True,
        "load_in_8bit":                     False,
        "bnb_4bit_quant_type":              "nf4",
        "bnb_4bit_compute_dtype":           "bfloat16",
        "bnb_4bit_use_double_quant":        True,
        "bnb_4bit_quant_storage":           "uint8",
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_skip_modules":            SKIP_MODULES_CONFIG,
    }
    return out


def write_config(model_name: str, output_dir: str) -> None:
    """
    Rebuild config.json from scratch using the original source config.

    Called after save_pretrained — overwrites whatever save_pretrained wrote
    with a config built from the known-good source, plus a clean
    quantization_config block.
    """
    src_cfg = _load_source_config(model_name)
    out_cfg = build_bnb_config(src_cfg)
    path    = Path(output_dir) / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)
    skip = out_cfg["quantization_config"]["llm_int8_skip_modules"]
    print(f"   ✅ config.json rebuilt from source  |  skip_modules: {skip}")


# ── Main conversion ────────────────────────────────────────────────────────────

def convert(model_name: str, output_dir: str, low_vram: bool = False, verbose: bool = False) -> None:
    """
    Full conversion pipeline: load → replace layers → quantize → save.

    Args:
        model_name: HF Hub repo id (e.g. "unsloth/Qwen3.5-0.8B") or local path.
        output_dir: Where to write the converted model.
        low_vram:   If True, quantize one layer at a time (peak VRAM ~400 MB)
                    and keep the model on CPU for saving. Use when the full
                    quantized model does not fit in VRAM.
        verbose:    If True, print every layer replacement and skip decision.
    """
    print(f"\n{'─' * 60}")
    print(f"  model  : {model_name}")
    print(f"  output : {output_dir}")
    print(f"{'─' * 60}\n")
    print_vram("before load")

    # ── 1. Load full model into CPU RAM ────────────────────────────────────────
    print("⏳ Loading model into CPU RAM (bf16, no quantization yet) …")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("✅ Model loaded into RAM.\n")

    # ── 1b. Dequantize any pre-quantized skip-module layers ───────────────────
    # Some source models (e.g. unsloth Hub checkpoints) may have a
    # quantization_config in config.json that causes transformers to load
    # visual / embedding layers as Linear4bit automatically. Restore them
    # to plain nn.Linear before our replacement pass runs.
    n_deq = dequantize_skip_modules(model, SKIP_MODULES)
    if n_deq:
        print(f"   ⚠️  Restored {n_deq} pre-quantized skip-module layer(s) to bf16")
        print("      (source model had quantization_config applied on load)\n")

    # ── 2. Replace Linear → Linear4bit in-place (still on CPU) ────────────────
    print("⏳ Replacing linear layers with BNB Linear4bit …")
    n = replace_with_bnb_linear4bit(model, SKIP_MODULES, verbose=verbose)
    print(f"   {n} layers replaced\n")
    gc.collect()

    # ── 3. Quantize — BNB nf4 compression is triggered by the first .cuda() ───
    # IMPORTANT: do not skip this step.
    # Params4bit on CPU only wraps the original bf16 weights — no compression
    # has occurred yet. The actual nf4 quantization (absmax computation, 4-bit
    # packing) is triggered by BNB internally on the first .cuda() call for each
    # parameter. Calling save_pretrained() before this step would write full
    # bf16 weights to disk — i.e. a ~18 GB model instead of ~4 GB.
    if low_vram:
        print("⏳ Quantizing layer by layer (--low-vram: peak VRAM ≈ one layer) …")
        quantize_layer_by_layer(model)
        print("✅ Model quantized. Weights remain on CPU for saving.\n")
    else:
        print("⏳ Moving to GPU (this triggers BNB nf4 quantization per layer) …")
        model = model.cuda()
        torch.cuda.empty_cache()
        print("✅ Model quantized and on GPU.\n")
        print_vram("after quantize")

    # ── 4. Load tokenizer ──────────────────────────────────────────────────────
    print("⏳ Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Tokenizer loaded.\n")

    # ── 5. Save model + tokenizer ──────────────────────────────────────────────
    print(f"💾 Saving to '{output_dir}' …")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    patch_tokenizer_config(output_dir, tokenizer)
    write_config(model_name, output_dir)

    # ── 6. Copy optional side-car files ───────────────────────────────────────
    copy_sidecar_files(model_name, output_dir)

    print(f"✅ Done. Saved to: {output_dir}\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert a Qwen3.5 model from f16 to BNB nf4 (4-bit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HF Hub repo id or local path of the source f16 model.",
    )
    ap.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Directory where the quantized model will be saved.",
    )
    ap.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print every layer replacement and skip decision during conversion.",
    )
    ap.add_argument(
        "--low-vram", action="store_true", default=False,
        help=(
            "Quantize one layer at a time and keep weights on CPU. "
            "Peak VRAM ≈ one layer (~400 MB) instead of the full model (~5 GB). "
            "Use when the quantized model does not fit in VRAM."
        ),
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token for private repos. Omit to use credentials from `hf auth login`.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-convert CLI command."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    convert(
        model_name=args.model,
        output_dir=args.output,
        low_vram=args.low_vram,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
