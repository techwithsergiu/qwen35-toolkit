#!/usr/bin/env python3
"""
strip.py  —  Remove the visual tower from a Qwen3.5 model.

Produces a text-only model that loads via Qwen3_5ForCausalLM (no visual
dependencies). Two modes:

  --mode bnb   BNB 4-bit input  →  text-only BNB 4-bit output   (for training)
  --mode f16   f16 / bf16 input →  text-only bf16 output        (for merge / GGUF)

Key idea:
  - Strip visual weights at the safetensors file level (no model load during strip).
  - Build a NEW text-only config.json from the original config["text_config"]
    as the root — this avoids accidentally inheriting vision dimensions.

Usage:
 qwen35-strip --model ./Qwen3.5-0.8B-bnb-4bit --output ./Qwen3.5-text-0.8B-bnb-4bit
 qwen35-strip --model unsloth/Qwen3.5-0.8B --output ./Qwen3.5-text-0.8B --mode f16

 qwen35-strip --model ./Qwen3.5-2B-bnb-4bit --output ./Qwen3.5-text-2B-bnb-4bit
 qwen35-strip --model unsloth/Qwen3.5-2B --output ./Qwen3.5-text-2B --mode f16

 qwen35-strip --model ./Qwen3.5-4B-bnb-4bit --output ./Qwen3.5-text-4B-bnb-4bit
 qwen35-strip --model unsloth/Qwen3.5-4B --output ./Qwen3.5-text-4B --mode f16

 qwen35-strip --model ./Qwen3.5-9B-bnb-4bit --output ./Qwen3.5-text-9B-bnb-4bit
 qwen35-strip --model unsloth/Qwen3.5-9B --output ./Qwen3.5-text-9B --mode f16
"""

import argparse
import gc
import json
import os
import re
import shutil
import struct
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
from transformers import AutoModelForCausalLM, AutoTokenizer

# Shared helpers — source detection, Hub size query, VRAM utils.
from qwen35_toolkit.verify import (
    is_hf_repo,
    model_size_gb,
    resolve_source,
    _pick_device,
    _move_to_cuda,
    count_quantized_layers,
    print_vram,
)


# ── Constants ──────────────────────────────────────────────────────────────────

# Weight key prefixes to drop (visual tower + multimodal glue layers).
DROP_PREFIXES: list[str] = [
    "model.visual",
    "model.vision_tower",
    "model.vision_model",
    "model.multi_modal_projector",
    "model.modality_projection",
    "model.merger",
    "model.router",
    "model.mm_projector",
    "model.vision_proj",
]

# config.json top-level keys that belong to vision — removed from the output.
VISION_CONFIG_KEYS: list[str] = [
    "vision_config",
    "image_token_id",
    "video_token_id",
    "image_token",
    "video_token",
    "vision_start_token_id",
    "vision_end_token_id",
    "vision_token_id",
]

VISUAL_SKIP_NAMES = {"model.visual", "visual", "vision_tower", "model.vision_tower"}

# Side-car files copied alongside the weight shards.
# preprocessor_config.json / processor_config.json / video_preprocessor_config.json
# are vision-specific and intentionally excluded from the text-only output.
SIDECAR_FILES: list[str] = [
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


# ── Source helpers ─────────────────────────────────────────────────────────────

def ensure_local(
    model_ref: str,
    hf_token: Optional[str] = None,
) -> tuple[Path, None]:
    """
    Return (local_path, None).

    Local paths are returned as-is.  HF repo ids are downloaded via
    snapshot_download into the default HF cache (~/.cache/huggingface/hub).
    Re-runs are instant — HF cache deduplicates by file hash.
    No temp dir is created, so nothing needs cleanup.
    """
    if not is_hf_repo(model_ref):
        p = Path(model_ref)
        if not p.is_dir():
            raise FileNotFoundError(f"Local model dir not found: {p}")
        return p, None

    print(f"   ⬇️  Fetching {model_ref} (HF cache) …")
    local = snapshot_download(
        repo_id=model_ref,
        token=hf_token,
    )
    print(f"   📦 Cached at: {local}")
    return Path(local), None


# ── Weight stripping ───────────────────────────────────────────────────────────

def _should_drop(key: str) -> bool:
    """Return True if the weight key belongs to the visual tower or multimodal glue layers."""
    return any(key.startswith(p) for p in DROP_PREFIXES)


def strip_weights(src: Path, dst: Path) -> float:
    """
    Copy safetensors shards from src to dst, dropping DROP_PREFIXES keys.
    Does not load the model — operates entirely at the file level.
    Returns approximate GB removed (estimated from header byte offsets).
    """
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in: {src}")

    removed_bytes = 0

    for shard in shards:
        with open(shard, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))

        drop_keys = [k for k in header if k != "__metadata__" and _should_drop(k)]

        if not drop_keys:
            shutil.copy2(shard, dst / shard.name)
            print(f"   📋 {shard.name}  (no visual keys — copied as-is)")
            continue

        for k in drop_keys:
            offsets = header[k].get("data_offsets", [0, 0])
            removed_bytes += int(offsets[1]) - int(offsets[0])

        tensors  = safetensors_load(str(shard))
        filtered = {k: v for k, v in tensors.items() if not _should_drop(k)}
        safetensors_save(filtered, str(dst / shard.name))
        print(f"   📋 {shard.name}  (dropped {len(drop_keys)} visual keys)")

    # Rebuild the shard index if present (multi-shard models).
    idx_src = src / "model.safetensors.index.json"
    if idx_src.exists():
        with open(idx_src, encoding="utf-8") as f:
            index = json.load(f)
        index["weight_map"] = {
            k: v for k, v in index.get("weight_map", {}).items()
            if not _should_drop(k)
        }
        # Recompute total_size from the newly written shards.
        try:
            total = sum(
                v.numel() * v.element_size()
                for s in dst.glob("*.safetensors")
                for v in safetensors_load(str(s)).values()
            )
            index.setdefault("metadata", {})["total_size"] = int(total)
        except Exception:
            pass
        with open(dst / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    return removed_bytes / 1024**3


# ── Config rebuild ─────────────────────────────────────────────────────────────

def build_text_config(src_cfg: dict, mode: str) -> dict:
    """
    Build a clean text-only config.json from the multimodal source config.

    Uses config["text_config"] as the root so that vision dimensions
    (hidden_size, num_layers, etc.) are never inherited from the VLM root.
    """
    if "text_config" not in src_cfg or not isinstance(src_cfg["text_config"], dict):
        raise ValueError(
            "config.json must contain a dict 'text_config' (Qwen3.5 VLM style)."
        )

    out: dict = dict(src_cfg["text_config"])
    out["architectures"] = ["Qwen3_5ForCausalLM"]
    out["model_type"]    = out.get("model_type", "qwen3_5_text")

    # Carry over global token ids (they live at the VLM root, not text_config).
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = src_cfg.get(k)
        if v is not None:
            out[k] = v

    # Safety pass — strip any residual vision keys.
    for k in VISION_CONFIG_KEYS:
        out.pop(k, None)

    if mode == "bnb":
        qc = src_cfg.get("quantization_config")
        if qc is None:
            raise ValueError(
                "mode=bnb but source config has no quantization_config.  "
                "Is this actually a packed BNB checkpoint?"
            )
        qc = dict(qc)
        # model.visual is gone — no need to skip it on load anymore.
        skip = [
            m for m in qc.get("llm_int8_skip_modules", [])
            if m not in VISUAL_SKIP_NAMES
        ]
        qc["llm_int8_skip_modules"] = skip
        out["quantization_config"] = qc
    else:
        out.pop("quantization_config", None)

    if "transformers_version" in src_cfg:
        out["transformers_version"] = src_cfg["transformers_version"]

    return out


def patch_config(src: Path, dst: Path, mode: str) -> None:
    """Read src/config.json, rebuild as text-only, write to dst/config.json."""
    with open(src / "config.json", encoding="utf-8") as f:
        src_cfg = json.load(f)
    out_cfg = build_text_config(src_cfg, mode)
    with open(dst / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)
    print(f"   ✅ config.json rebuilt  "
          f"(architectures: {out_cfg.get('architectures')}  "
          f"model_type: {out_cfg.get('model_type')})")


# ── Chat template patch ────────────────────────────────────────────────────────

def _strip_vision_from_template(template: str) -> str:
    """Remove image / video branches from a Qwen3.5 Jinja2 chat template."""
    template = re.sub(r'\{%-\s*set image_count\s*=.*?%\}\n?', '', template)
    template = re.sub(r'\{%-\s*set video_count\s*=.*?%\}\n?', '', template)
    template = re.sub(
        r"\{%-\s*if 'image' in item.*?(?=\{%-\s*elif 'video'|\{%-\s*elif 'text')",
        '',
        template,
        flags=re.DOTALL,
    )
    template = re.sub(
        r"\{%-\s*elif 'video' in item.*?(?=\{%-\s*elif 'text')",
        '',
        template,
        flags=re.DOTALL,
    )
    template = template.replace("{%- elif 'text' in item %}", "{%- if 'text' in item %}")
    return template


def patch_tokenizer_config(dst: Path) -> None:
    """Strip vision token branches from the chat template in tokenizer_config.json."""
    path = dst / "tokenizer_config.json"
    if not path.exists():
        print("   ⚠️  tokenizer_config.json not found — skipping template patch")
        return
    with open(path, encoding="utf-8") as f:
        tcfg = json.load(f)
    template = tcfg.get("chat_template", "")
    if not template:
        print("   ℹ️  chat_template absent — skipping")
        return
    patched = _strip_vision_from_template(template)
    if patched == template:
        print("   ℹ️  chat_template unchanged")
        return
    tcfg["chat_template"] = patched
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tcfg, f, indent=2, ensure_ascii=False)
    print(f"   ✅ chat_template stripped  ({len(template) - len(patched):+d} chars)")


def patch_chat_template_jinja(src: Path, dst: Path) -> None:
    """
    Copy chat_template.jinja from src to dst, stripping image/video branches.

    The .jinja file is a standalone chat template used by some tokenizers
    alongside (or instead of) the chat_template field in tokenizer_config.json.
    We apply the same _strip_vision_from_template() logic to keep both in sync.
    If the file does not exist in src, silently skip.
    """
    src_path = src / "chat_template.jinja"
    if not src_path.exists():
        return
    dst_path = dst / "chat_template.jinja"
    with open(src_path, encoding="utf-8") as f:
        template = f.read()
    patched = _strip_vision_from_template(template)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(patched)
    delta = len(template) - len(patched)
    if delta:
        print(f"   ✅ chat_template.jinja stripped  ({delta:+d} chars)")
    else:
        print("   ℹ️  chat_template.jinja copied unchanged")


# ── Side-car copy ──────────────────────────────────────────────────────────────

def copy_sidecars(src: Path, dst: Path) -> None:
    """Copy tokenizer and metadata side-car files that weight copy skips."""
    copied = []
    for name in SIDECAR_FILES:
        s = src / name
        d = dst / name
        if s.exists() and not d.exists():
            shutil.copy2(s, d)
            copied.append(name)
    if copied:
        print(f"   📋 Copied: {copied}")


# ── Verification ───────────────────────────────────────────────────────────────

def verify_structure(dst: Path, mode: str) -> bool:
    """
    Structural verification — no model load, no GPU required.

    Checks:
      - config.json has correct architectures / no vision_config
      - BNB mode: quantization_config present
      - No visual keys remain in safetensors shards
    """
    ok = True

    with open(dst / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    arch       = cfg.get("architectures", [])
    has_vision = "vision_config" in cfg
    has_bnb    = "quantization_config" in cfg

    arch_ok   = "Qwen3_5For" in str(arch)
    arch_mark = "✅" if arch_ok    else "❌"
    vis_mark  = "✅" if not has_vision else "❌"
    print(f"   {arch_mark} architectures: {arch}")
    print(f"   {vis_mark} vision_config absent")
    if mode == "bnb":
        bnb_mark = "✅" if has_bnb else "❌"
        print(f"   {bnb_mark} quantization_config present")

    if not arch_ok or has_vision:
        ok = False
    if mode == "bnb" and not has_bnb:
        ok = False

    visual_remaining = []
    for shard in sorted(dst.glob("*.safetensors")):
        with open(shard, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for k in header:
            if k != "__metadata__" and _should_drop(k):
                visual_remaining.append(k)

    if visual_remaining:
        print(f"   ❌ {len(visual_remaining)} visual keys still present in shards!")
        ok = False
    else:
        print("   ✅ No visual keys in safetensors shards")

    return ok


def verify_inference(dst: Path, mode: str) -> bool:
    """
    Full load + inference check with automatic device selection.

    Visual tower is already stripped from the output files, so
    _pick_device() is called with visual_already_dropped=True — it uses
    actual file size as the total (no visual subtraction needed).

    Strategy:
      cuda_direct / cuda_drop  →  CPU load → _move_to_cuda()
                                   (both strategies are identical post-strip:
                                    no visual to drop, just load and move)
      cpu                      →  stay on CPU

    _move_to_cuda() provides runtime OOM safety — if the estimate was off,
    inference falls back to CPU rather than crashing.
    No inference is ever skipped.
    """
    # visual_already_dropped=True: strip_visual removed visual from files,
    # so _pick_device should not subtract visual GB from the total.
    strategy = _pick_device(str(dst), visual_already_dropped=True)
    use_cuda = strategy in ("cuda_direct", "cuda_drop")

    try:
        if mode == "bnb":
            # Always CPU-first for BNB — avoids Params4bit._is_hf_initialized bug.
            model = AutoModelForCausalLM.from_pretrained(
                str(dst), device_map={"": "cpu"}, trust_remote_code=True,
            )
            if use_cuda:
                on_cuda = _move_to_cuda(model)  # OOM-safe; falls back to CPU
            else:
                on_cuda = False
        else:
            dtype_kw = dict(dtype=torch.bfloat16)
            if use_cuda:
                # Try loading directly to cuda:0; fall back to CPU on OOM.
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(dst), device_map={"": "cuda:0"},
                        trust_remote_code=True, **dtype_kw,
                    )
                    on_cuda = True
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("   ⚠️  OOM loading f16 to GPU — falling back to CPU")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(dst), device_map={"": "cpu"},
                        trust_remote_code=True, **dtype_kw,
                    )
                    on_cuda = False
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(dst), device_map={"": "cpu"},
                    trust_remote_code=True, **dtype_kw,
                )
                on_cuda = False

        tok = AutoTokenizer.from_pretrained(str(dst), trust_remote_code=True)
        device = "cuda" if on_cuda else "cpu"
        print(f"   ✅ Loaded as {type(model).__name__}  (device: {device})")
        print(f"   ✅ Tokenizer loaded  |  vocab size: {tok.vocab_size:,}")
        print_vram("after load")

        if mode == "bnb":
            ql, tl, vq, _ = count_quantized_layers(model)
            has_bad_visual = len(vq) > 0
            visual_warning = f"  ⚠️  {len(vq)} visual layers still quantized!" if has_bad_visual else ""
            print(f"   ✅ Quantized layers: {ql} / {tl} linear{visual_warning}")

    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        traceback.print_exc()
        return False

    try:
        prompt = "2+2="
        inp    = tok(prompt, return_tensors="pt")
        if on_cuda:
            inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        print(f"   ✅ Inference OK: {prompt!r} → {answer!r}")
        return True

    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        traceback.print_exc()
        return False

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_output(dst: Path, mode: str) -> bool:
    """Structural check + automatic device inference check."""
    print("\n── Verification ─────────────────────────────────────────")
    struct_ok = verify_structure(dst, mode)
    infer_ok  = verify_inference(dst, mode)
    return struct_ok and infer_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def strip_visual(
    model_ref: str,
    output_dir: str,
    mode: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Main pipeline: strip the visual tower from a Qwen3.5 model and verify.

    Steps:
      1. strip_weights()       — drop visual keys from safetensors shards
      2. copy_sidecars()       — copy tokenizer and metadata files
      3. patch_config()        — rebuild config.json as text-only
      4. patch_tokenizer_config() — strip image/video branches from chat template
      5. verify_output()       — structural check + inference check

    Source may be a local directory or a HF Hub repo id.
    HF repos are fetched via snapshot_download into the default HF cache;
    re-runs are instant — files are not re-downloaded if already cached.
    """
    is_remote, src_label = resolve_source(model_ref)

    print(f"\n{'═' * 60}")
    print(f"  strip_visual_qwen35.py  |  mode: {mode}")
    print(f"  {model_ref}  [{src_label}]")
    print(f"{'═' * 60}\n")

    size_gb = model_size_gb(model_ref)
    if size_gb > 0:
        where = "on Hub" if is_remote else "on disk"
        print(f"   📦 Model weight size {where}: {size_gb:.2f} GB\n")

    src, _ = ensure_local(model_ref, hf_token=hf_token)
    dst          = Path(output_dir)

    if dst.exists():
        print(f"❌ Output directory already exists: {dst}")
        print("   Remove it first or choose a different --output path.")
        sys.exit(1)

    dst.mkdir(parents=True)

    # ── 1. Strip visual weights ────────────────────────────────────────────────
    print("── 1. Strip visual weights ──────────────────────────────")
    try:
        removed_gb = strip_weights(src, dst)
        print(f"   🗑️  Removed (estimate): ~{removed_gb:.2f} GB")
    except Exception as e:
        print(f"   ❌ Weight stripping failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── 2. Copy side-car files ─────────────────────────────────────────────────
    print("\n── 2. Copy side-car files ───────────────────────────────")
    copy_sidecars(src, dst)

    # ── 3. Rebuild config.json ─────────────────────────────────────────────────
    print("\n── 3. Rebuild config.json ───────────────────────────────")
    try:
        patch_config(src, dst, mode)
    except ValueError as e:
        print(f"   ❌ {e}")
        sys.exit(1)

    # ── 4. Patch chat template ─────────────────────────────────────────────────
    print("\n── 4. Patch chat template ───────────────────────────────")
    patch_tokenizer_config(dst)
    patch_chat_template_jinja(src, dst)

    # ── 5. Verify ──────────────────────────────────────────────────────────────
    verify_output(dst, mode)

    # ── Cleanup — nothing to do, HF cache is managed by huggingface_hub ──────

    print(f"\n✅ Done  →  {dst}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for strip_visual_qwen35.py."""
    ap = argparse.ArgumentParser(
        description="Strip Qwen3.5 visual tower, export as text-only model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", required=True,
        help="Local model directory or HF Hub repo id.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Output directory (must not exist).",
    )
    ap.add_argument(
        "--mode", choices=["bnb", "f16"], default="bnb",
        help=(
            "bnb: packed BNB 4-bit input → text-only BNB 4-bit (for training). "
            "f16: bf16/f16 input → text-only bf16 (for merge / GGUF). "
            "Default: bnb"
        ),
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token for private repos. Omit to use credentials from `hf auth login`.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-strip CLI command."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    strip_visual(
        model_ref  = args.model,
        output_dir = args.output,
        mode       = args.mode,
        hf_token   = hf_token,
    )


if __name__ == "__main__":
    main()
