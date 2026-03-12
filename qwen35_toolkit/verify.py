"""
verify.py
───────────────────
Universal sanity-check for both BNB 4-bit and f16/bf16 models.

Mode is detected automatically from config.json:
  • quantization_config present  →  bnb  (BNB 4-bit checks)
  • no quantization_config       →  f16  (full-precision checks)

Accepts both a local directory and a HuggingFace Hub repo id:
  qwen35-verify --model ./Qwen3.5-text-0.8B-bnb-4bit   # BNB — auto-detected
  qwen35-verify --model ./Qwen3.5-text-0.8B            # f16 — auto-detected
  qwen35-verify --model unsloth/Qwen3-1.7B-bnb-4bit

  qwen35-verify --model ./Qwen3.5-text-0.8B-bnb-4bit
  qwen35-verify --model ./Qwen3.5-text-2B-bnb-4bit
  qwen35-verify --model ./Qwen3.5-text-4B-bnb-4bit
  qwen35-verify --model ./Qwen3.5-text-9B-bnb-4bit

Works with any architecture loadable via AutoModelForCausalLM / AutoTokenizer.
For Qwen3.5-specific checks (thinking mode, image inference, Qwen3_5ForConditionalGeneration)
use verify_qwen35.py instead, which imports and extends this module.

Checks performed (both modes):
  1. config.json  — BNB: validates quantization_config fields and skip_modules
                    f16: reports architectures / model_type / torch_dtype
  2. Load         — model loads without errors; device chosen automatically
                    (_pick_device → cuda_direct / cuda_drop / cpu;
                     _move_to_cuda catches runtime OOM, falls back to CPU)
  3. Precision    — BNB: all eligible linear layers are Linear4bit
                    f16: reports layer count and predominant weight dtype
  4. Image        — skipped (generic verifier; use verify_qwen35.py for
                    full visual pipeline test)
  5. Inference    — model generates a coherent response for each test task
"""

import argparse
import gc
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# ── Optional bitsandbytes ──────────────────────────────────────────────────────
# Imported at module level so the script works on machines without bitsandbytes
# (f16 models don't need it).  Linear4bit and bnb_F are re-exported for callers
# such as verify_qwen35.py that import them directly from this module.

try:
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.nn import Linear4bit
    _BNB_AVAILABLE = True
except ImportError:
    bnb_F = None               # type: ignore[assignment]
    Linear4bit = None          # type: ignore[assignment,misc]
    _BNB_AVAILABLE = False


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "./model"

# Each task is (prompt, max_new_tokens, label).
# max_new_tokens is a safety cap; the model should stop on its own via eos.
DEFAULT_TEST_TASKS: list[tuple[str, int, str]] = [
    ("What is 2 + 2? Answer in one word.",                                              256,  "Math"),
    ("Translate to French: 'The sky is blue.'",                                         512,  "Translation"),
    ("Write a Python function that returns the sum of two numbers.",                     1024, "Coding"),
    ("Summarize in one sentence: The sun is a star at the center of our solar system.",  512,  "Summarization"),
    ("What is the capital of France?",                                                   512,  "Factual QA"),
]

# Expected values in quantization_config — used to validate config.json.
EXPECTED_QUANT_CONFIG: dict = {
    "quant_method":              "bitsandbytes",
    "load_in_4bit":              True,
    "bnb_4bit_quant_type":       "nf4",
    "bnb_4bit_compute_dtype":    "bfloat16",
    "bnb_4bit_use_double_quant": True,
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    """Aggregated result of a verify() run."""
    model_path:   str
    passed:       bool
    disk_gb:      float = 0.0
    quant_layers: int   = 0
    total_layers: int   = 0
    vocab_size:   int   = 0
    task_results: list  = field(default_factory=list)


# ── Source detection ───────────────────────────────────────────────────────────

def is_hf_repo(model_path: str) -> bool:
    """
    Return True if model_path looks like a HuggingFace Hub repo id.

    A local path is identified by:
      • Being an existing directory on disk, or
      • Starting with a path prefix: ./ ../ / ~

    Everything else that contains a "/" is treated as "owner/model-name".

    Examples:
      "./Qwen3-1.7B-bnb-4bit"           → False  (local, starts with ./)
      "/data/models/qwen"               → False  (local, absolute)
      "unsloth/Qwen3-1.7B-bnb-4bit"     → True
      "techwithsergiu/Qwen3.5-0.8B"     → True
    """
    if os.path.isdir(model_path):
        return False
    if model_path.startswith(("./", "../", "/", "~")):
        return False
    return "/" in model_path


def resolve_source(model_path: str) -> tuple[bool, str]:
    """
    Return (is_remote, source_label) for display purposes.

    Args:
        model_path: Local path or HF repo id.

    Returns:
        (True, "HF Hub")  or  (False, "local")
    """
    remote = is_hf_repo(model_path)
    return remote, ("HF Hub" if remote else "local")


# ── Helpers ────────────────────────────────────────────────────────────────────


# Tensor name prefixes that belong to visual / multimodal components.
# Used to estimate visual tower size from safetensors metadata and to
# detect incorrectly quantized visual layers after load.
#
# Qwen3.5 structure (verified via inspect_safetensors.py):
#   model.language_model.*  — text backbone  (quantized)
#   model.visual.*          — visual tower   (bf16, skip)
#   lm_head.*               — output head    (bf16, skip)
#
# Prefixes are matched as substrings of the full tensor/module path,
# so "model.visual" catches both "model.visual.blocks.*" and
# top-level "visual.*" used by some other model families.
VISUAL_PREFIXES = (
    "model.visual.",      # Qwen3.5 / Qwen2.5-VL
    "visual.",            # some other VL models
    "vision_tower.",      # LLaVA-style
    "vision_model.",      # CLIP-style
    "multi_modal_projector.",
    "merger.",
    "modality_projection.",
)


def _is_visual_name(name: str) -> bool:
    """
    Return True if the module/tensor name belongs to the visual tower.

    Matches both "model.visual.blocks.0.attn" (starts with prefix) and
    ".visual.blocks.0.attn" inside a deeper path (contains ".prefix").
    The rstrip(".") normalises prefixes that end with a dot.
    """
    for prefix in VISUAL_PREFIXES:
        stem = prefix.rstrip(".")
        if name.startswith(stem) or f".{stem}" in name:
            return True
    return False


def _read_safetensors_sizes(model_path: str) -> dict[str, int]:
    """
    Read all safetensors headers and return {tensor_name: byte_size}.
    No weights are loaded — only the JSON header (a few KB per shard).
    Returns empty dict for HF repos or on any error.
    """
    if is_hf_repo(model_path):
        return {}
    sizes: dict[str, int] = {}
    try:
        for shard in sorted(Path(model_path).rglob("*.safetensors")):
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                offsets = meta.get("data_offsets")
                if offsets and len(offsets) == 2:
                    sizes[key] = offsets[1] - offsets[0]
    except Exception:
        pass
    return sizes


def _estimate_prefix_gb(sizes: dict[str, int], prefixes: tuple) -> float:
    """Sum bytes of tensors whose names start with any of the given prefixes."""
    return sum(v for k, v in sizes.items() if any(k.startswith(p) for p in prefixes)) / 1024**3


def _estimate_visual_gb(model_path: str) -> float:
    """Estimate visual tower size in GB from safetensors headers."""
    return _estimate_prefix_gb(_read_safetensors_sizes(model_path), VISUAL_PREFIXES)


def restore_visual_to_fp(model, model_path: str = "") -> int:
    """
    Find any Linear4bit layers inside visual-tower submodules and replace them
    with plain nn.Linear (bf16) loaded from the ORIGINAL safetensors weights.

    Problem: transformers 4.45+ re-quantizes ALL eligible nn.Linear on load,
    ignoring llm_int8_skip_modules. Visual layers end up as Linear4bit even
    though they were saved as bf16.

    Fix: instead of dequantizing (which introduces nf4 rounding error and
    corrupts visual encoder output), we reload the exact bf16 tensors from
    the safetensors files on disk. This gives pixel-perfect original weights.

    Falls back to dequantization if model_path is empty or files are missing.

    Returns:
        Number of layers restored.
    """
    if not _BNB_AVAILABLE:
        return 0

    # Build a lookup of original bf16 weights from safetensors (fast — only
    # reads tensors whose names match visual prefixes).
    orig_weights: dict[str, torch.Tensor] = {}
    if model_path and not is_hf_repo(model_path):
        try:
            for shard in sorted(Path(model_path).rglob("*.safetensors")):
                with open(shard, "rb") as f:
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(hlen))
                    data_start = 8 + hlen
                    for key, meta in header.items():
                        if key == "__metadata__":
                            continue
                        if not _is_visual_name(key) or not key.endswith(".weight"):
                            continue
                        offsets = meta.get("data_offsets")
                        dtype_str = meta.get("dtype", "")
                        if not offsets or dtype_str not in ("BF16", "F32", "F16"):
                            continue
                        f.seek(data_start + offsets[0])
                        raw = f.read(offsets[1] - offsets[0])
                        shape = meta["shape"]
                        np_dtype = {"BF16": "<u2", "F32": "<f4", "F16": "<f2"}[dtype_str]
                        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                        t = torch.from_numpy(arr.copy())
                        if dtype_str == "BF16":
                            t = t.view(torch.bfloat16)
                        else:
                            t = t.to(torch.bfloat16)
                        orig_weights[key] = t
        except Exception:
            orig_weights = {}

    from_disk = 0
    from_deq  = 0
    restored = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, Linear4bit):
            continue
        if not _is_visual_name(name):
            continue

        target_device = module.weight.device

        weight_key = f"{name}.weight"
        if weight_key in orig_weights:
            weight_fp = orig_weights[weight_key].to(target_device)
            from_disk += 1
        else:
            # Fallback: dequantize (introduces nf4 rounding error).
            try:
                weight_fp = bnb_F.dequantize_4bit(
                    module.weight.data,
                    module.weight.quant_state,
                    quant_type="nf4",
                ).to(torch.bfloat16)
            except Exception:
                weight_fp = module.weight.data.to(torch.bfloat16)
            from_deq += 1

        new_layer = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
            device=target_device,
        )
        new_layer.weight = torch.nn.Parameter(weight_fp, requires_grad=False)
        if module.bias is not None:
            new_layer.bias = torch.nn.Parameter(
                module.bias.data.to(torch.bfloat16), requires_grad=False
            )

        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)
        restored += 1

    if restored:
        src = []
        if from_disk:
            src.append(f"{from_disk} from disk (exact bf16)")
        if from_deq:
            src.append(f"{from_deq} via dequantize (fallback)")
        print(f"   🔧 Restored {restored} visual layer(s): {', '.join(src)}")
        gc.collect()
        torch.cuda.empty_cache()
    return restored


def drop_visual_tower(model) -> float:
    """
    Delete visual tower submodules from a loaded model and free their memory.

    Searches for common visual tower attribute names and removes them in-place.
    Returns the approximate VRAM freed in GB (0.0 for text-only models).

    Call this after from_pretrained() when the
    visual tower was offloaded to CPU — deleting it frees CPU RAM and keeps
    the model object clean for text-only inference.
    """
    visual_attrs = ["visual", "vision_tower", "vision_model",
                    "multi_modal_projector", "merger"]
    freed_params = 0
    removed      = []
    for attr in visual_attrs:
        mod = getattr(model, attr, None)
        if mod is None:
            # try model.model.<attr>
            inner = getattr(model, "model", None)
            if inner is not None:
                mod = getattr(inner, attr, None)
                if mod is not None:
                    delattr(inner, attr)
                    removed.append(f"model.{attr}")
        else:
            delattr(model, attr)
            removed.append(attr)
        if mod is not None:
            for p in mod.parameters():
                freed_params += p.numel() * p.element_size()
            del mod

    gc.collect()
    torch.cuda.empty_cache()

    freed_gb = freed_params / 1024**3
    if removed:
        print(f"   🗑️  Removed visual modules: {removed}  (~{freed_gb:.2f} GB freed)")
    return freed_gb


def print_vram(label: str = "") -> None:
    """Print current VRAM usage. No-op when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    used  = torch.cuda.memory_allocated(0) / 1024**3
    tag   = f" [{label}]" if label else ""
    print(f"   VRAM : {used:.1f} GB used / {total:.1f} GB total{tag}")


def model_size_gb(model_path: str) -> float:
    """
    Return total size of model weight files in GB.

    For local paths: sums .safetensors / .bin shards on disk.
    For HF repos:    queries the Hub API for file sizes (no download).
    Returns 0.0 if the size cannot be determined.
    """
    if not is_hf_repo(model_path):
        total = sum(
            f.stat().st_size
            for f in Path(model_path).rglob("*")
            if f.is_file() and f.suffix in (".safetensors", ".bin")
        )
        return total / 1024**3

    # Remote repo — query Hub API.
    try:
        api   = HfApi()
        files = list(api.list_repo_tree(model_path, recursive=True))
        total = sum(
            entry.size
            for entry in files
            if hasattr(entry, "size")
            and entry.size is not None
            and Path(entry.rfilename).suffix in (".safetensors", ".bin")
        )
        return total / 1024**3
    except Exception:
        return 0.0


# ── Individual checks ──────────────────────────────────────────────────────────

def detect_mode(model_path: str) -> str:
    """
    Detect whether the model is BNB 4-bit or f16/bf16 from config.json.

    Returns "bnb" if quantization_config is present in config.json,
    "f16" otherwise.  Works for both local paths and HF Hub repo ids.
    """
    try:
        if is_hf_repo(model_path):
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        else:
            config_file = str(Path(model_path) / "config.json")
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        return "bnb" if cfg.get("quantization_config") else "f16"
    except Exception:
        return "f16"   # safe default — f16 checks never crash on BNB models


def detect_arch(model_path: str) -> str:
    """
    Detect whether the model has a visual tower from config.json.

    Returns "vlm"  if architectures contains a conditional-generation class
            (e.g. Qwen3_5ForConditionalGeneration, LlavaForConditionalGeneration).
    Returns "text" otherwise (CausalLM, text-only models).
    Works for both local paths and HF Hub repo ids.
    """
    try:
        if is_hf_repo(model_path):
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        else:
            config_file = str(Path(model_path) / "config.json")
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if any("ForConditionalGeneration" in a for a in archs):
            return "vlm"
        return "text"
    except Exception:
        return "text"   # safe default — text checks never crash on VLM models


def check_config(model_path: str, mode: Optional[str] = None) -> bool:
    """
    Verify config.json.  Behaviour depends on mode:

    BNB mode: checks that quantization_config is present and all expected
              fields (quant_method, load_in_4bit, …) match EXPECTED_QUANT_CONFIG.
    f16 mode: confirms config.json is readable and reports architectures /
              model_type / torch_dtype — no quantization fields required.

    mode=None → auto-detected via detect_mode().

    For local paths: reads the file directly from disk.
    For HF repos:    downloads config.json via hf_hub_download (cached).

    Returns True on success, False if a required check fails.
    """
    if mode is None:
        mode = detect_mode(model_path)

    try:
        if is_hf_repo(model_path):
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        else:
            config_file = str(Path(model_path) / "config.json")
            if not Path(config_file).exists():
                print("   ❌ config.json not found")
                return False
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"   ❌ Could not fetch config.json: {e}")
        return False

    if mode == "bnb":
        qcfg = cfg.get("quantization_config")
        if not qcfg:
            print("   ❌ quantization_config missing from config.json")
            return False
        issues = [
            f"{k}={qcfg.get(k)!r} (expected {v!r})"
            for k, v in EXPECTED_QUANT_CONFIG.items()
            if qcfg.get(k) != v
        ]
        if issues:
            for issue in issues:
                print(f"   ⚠️  {issue}")
            return False
        skip = qcfg.get("llm_int8_skip_modules", [])
        print(f"   ✅ config.json OK  |  skip_modules: {skip}")
    else:
        # f16 mode — report key fields, no quantization checks
        arch  = cfg.get("architectures", ["unknown"])
        mtype = cfg.get("model_type", "unknown")
        dtype = cfg.get("torch_dtype", "unknown")
        print(f"   ✅ config.json OK  |  arch: {arch}  model_type: {mtype}  torch_dtype: {dtype}")

    return True


def count_quantized_layers(
    model: PreTrainedModel,
) -> tuple[int, int, list[str], list[str]]:
    """
    Count Linear4bit layers vs total nn.Linear layers in the model.
    Also collect the full names of any visual-tower layers that are
    incorrectly quantized — they must stay at full precision.

    Safe to call on f16 models: if bitsandbytes is not available or no
    Linear4bit layers exist, quantized_count will be 0.

    Returns:
        (quantized_count, total_linear_count,
         visual_quantized_names, visual_fp_names)
    """
    total = quantized = 0
    visual_quantized: list[str] = []
    visual_fp:        list[str] = []

    for name, module in model.named_modules():
        is_visual = _is_visual_name(name)
        if isinstance(module, torch.nn.Linear):
            total += 1
        if Linear4bit is not None and isinstance(module, Linear4bit):
            quantized += 1
            if is_visual:
                visual_quantized.append(name)
        elif isinstance(module, torch.nn.Linear) and is_visual:
            visual_fp.append(name)

    return quantized, total, visual_quantized, visual_fp



def _pick_device(model_path: str, visual_already_dropped: bool = False) -> str:
    """
    Choose the best loading strategy based on model size vs VRAM.

    Returns one of three strategies:
      "cuda_direct"  — full model fits in VRAM → load to CPU → model.cuda()
                       image test runs on GPU (fast)
      "cuda_drop"    — full model doesn't fit, but backbone does →
                       load to CPU → image test on CPU → drop visual → cuda()
      "cpu"          — backbone doesn't fit or no CUDA →
                       stay on CPU for all inference

    visual_already_dropped: set True when the visual tower has already been
    stripped from the files (e.g. strip_visual output) — vis_gb treated as 0.

    Size accounting:
      total_gb  = actual file size (includes backbone + visual + lm_head +
                  embed_tokens — everything that will be loaded)
      vis_gb    = visual tower only (estimated from key prefixes)
      rest_gb   = total - visual (backbone + lm_head + embeddings)
                  This is what ends up in VRAM after drop_visual_tower().

    Threshold: 95% VRAM for all strategies.
    _move_to_cuda() catches any OOM at runtime and falls back to CPU.
      cuda_direct : total  ≤ 95% VRAM
      cuda_drop   : rest   ≤ 95% VRAM  (visual dropped before cuda())
      cpu         : rest   > 95% VRAM or no CUDA
    """
    if not torch.cuda.is_available():
        return "cpu"

    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    sizes    = _read_safetensors_sizes(model_path)
    if visual_already_dropped:
        vis_gb = 0.0
    else:
        vis_gb = _estimate_prefix_gb(sizes, VISUAL_PREFIXES)
    total_gb = model_size_gb(model_path)           # actual on-disk size
    rest_gb  = total_gb - vis_gb                   # what stays after visual drop

    # 95% threshold for all strategies — _move_to_cuda() catches any
    # remaining OOM at runtime and falls back to CPU gracefully.
    if total_gb <= vram_gb * 0.95:
        strategy = "cuda_direct"
    elif rest_gb <= vram_gb * 0.95:
        strategy = "cuda_drop"
    else:
        strategy = "cpu"

    label = {
        "cuda_direct": "full model fits → cuda:0  (image test on GPU)",
        "cuda_drop":   f"drop visual ({vis_gb:.1f} GB) → cuda:0  (image test on CPU)",
        "cpu":         f"CPU  (rest {rest_gb:.1f} GB > VRAM or no CUDA)",
    }[strategy]
    size_detail = ""
    if vis_gb > 0:
        size_detail = f"  (visual: {vis_gb:.1f} GB  rest: {rest_gb:.1f} GB)"
    print(f"   📋 total: {total_gb:.1f} GB{size_detail}  VRAM: {vram_gb:.1f} GB  →  {label}")
    return strategy


def _move_to_cuda(model) -> bool:
    """
    Attempt model.cuda(). Returns True on success, False on OOM.
    Falls back gracefully — caller keeps model on CPU and continues.
    """
    try:
        model.cuda()
        return True
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("   ⚠️  OOM during model.cuda() — falling back to CPU inference")
        return False


def load_model_and_tokenizer(
    model_path: str,
    mode: Optional[str] = None,
) -> tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizerBase]]:
    """
    Load model via AutoModelForCausalLM and tokenizer via AutoTokenizer.

    Accepts both local paths and HF Hub repo ids — from_pretrained handles both.

    mode=None → auto-detected via detect_mode().

    BNB mode: always loads to CPU first — avoids two BNB CUDA issues:
      - device_map="auto" → Params4bit._is_hf_initialized bug
      - explicit BitsAndBytesConfig → uint8 vs bf16 size mismatch
      After load: restore_visual_to_fp(), then move to device per strategy.

    f16 mode: loads with torch_dtype=bfloat16 directly to the target device
      (no CPU-first detour needed — no BNB quantization happens on load).

    Device strategy chosen by _pick_device() for both modes:
      cuda_direct  →  full model fits → move/load to GPU
      cuda_drop    →  only backbone fits → drop visual tower → GPU
      cpu          →  too large or no CUDA → stay on CPU

    _move_to_cuda() is the runtime safety net: catches OutOfMemoryError and
    falls back to CPU if the size estimate was off.  Inference always runs.

    Returns (model, tokenizer) or (None, None) on failure.
    """
    if mode is None:
        mode = detect_mode(model_path)
    strategy = _pick_device(model_path)

    try:
        if mode == "bnb":
            # BNB: CPU-first mandatory to avoid BNB init bugs.
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map={"": "cpu"}, trust_remote_code=True,
            )
            # Restore visual layers that transformers may have re-quantized on load.
            restore_visual_to_fp(model, model_path)
            # Count now: visual=bf16, text=Linear4bit — accurate snapshot.
            _ql, _tl, _vq, _vf = count_quantized_layers(model)
            model._pre_drop_counts = (_ql, _tl, _vq, _vf)
            if strategy == "cuda_drop":
                drop_visual_tower(model)
                on_cuda = _move_to_cuda(model)
            elif strategy == "cuda_direct":
                on_cuda = _move_to_cuda(model)
            else:
                on_cuda = False
        else:
            # f16: load directly to target device with bfloat16.
            if strategy in ("cuda_direct", "cuda_drop"):
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, device_map={"": "cuda:0"},
                        dtype=torch.bfloat16, trust_remote_code=True,
                    )
                    on_cuda = True
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("   ⚠️  OOM loading f16 to GPU — falling back to CPU")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, device_map={"": "cpu"},
                        dtype=torch.bfloat16, trust_remote_code=True,
                    )
                    on_cuda = False
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map={"": "cpu"},
                    dtype=torch.bfloat16, trust_remote_code=True,
                )
                on_cuda = False

        actual = "cuda" if on_cuda else "cpu"
        print(f"   ✅ Model loaded  |  mode: {mode}  strategy: {strategy}  device: {actual}")
        print_vram("after load")
    except Exception as e:
        print(f"   ❌ Model load failed: {e}")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"   ✅ Tokenizer loaded  |  vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"   ❌ Tokenizer load failed: {e}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

    return model, tokenizer


# ── Inference runner ───────────────────────────────────────────────────────────

def run_inference_tasks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tasks: list[tuple[str, int, str]],
    chat_template_kwargs: Optional[dict] = None,
) -> tuple[bool, list[dict]]:
    """
    Run a list of inference tasks and print per-task results.

    Args:
        model:                 Loaded model (CPU or GPU — inputs are sent to
                               the device of the first model parameter).
        tokenizer:             Loaded tokenizer.
        tasks:                 List of (prompt, max_new_tokens, label).
        chat_template_kwargs:  Extra kwargs forwarded to apply_chat_template
                               (e.g. {"enable_thinking": True} for Qwen3.5).

    Returns:
        (all_passed, results_list)
    """
    if chat_template_kwargs is None:
        chat_template_kwargs = {}

    all_passed   = True
    results      = []
    total_tokens = 0
    total_time   = 0.0

    for prompt, max_tokens, task_name in tasks:
        try:
            messages  = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            # Send inputs to the device of the first model parameter.
            # With device_map="auto" the model may span GPU+CPU, so
            # hardcoding "cuda:0" would fail for CPU-only layers.
            input_device = next(model.parameters()).device
            inputs    = tokenizer(formatted, return_tensors="pt").to(input_device)
            input_len = inputs["input_ids"].shape[-1]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    # Explicitly set to None so transformers ignores whatever
                    # temperature/top_p/top_k are saved in generation_config.json.
                    # Without this, a saved config with sampling params causes a
                    # "flags not valid" warning when do_sample=False.
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            new_ids  = output_ids[0][input_len:]
            n_tokens = len(new_ids)
            if elapsed > 0:
                tok_per_sec = n_tokens / elapsed
            else:
                tok_per_sec = 0.0
            total_tokens += n_tokens
            total_time   += elapsed

            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            stopped  = "eos" if n_tokens < max_tokens else f"cap ({max_tokens})"

            print(f"   ✅ [{task_name}]")
            print(f"      prompt   : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            print(f"      response : {response!r}")
            print(f"      tokens   : {n_tokens} generated  |  stopped by: {stopped}")
            print(f"      time     : {elapsed:.2f}s  |  speed: {tok_per_sec:.1f} tok/s")

            results.append({
                "task": task_name, "ok": True,
                "tokens": n_tokens, "time": elapsed, "speed": tok_per_sec,
            })
        except Exception as e:
            print(f"   ❌ [{task_name}] failed: {e}")
            results.append({"task": task_name, "ok": False})
            all_passed = False

    if total_time > 0:
        avg_speed = total_tokens / total_time
    else:
        avg_speed = 0.0
    print(f"\n   ── Summary {'─' * 49}")
    print(f"      total tokens : {total_tokens}")
    print(f"      total time   : {total_time:.2f}s")
    print(f"      avg speed    : {avg_speed:.1f} tok/s")

    return all_passed, results


# ── Main verify ────────────────────────────────────────────────────────────────

def verify(
    model_path: str,
    tasks: Optional[list[tuple[str, int, str]]] = None,
    mode: Optional[str] = None,
) -> VerifyResult:
    """
    Full verification pipeline — works for both BNB 4-bit and f16/bf16 models.

    Accepts both local paths and HF Hub repo ids.
    mode=None → auto-detected via detect_mode() (reads config.json).
    Loading strategy is chosen automatically by _pick_device() based on
    actual file size vs VRAM.  _move_to_cuda() provides a runtime OOM guard.

    Args:
        model_path: Local directory path or HF Hub repo id.
        tasks:      Override the default test tasks.
        mode:       "bnb", "f16", or None (auto-detect).

    Returns:
        VerifyResult with pass/fail status and collected metrics.
    """
    if tasks is None:
        tasks = DEFAULT_TEST_TASKS
    if mode is None:
        mode = detect_mode(model_path)

    result          = VerifyResult(model_path=model_path, passed=False)
    is_remote, src  = resolve_source(model_path)

    print(f"\n{'═' * 60}")
    print(f"  {model_path}  [{src} · {mode}]")
    print(f"{'═' * 60}")

    if not is_remote and not os.path.isdir(model_path):
        print("   ❌ Directory not found — skipping.")
        return result

    size_gb = model_size_gb(model_path)
    if size_gb > 0:
        label = "on Hub" if is_remote else "on disk"
        print(f"   📦 Model weight size {label}: {size_gb:.2f} GB")

    # ── 1. Config ──────────────────────────────────────────────────────────────
    print("\n── 1. Config ────────────────────────────────────────────")
    if not check_config(model_path, mode=mode):
        return result

    # ── 2. Load model + tokenizer ────────────────────────────────────────────
    print("\n── 2. Load ──────────────────────────────────────────────")
    model, tokenizer = load_model_and_tokenizer(model_path, mode=mode)
    if model is None:
        return result

    result.vocab_size  = tokenizer.vocab_size
    result.disk_gb     = size_gb

    # ── 3. Precision check ─────────────────────────────────────────────────────
    print("\n── 3. Precision ─────────────────────────────────────────")
    if mode == "bnb":
        if hasattr(model, "_pre_drop_counts"):
            (result.quant_layers, result.total_layers,
             vis_quant, vis_fp) = model._pre_drop_counts
            del model._pre_drop_counts
        else:
            (result.quant_layers, result.total_layers,
             vis_quant, vis_fp) = count_quantized_layers(model)
        if result.total_layers > 0:
            pct = result.quant_layers / result.total_layers * 100
        else:
            pct = 0

        if result.quant_layers == 0:
            print("   ❌ No Linear4bit layers found — model may not be quantized!")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return result
        print(f"   ✅ {result.quant_layers}/{result.total_layers} linear layers quantized ({pct:.0f}%)")

        # Visual tower sanity check — visual must stay at full precision.
        # After drop_visual_tower() layers are absent from named_modules();
        # filter to only live layers so absent ones aren't flagged as errors.
        _live = dict(model.named_modules())
        vis_quant_live = [n for n in vis_quant if n in _live]
        vis_fp_live    = [n for n in vis_fp    if n in _live]
        vis_dropped    = len(vis_fp) + len(vis_quant) - len(vis_quant_live) - len(vis_fp_live)

        if vis_fp_live:
            print(f"   ✅ Visual tower: {len(vis_fp_live)} layer(s) correctly at full precision")
        if vis_dropped and not vis_fp_live and not vis_quant_live:
            print(f"   ✅ Visual tower: removed ({vis_dropped} layer(s) dropped)")
        if vis_quant_live:
            sample_name = vis_quant_live[0]
            sample_type = type(_live[sample_name]).__name__
            print(f"   ❌ Visual tower: {len(vis_quant_live)} layer(s) are INCORRECTLY quantized!")
            print(f"      Sample: {sample_name}  →  {sample_type}")
            print("      This will cause AssertionError during image inference.")
            print("      Reconvert with: python convert_qwen35_to_bnb4bit.py --verbose")
            print("      and check that visual layers appear as '└── skipped'.")
            print("      Affected layers (first 5):")
            for ln in vis_quant_live[:5]:
                print(f"         {ln}")
            if len(vis_quant_live) > 5:
                print(f"         … and {len(vis_quant_live) - 5} more")
            result.passed = False
        elif not vis_fp_live and not vis_dropped:
            print("   ℹ️  No visual tower detected (text-only model)")

    else:
        # f16 mode — report total linear layers and actual weight dtype.
        total = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
        # Find the dtype of the first non-float32 parameter (prefer bf16/f16 over
        # fallback float32 that some embedding layers may have).
        sample_dtype = None
        for p in model.parameters():
            if p.dtype != torch.float32:
                sample_dtype = p.dtype
                break
        if sample_dtype is None:
            for p in model.parameters():
                sample_dtype = p.dtype
                break
        if sample_dtype:
            dtype_str = str(sample_dtype).replace("torch.", "")
        else:
            dtype_str = "unknown"
        result.total_layers = total
        print(f"   ✅ {total} linear layers  |  weight dtype: {dtype_str}  (f16 mode — no quantization expected)")
        # Visual tower report — same language as BNB mode for consistency.
        has_visual = any(
            _is_visual_name(name)
            for name, _ in model.named_modules()
        )
        if has_visual:
            print(f"   ✅ Visual tower: present at {dtype_str}")
        else:
            print("   ℹ️  No visual tower detected (text-only model)")

    # ── 4. Image inference ────────────────────────────────────────────────────
    print("\n── 4. Image ─────────────────────────────────────────────")
    print("   ℹ️  skipped (use verify_qwen35.py for VLM)")

    # ── 5. Inference ───────────────────────────────────────────────────────────
    print("\n── 5. Inference ─────────────────────────────────────────")
    all_passed, result.task_results = run_inference_tasks(model, tokenizer, tasks)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result.passed = all_passed
    if all_passed:
        print("\n   ✅ ALL CHECKS PASSED\n")
    else:
        print("\n   ⚠️  SOME CHECKS FAILED — see above\n")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for verify_model.py."""
    ap = argparse.ArgumentParser(
        description="Sanity-check a BNB 4-bit quantized model (generic, any architecture).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Local model directory or HF Hub repo id (e.g. techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit).",
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token for private repos. Omit to use credentials from `hf auth login`.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the verify CLI command."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    verify(model_path=args.model)


if __name__ == "__main__":
    main()
