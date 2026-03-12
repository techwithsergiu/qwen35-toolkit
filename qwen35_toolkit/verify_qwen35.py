"""
verify_qwen35.py
─────────────────────
Qwen3.5-specific sanity-check for BNB 4-bit and f16/bf16 models.
Mode is auto-detected from config.json — no flags needed.

Accepts both a local directory and a HuggingFace Hub repo id:
  qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit               # vlm  + bnb
  qwen35-verify-qwen35 --model unsloth/Qwen3.5-0.8B                  # vlm  + f16
  qwen35-verify-qwen35 --model ./Qwen3.5-text-0.8B-bnb-4bit          # text + bnb
  qwen35-verify-qwen35 --model ./Qwen3.5-text-0.8B                   # text + f16
  qwen35-verify-qwen35 --model techwithsergiu/Qwen3.5-0.8B-bnb-4bit  # HF Hub

  qwen35-verify-qwen35 --model ./Qwen3.5-0.8B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-2B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-4B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-9B-bnb-4bit

  qwen35-verify-qwen35 --model ./Qwen3.5-text-0.8B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-text-2B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-text-4B-bnb-4bit
  qwen35-verify-qwen35 --model ./Qwen3.5-text-9B-bnb-4bit

Extends qwen35_toolkit.verify.py with:
  • Architecture-aware loader — two independent axes:
      arch: vlm  → Qwen3_5ForConditionalGeneration (visual tower present)
      arch: text → AutoModelForCausalLM            (text-only after strip)
      mode: bnb  → BNB NF4 quantized
      mode: f16  → full bf16 precision
    giving four combinations: vlm+bnb  vlm+f16  text+bnb  text+f16
  • Inference in both THINKING OFF and THINKING ON modes
    (thinking mode prefixes the answer with a <think>…</think> block;
     skip_special_tokens=True strips it from the decoded output)
  • Image inference test (step 4, VLM only) — verifies the full visual
    pipeline end-to-end before the visual tower is dropped

Device strategy (see _pick_device in qwen35_toolkit.verify.py):
  cuda_direct  →  full model fits in VRAM → move to GPU → image test on GPU
  cuda_drop    →  only backbone fits → image test on CPU → drop visual → cuda()
  cpu          →  backbone too large or no CUDA → image test on CPU, stay on CPU

In all cases _move_to_cuda() is used as a runtime OOM guard — falls back to
CPU if the file-size estimate was optimistic.

Checks performed:
  1. config.json    — BNB: quantization_config; f16: architectures / torch_dtype
  2. Load           — model loads; device chosen automatically
                      arch+mode → vlm+bnb / vlm+f16 / text+bnb / text+f16
  3. Precision      — BNB: counts Linear4bit + visual tower dtype
                      f16: weight dtype + linear layer count
  4. Image          — VLM only: visual pipeline e2e; skipped for text-only
  5a. Inference     — THINKING OFF  (all modes)
  5b. Inference     — THINKING ON   (all modes)
"""

import argparse
import base64
import gc
import io
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    PILImage = None       # type: ignore[assignment,misc]
    _PIL_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision_info
    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    _qwen_process_vision_info = None   # type: ignore[assignment]
    _QWEN_VL_UTILS_AVAILABLE = False

# Re-use all generic infrastructure from the base module.
from qwen35_toolkit.verify import (
    DEFAULT_TEST_TASKS,
    VerifyResult,
    _pick_device,
    _move_to_cuda,
    check_config,
    detect_mode,
    detect_arch,
    drop_visual_tower,
    restore_visual_to_fp,
    count_quantized_layers,
    model_size_gb,
    print_vram,
    resolve_source,
    run_inference_tasks,
)  # load_model_and_tokenizer not imported — Qwen3.5 uses its own loader


# ── Defaults ───────────────────────────────────────────────────────────────────

# Available sizes: 0.8B  2B  4B  9B — local or HF Hub repo id
DEFAULT_MODEL = "./Qwen3.5-0.8B-bnb-4bit"


# ── Visual tower helpers ───────────────────────────────────────────────────────

def _find_visual_module(model):
    """Return the visual submodule if present anywhere in the tree, else None."""
    visual_names = {"visual", "vision_tower", "vision_model"}
    for name, mod in model.named_modules():
        if name.split(".")[-1] in visual_names:
            return mod
    return None


def _has_visual(model) -> bool:
    """Return True if the model still has a visual tower submodule."""
    return _find_visual_module(model) is not None


# ── Qwen3.5 loader ────────────────────────────────────────────────────────────

def load_qwen35(model_path: str) -> tuple:
    """
    Load a Qwen3.5 model. Both arch and mode are auto-detected from config.json.

    arch=vlm  → Qwen3_5ForConditionalGeneration (visual tower present)
    arch=text → AutoModelForCausalLM (text-only; Qwen3_5ForCausalLM arch)
    mode=bnb  → BNB NF4; restore_visual_to_fp() applied for vlm+bnb
    mode=f16  → full bf16; loaded with torch_dtype="auto"

    Always loads to CPU first (same BNB reasons as load_model_and_tokenizer).
    Device strategy chosen by _pick_device():

      cuda_direct  →  _move_to_cuda() (full model, visual included)
                      image test runs on GPU in step 4 (fast)
                      drop_visual_tower() + free VRAM done in step 4

      cuda_drop    →  model stays on CPU
                      image test runs on CPU in step 4
                      drop_visual_tower() + _move_to_cuda() done in step 4

      cpu          →  model stays on CPU throughout
                      image test runs on CPU in step 4

    _move_to_cuda() catches OutOfMemoryError and falls back to CPU in all
    branches — inference always runs regardless.

    Returns (model, tokenizer) or (None, None) on failure.
    """
    mode     = detect_mode(model_path)
    arch     = detect_arch(model_path)
    strategy = _pick_device(model_path)

    try:
        if arch == "vlm":
            # VLM (both bnb and f16): use Qwen3_5ForConditionalGeneration.
            # Required for visual tower and enable_thinking to work correctly.
            load_kwargs = {"device_map": {"": "cpu"}, "trust_remote_code": True}
            if mode == "f16":
                load_kwargs["torch_dtype"] = "auto"
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs,
            )
            if mode == "bnb":
                # Restore visual layers that transformers may have re-quantized.
                restore_visual_to_fp(model, model_path)
        else:
            # text-only (both bnb and f16): use AutoModelForCausalLM.
            # Qwen3_5ForConditionalGeneration fails on Qwen3_5ForCausalLM arch.
            load_kwargs = {"device_map": {"": "cpu"}, "trust_remote_code": True}
            if mode == "f16":
                load_kwargs["torch_dtype"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs,
            )
        # Count now: visual=bf16, text=Linear4bit — accurate snapshot before drop.
        _ql, _tl, _vq, _vf = count_quantized_layers(model)
        model._pre_drop_counts = (_ql, _tl, _vq, _vf)
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

    # For cuda_direct: move to GPU now so image test (step 4) runs on GPU.
    # cuda_drop and cpu stay on CPU — step 4 handles the move after drop.
    if strategy == "cuda_direct":
        _move_to_cuda(model)

    model._strategy = strategy   # read in steps 3 and 4
    model._mode     = mode        # read in steps 3 and 4
    model._arch     = arch        # read in step 4
    print(f"   ✅ Model loaded  (Qwen3_5ForConditionalGeneration)  |  strategy: {strategy}")
    print_vram("after load")
    return model, tokenizer


# ── Main verify ────────────────────────────────────────────────────────────────

# Small solid-colour test images encoded as base64 PNG (generated inline —
# no file I/O, no network, works anywhere).
# Red 64×64 PNG (minimal valid PNG, single IDAT chunk).


# 448×224 image: left half red, right half blue.
# Expected answer: "red".
_DIGIT_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAcAAAADgCAIAAAAbnCGmAAADCklEQVR4nO3UoRGAQAADQaAWKvhKqJwKvhg8AnEGmNmVUVG3zjEW+Jj9PG/LOOYrT+DB9vYBgL8SUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABIgEFiAQUIBJQgEhAASIBBYgEFCASUIBIQAEiAQWIBBQgElCASEABogszxQYeXUThrAAAAABJRU5ErkJggg=="
_DIGIT_EXPECTED = "red"


def _run_image_inference(model, tokenizer) -> bool:
    """
    Run a simple image captioning task to verify the visual tower works.

    Uses a minimal base64-encoded PNG so no network access or file I/O is
    needed. Checks that the model produces a non-empty response — we cannot
    verify the semantic content without a real image, but a valid response
    confirms the visual pipeline (image tokenization → visual encoder →
    merger → language model) executed without crashing.

    Returns True if the test passed, False otherwise.
    """
    if not _PIL_AVAILABLE:
        print("   ⚠️  PIL not installed — skipping image test (pip install pillow)")
        return True  # not a model problem

    try:
        img_bytes = base64.b64decode(_DIGIT_PNG_B64)
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        # Keep full 224×224 — do NOT resize, processor handles scaling.

        # Load processor — required for image tokenization.
        # Use the public name_or_path property instead of the private _name_or_path.
        model_path_for_proc = model.config.name_or_path or tokenizer.name_or_path
        try:
            processor = AutoProcessor.from_pretrained(
                model_path_for_proc, trust_remote_code=True
            )
        except Exception as proc_err:
            print(f"   ❌ [Image] AutoProcessor failed: {proc_err}")
            print("      Cannot run image test without processor.")
            return False

        # Pass image as data URI — process_vision_info requires a URL/path/
        # data URI, not a raw PIL object. Raw PIL is silently ignored by some
        # versions of qwen_vl_utils causing the model to see no image at all.
        data_uri = f"data:image/png;base64,{_DIGIT_PNG_B64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data_uri},
                    {"type": "text",  "text": "What color is the left half of this image? Answer with one word."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        if _QWEN_VL_UTILS_AVAILABLE:
            image_inputs, video_inputs = _qwen_process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=[text], images=[image], return_tensors="pt")

        # Move inputs to model device and cast floating-point tensors to model dtype.
        non_meta = next(p for p in model.parameters() if p.device.type != "meta")
        device, model_dtype = non_meta.device, non_meta.dtype
        moved_inputs = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                continue
            if v.is_floating_point():
                moved_inputs[k] = v.to(device=device, dtype=model_dtype)
            else:
                moved_inputs[k] = v.to(device=device)
        inputs = moved_inputs

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        n_input = inputs["input_ids"].shape[1]
        response = tokenizer.decode(out[0][n_input:], skip_special_tokens=True).strip()

        if not response:
            print("   ❌ [Image] empty response — visual pipeline may be broken")
            return False

        passed = _DIGIT_EXPECTED in response.lower()
        mark = "✅" if passed else "❌"
        print(f"   {mark} [Image] response: {response!r}  "
              f"(expected: {_DIGIT_EXPECTED!r}, case-insensitive)")
        return passed

    except Exception as e:
        print(f"   ❌ [Image] failed: {e}")
        return False



def verify_qwen35(model_path: str) -> VerifyResult:
    """
    Full verification pipeline for a Qwen3.5 model (BNB or f16, VLM or text-only).
    Mode is auto-detected from config.json — no flags needed.

    Accepts both local paths and HF Hub repo ids.
    Loading strategy is chosen automatically by _pick_device() based on
    actual file size vs VRAM.  _move_to_cuda() provides a runtime OOM guard.
    See load_qwen35() for the full per-strategy flow.

    Runs inference in up to three passes:
      4.  IMAGE        — VLM only: visual pipeline e2e; skipped for text-only
      5a. THINKING OFF — standard chat response (all model types)
      5b. THINKING ON  — model emits a <think>…</think> chain-of-thought block

    Returns:
        VerifyResult with pass/fail status and collected metrics.
    """
    result          = VerifyResult(model_path=model_path, passed=False)
    is_remote, src  = resolve_source(model_path)

    mode = detect_mode(model_path)
    arch = detect_arch(model_path)

    print(f"\n{'═' * 60}")
    print(f"  {model_path}  [Qwen3.5 · {src} · {arch} · {mode}]")
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
    if not check_config(model_path):
        return result

    # ── 2. Load model + tokenizer ────────────────────────────────────────────
    print("\n── 2. Load ──────────────────────────────────────────────")
    model, tokenizer = load_qwen35(model_path)
    if model is None:
        return result

    result.vocab_size = tokenizer.vocab_size
    result.disk_gb    = size_gb

    # ── 3. Precision ──────────────────────────────────────────────────────────────
    print("\n── 3. Precision ─────────────────────────────────────────")
    if hasattr(model, "_pre_drop_counts"):
        (result.quant_layers, result.total_layers,
         vis_quant, vis_fp) = model._pre_drop_counts
        del model._pre_drop_counts
    else:
        (result.quant_layers, result.total_layers,
         vis_quant, vis_fp) = count_quantized_layers(model)

    if mode == "bnb":
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

        # ── 3b. Visual tower sanity check (BNB only) ──────────────────────────
        # Tower is still present here (drop happens in step 4 below).
        _live = dict(model.named_modules())
        vis_quant_live = [n for n in vis_quant if n in _live]
        vis_fp_live    = [n for n in vis_fp    if n in _live]

        if vis_fp_live:
            sample_vis = _live.get(vis_fp_live[0])
            vis_dtype  = "bf16"
            if sample_vis is not None:
                params = list(sample_vis.parameters())
                if params:
                    vis_dtype = str(params[0].dtype).replace("torch.", "")
            print(f"   ✅ Visual tower: {len(vis_fp_live)} layer(s) at full precision ({vis_dtype})")
        elif not vis_fp_live and not vis_quant_live:
            print("   ℹ️  No visual tower detected (text-only model)")
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
    else:
        # f16 — report weight dtype
        sample_params = [p for p in model.parameters() if p.device.type != "meta"]
        if sample_params:
            dtype_str = str(sample_params[0].dtype).replace("torch.", "")
            n_linear  = sum(
                1 for _, m in model.named_modules()
                if type(m).__name__ in ("Linear", "Linear4bit")
            )
            print(f"   ✅ {n_linear} linear layers  |  dtype: {dtype_str}")
        else:
            print("   ℹ️  No parameters found on non-meta device")

    # ── 4. Image inference ────────────────────────────────────────────────────
    print("\n── 4. Inference  [IMAGE] ────────────────────────────────")
    passed_img = True
    if arch == "text":
        print("   ℹ️  Skipped (text-only model)")
    elif _has_visual(model):
        strategy = getattr(model, "_strategy", "cpu")
        # cuda_direct: model already on GPU (moved in load_qwen35).
        # cuda_drop / cpu: visual tower on CPU — run test here, then move backbone.
        passed_img = _run_image_inference(model, tokenizer)
        drop_visual_tower(model)
        if strategy == "cuda_direct":
            torch.cuda.empty_cache()
        elif strategy == "cuda_drop":
            torch.cuda.empty_cache()
            _move_to_cuda(model)
        print_vram("after visual drop")
    else:
        print("   ⚠️  Skipped (arch=vlm but no visual tower found — model may have been stripped already)")

    # ── 5a. Inference — THINKING OFF ──────────────────────────────────────────
    print("\n── 5a. Inference  [THINKING OFF] ────────────────────────")
    passed_off, tasks_off = run_inference_tasks(
        model, tokenizer, DEFAULT_TEST_TASKS,
        chat_template_kwargs={"enable_thinking": False},
    )

    # ── 5b. Inference — THINKING ON ───────────────────────────────────────────
    print("\n── 5b. Inference  [THINKING ON] ─────────────────────────")
    passed_on, tasks_on = run_inference_tasks(
        model, tokenizer, DEFAULT_TEST_TASKS,
        chat_template_kwargs={"enable_thinking": True},
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result.task_results = tasks_off + tasks_on
    result.passed       = passed_off and passed_on and passed_img

    if result.passed:
        print("\n   ✅ ALL CHECKS PASSED\n")
    else:
        print("\n   ⚠️  SOME CHECKS FAILED — see above\n")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for verify_qwen35.py."""
    ap = argparse.ArgumentParser(
        description="Sanity-check a Qwen3.5 model — BNB 4-bit or f16, VLM or text-only (auto-detected).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Local model directory or HF Hub repo id (e.g. techwithsergiu/Qwen3.5-0.8B-bnb-4bit).",
    )
    ap.add_argument(
        "--hf-token", default=None,
        help="HF access token for private repos. Omit to use credentials from `hf auth login`.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-verify-qwen35 CLI command."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    verify_qwen35(model_path=args.model)


if __name__ == "__main__":
    main()
