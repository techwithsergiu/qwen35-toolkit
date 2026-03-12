"""
inspect_safetensors.py
──────────────────────
Reads safetensors shard headers and prints all unique top-2-level weight
prefixes with their cumulative sizes.

No weights are loaded — only the JSON header of each shard is read
(a few KB per file), so this runs instantly even on 70B models.

Useful for:
  • Confirming which components are present (language_model, visual, lm_head …)
  • Estimating visual tower size before stripping
  • Verifying a strip_visual run removed the expected prefixes

Usage:
  qwen35-inspect ./Qwen3.5-4B-bnb-4bit
  qwen35-inspect ./Qwen3.5-text-4B-bnb-4bit
  qwen35-inspect unsloth/Qwen3.5-4B          # NOT supported (local only)

Example output:
  Found 5 shard(s)

  PREFIX                                            SIZE
  ─────────────────────────────────────────────────────
  model.language_model                          3.54 GB
  model.visual                                  0.89 GB
  lm_head                                      90.0 MB
  model.visual.merger                          12.3 MB
"""

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


# ── Core logic ─────────────────────────────────────────────────────────────────

def _read_prefix_sizes(model_path: Path) -> dict[str, int]:
    """
    Walk every .safetensors shard in model_path and accumulate byte sizes
    per top-2-level prefix (e.g. "model.visual", "lm_head").

    Only the JSON header of each shard is read — no tensor data is loaded.

    Returns:
        {prefix: total_bytes} sorted by size descending.
    """
    shards = sorted(model_path.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in: {model_path}")

    prefix_bytes: dict[str, int] = defaultdict(int)

    for shard in shards:
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            parts   = key.split(".")
            prefix  = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            offsets = meta.get("data_offsets")
            if offsets and len(offsets) == 2:
                prefix_bytes[prefix] += offsets[1] - offsets[0]

    return dict(sorted(prefix_bytes.items(), key=lambda x: -x[1]))


def _fmt_size(n_bytes: int) -> str:
    """Format byte count as a human-readable string (MB or GB)."""
    gb = n_bytes / 1024**3
    mb = n_bytes / 1024**2
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


# ── Printing ───────────────────────────────────────────────────────────────────

def inspect(model_path: str) -> None:
    """
    Print a summary table of weight prefixes and their sizes for a local
    safetensors model directory.

    Args:
        model_path: Local directory containing .safetensors shards.
    """
    path = Path(model_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    shards = sorted(path.rglob("*.safetensors"))
    print(f"Found {len(shards)} shard(s)\n")

    prefix_sizes = _read_prefix_sizes(path)

    col = 45
    print(f"{'PREFIX':<{col}}  {'SIZE':>8}")
    print(f"{'─' * col}  {'─' * 8}")
    for prefix, nbytes in prefix_sizes.items():
        print(f"{prefix:<{col}}  {_fmt_size(nbytes):>8}")

    total = sum(prefix_sizes.values())
    print(f"\n{'TOTAL':<{col}}  {_fmt_size(total):>8}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Inspect safetensors shards: print weight prefixes and sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "model_path",
        nargs="?",
        default=".",
        help="Local directory containing .safetensors shards.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-inspect CLI command."""
    args = parse_args()
    try:
        inspect(args.model_path)
    except FileNotFoundError as e:
        sys.exit(f"❌  {e}")


if __name__ == "__main__":
    main()
