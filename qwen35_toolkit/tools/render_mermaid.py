"""
render_mermaid.py
─────────────────────
Pre-renders every ```mermaid block in a Markdown file to PNG using the
Mermaid CLI (mmdc), then outputs a new Markdown file with the blocks
replaced by image references — ready for PDF conversion in VS Code
Office Viewer (or any other tool).

Requirements:
    npm install -g @mermaid-js/mermaid-cli
    npx puppeteer browsers install chrome-headless-shell

Usage:
    qwen35-render-mermaid --input README.md
    qwen35-render-mermaid --input README.md --output my_doc_pdf.md
    qwen35-render-mermaid --input README.md --scale 3.0

    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-0.8B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-2B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-4B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-bnb-4bit.md --output ./Qwen3.5-9B-bnb-4bit/README.md

    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-0.8B/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-2B/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-4B/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text.md --output ./Qwen3.5-text-9B/README.md

    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-0.8B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-2B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-4B-bnb-4bit/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-bnb-4bit.md --output ./Qwen3.5-text-9B-bnb-4bit/README.md

    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-0.8B-GGUF/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-2B-GGUF/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-4B-GGUF/README.md
    qwen35-render-mermaid --scale 3.0 --input ./model_cards/README_Qwen3.5-text-GGUF.md --output ./Qwen3.5-text-9B-GGUF/README.md

Output:
    <input_stem>_pdf.md        — Markdown with ```mermaid blocks replaced by ![](...)
    diagrams/diagram_01.png    — Rendered PNGs (relative to the output .md)
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ── mmdc config file — controls PNG background and font ───────────────────────
MMDC_CONFIG = """{
  "theme": "default",
  "backgroundColor": "white"
}
"""


def find_mmdc() -> str:
    """Return path to mmdc or exit with a helpful message."""
    mmdc = shutil.which("mmdc")
    if mmdc:
        return mmdc
    # Common npm global locations
    for candidate in [
        Path.home() / ".npm-global" / "bin" / "mmdc",
        Path("/usr/local/bin/mmdc"),
    ]:
        if candidate.exists():
            return str(candidate)
    sys.exit(
        "❌  mmdc not found.\n"
        "    Install: npm install -g @mermaid-js/mermaid-cli\n"
        "    Then:    npx puppeteer browsers install chrome-headless-shell"
    )


def extract_mermaid_blocks(text: str) -> list[tuple[int, int, str]]:
    """
    Return list of (start, end, diagram_source) for every ```mermaid block.
    start/end are character indices in text.
    """
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    return [(m.start(), m.end(), m.group(1).strip()) for m in pattern.finditer(text)]


def render_block(
    mmdc: str,
    diagram: str,
    out_path: Path,
    scale: float,
    config_path: Path,
) -> bool:
    """Render a single Mermaid diagram to PNG. Returns True on success."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as f:
        f.write(diagram)
        tmp = Path(f.name)

    try:
        result = subprocess.run(
            [
                mmdc,
                "--input",  tmp,
                "--output", out_path,
                "--scale",  str(scale),
                "--configFile", config_path,
                "--backgroundColor", "white",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            print(f"   ⚠️  mmdc error:\n{result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌  mmdc failed with code {e.returncode}: {e.stderr}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def process(
    input_path: Path,
    output_path: Path,
    diagrams_dir: Path,
    scale: float,
) -> None:
    text = input_path.read_text(encoding="utf-8")
    blocks = extract_mermaid_blocks(text)

    if not blocks:
        print("ℹ️  No ```mermaid blocks found — copying file unchanged.")
        output_path.write_text(text, encoding="utf-8")
        return

    diagrams_dir.mkdir(parents=True, exist_ok=True)

    # Write mmdc config
    config_path = diagrams_dir / "mmdc_config.json"
    config_path.write_text(MMDC_CONFIG, encoding="utf-8")

    mmdc = find_mmdc()
    print(f"🔧 mmdc: {mmdc}")
    print(f"📄 Input : {input_path}")
    print(f"📄 Output: {output_path}")
    print(f"🖼  Diagrams → {diagrams_dir}/\n")

    # Image paths must be relative to the output markdown file
    img_rel_base = diagrams_dir.relative_to(output_path.parent)

    # Process blocks in reverse so character indices stay valid
    result_text = text
    for i, (start, end, diagram) in enumerate(reversed(blocks), start=1):
        idx = len(blocks) - i + 1   # maintain original numbering
        img_name = f"diagram_{idx:02d}.png"
        img_abs  = diagrams_dir / img_name
        img_rel  = img_rel_base / img_name

        print(f"  [{idx}/{len(blocks)}] Rendering → {img_name} …", end=" ", flush=True)
        ok = render_block(mmdc, diagram, img_abs, scale, config_path)
        print("✅" if ok else "❌ (kept as mermaid block)")

        if ok:
            replacement = f"![]({img_rel.as_posix()})"
            result_text = result_text[:start] + replacement + result_text[end:]

    output_path.write_text(result_text, encoding="utf-8")
    config_path.unlink(missing_ok=True)
    print(f"\n✅ Done → {output_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pre-render Mermaid blocks to PNG for PDF conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",  required=True, help="Input Markdown file.")
    ap.add_argument("--output", default=None,
                    help="Output Markdown file. Default: <input>_pdf.md")
    ap.add_argument("--diagrams-dir", default=None,
                    help="Directory for PNG files. Default: <output_dir>/diagrams/")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="PNG scale factor (2.0 = retina quality).")
    return ap.parse_args()


def main() -> None:
    """Entry point for the qwen35-render-mermaid CLI command."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"❌  File not found: {input_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_pdf.md")
    )
    diagrams_dir = (
        Path(args.diagrams_dir).resolve()
        if args.diagrams_dir
        else output_path.parent / "diagrams"
    )
    process(input_path, output_path, diagrams_dir, args.scale)


if __name__ == "__main__":
    main()
