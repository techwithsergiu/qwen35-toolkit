"""
qwen35_toolkit — Convert, strip, verify and publish Qwen3.5 models.

Typical usage from a training pipeline:

    from qwen35_toolkit.convert         import convert
    from qwen35_toolkit.strip           import strip_visual
    from qwen35_toolkit.verify_qwen35   import verify_qwen35
    from qwen35_toolkit.tools.upload    import upload

    convert(model_name="unsloth/Qwen3.5-4B", output_dir="./model-bnb")
    strip_visual(model_ref="./model-bnb", output_dir="./model-text", mode="bnb")
    verify_qwen35(model_path="./model-text")
"""

from qwen35_toolkit.convert import convert
from qwen35_toolkit.strip import strip_visual
from qwen35_toolkit.verify import verify
from qwen35_toolkit.verify_qwen35 import verify_qwen35
from qwen35_toolkit.tools.upload import upload

__all__ = ["convert", "strip_visual", "verify", "verify_qwen35", "upload"]
__version__ = "0.1.0"
