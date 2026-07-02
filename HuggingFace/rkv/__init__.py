"""
This package provides efficient decoding-time KV cache compression methods.
"""

import warnings

__version__ = "0.1.0"

# The monkeypatch swaps transformers attention/CausalLM forwards, so it is
# coupled to the transformers internal API. Outside the tested range the
# patched model can silently produce garbled output (issue #17 / #26).
_TRANSFORMERS_MIN = "4.48.1"
_TRANSFORMERS_MAX_EXCLUSIVE = "4.56"

try:
    from packaging.version import Version

    import transformers

    _v = Version(transformers.__version__)
    if not (Version(_TRANSFORMERS_MIN) <= _v < Version(_TRANSFORMERS_MAX_EXCLUSIVE)):
        warnings.warn(
            f"rkv is tested with transformers>={_TRANSFORMERS_MIN},"
            f"<{_TRANSFORMERS_MAX_EXCLUSIVE} but found {transformers.__version__}. "
            "The attention monkeypatch may silently produce garbled output on "
            "other versions; please install a supported transformers release.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    pass

from .monkeypatch import replace_llama, replace_qwen2, replace_qwen3

__all__ = ["replace_llama", "replace_qwen2", "replace_qwen3"]
