"""
This package provides efficient decoding-time KV cache compression methods.
"""

__version__ = "0.1.0"

# The monkeypatch swaps transformers attention/CausalLM forwards, so it is
# coupled to the transformers internal API. Outside the tested range the
# patched model can silently produce garbled output (issue #17 / #26).
_TRANSFORMERS_MIN = "4.48.1"
_TRANSFORMERS_MAX_EXCLUSIVE = "4.56"

__all__ = ["replace_llama", "replace_qwen2", "replace_qwen3"]


def __getattr__(name):
    # Loading the modern hybrid adapter must not import legacy forward patches.
    if name not in __all__:
        raise AttributeError(name)
    import warnings
    from packaging.version import Version
    import transformers

    version = Version(transformers.__version__)
    if not (Version(_TRANSFORMERS_MIN) <= version < Version(_TRANSFORMERS_MAX_EXCLUSIVE)):
        warnings.warn(
            f"rkv legacy monkeypatches are tested with transformers>={_TRANSFORMERS_MIN},"
            f"<{_TRANSFORMERS_MAX_EXCLUSIVE} but found {transformers.__version__}. "
            "The attention monkeypatch may silently produce garbled output on "
            "other versions; please install a supported transformers release.",
            RuntimeWarning,
            stacklevel=2,
        )
    from . import monkeypatch

    value = getattr(monkeypatch, name)
    globals()[name] = value
    return value
