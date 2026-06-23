import ctypes
import os
import sysconfig


def _preload_pip_nvidia_runtime_libs():
    platlib = sysconfig.get_paths().get("platlib")
    if not platlib:
        return

    lib_paths = [
        os.path.join(platlib, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"),
        os.path.join(platlib, "nvidia", "cuda_runtime", "lib", "libcudart.so.12"),
    ]
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_pip_nvidia_runtime_libs()

# SGLang public APIs

# Frontend Language APIs
from sglang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)
from sglang.utils import LazyImport

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Other configs
from sglang.global_config import global_config
from sglang.version import __version__

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "Anthropic",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
