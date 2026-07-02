"""SGLang (0.4.3.post1 fork) R-KV smoke: Qwen2.5-0.5B-Instruct,
triton backend, cuda graph off. Toggle with RKV_ON=0/1; BATCH=1/2.
Needs a __main__ guard: sglang launches subprocesses via mp spawn."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RKV_ON = os.environ.get("RKV_ON", "1") == "1"
BATCH = int(os.environ.get("BATCH", "1"))

PROMPT = ("Explain why decode-time KV cache compression needs separate real "
          "sequence lengths and compressed KV lengths. " * 12)


def main():
    import sglang as sgl

    kwargs = dict(
        model_path=MODEL,
        dtype="bfloat16",
        attention_backend="triton",
        mem_fraction_static=0.25,
        disable_cuda_graph=True,
    )
    if RKV_ON:
        kwargs.update(
            compress_algorithm="RKV",
            compress_max_window=8,
            compress_max_prompt=32,
            compress_divide_length=16,
            compress_divide_method="step_length",
        )

    t0 = time.time()
    llm = sgl.Engine(**kwargs)
    load_s = time.time() - t0

    prompts = [PROMPT] * BATCH
    t0 = time.time()
    outputs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 128})
    gen_s = time.time() - t0

    if isinstance(outputs, dict):
        outputs = [outputs]

    tag = ("rkv_on" if RKV_ON else "rkv_off") + f",batch={BATCH}"
    all_ok = True
    for i, output in enumerate(outputs):
        text = output.get("text", "") if isinstance(output, dict) else str(output)
        print(f"\n===== SGLANG ({tag}) output {i} =====")
        print(f"Output({len(text)} chars): {text[:500]!r}")
        all_ok &= report(f"sglang[{tag}] out{i}", text)

    print(f"LOAD_SECONDS {load_s:.3f}")
    print(f"GENERATE_SECONDS {gen_s:.3f}")
    print("SGLANG_SMOKE_PASS" if all_ok else "SGLANG_SMOKE_FAIL")
    llm.shutdown()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
