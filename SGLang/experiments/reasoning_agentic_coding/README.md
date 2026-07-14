# Reasoning, agentic, and coding evaluation

This campaign compares the latest R-KV SGLang port at repository commit
`026639a81c1f8a4a43dbee3823b764a2ccbbd6aa` with matched FullKV baselines on
`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` revision
`003f183a92fbe5b9a8325aaa8b2ae797c91dd90f`.

The checkpoint is the strongest official open-weight model we found that fits
the implementation's actual gate: standard full-attention GQA, no MLA, no
hybrid sliding/linear attention, and single-node TP8. The launch also uses EP8
so the checkpoint's 128-wide FP8 expert blocks are not split across tensor
parallel ranks. FP8 applies to weights; the KV cache remains BF16/FP16
(`--kv-cache-dtype auto`).

## Paired lanes

Do not combine decode and prefill results into one score.

| Arm | Purpose | Key configuration |
| --- | --- | --- |
| `d-prod` | production FullKV context | radix/overlap/graphs enabled |
| `d-full` | fair decode FullKV | same radix/overlap/page constraints as R-KV |
| `d-4k` | main decode R-KV | budget 4096, window 8, buffer 128 |
| `d-8k` | conservative decode R-KV | budget 8192, window 8, buffer 128 |
| `p-full` | fair prefill FullKV | prefill graph off, chunk size 2048 |
| `p-4k` | main agentic prefill R-KV | buffered budget 4096 |
| `p-2k` | agentic pressure point | buffered budget 2048 |
| `p-o4k` | tiny accuracy oracle only | one-shot budget 4096 |

The decode lane targets AIME 2024/2025, LiveCodeBench v6, and HumanEval+.
The prefill lane targets deterministic BFCL V4 `multi_turn_long_context` and
memory tasks, followed by a fixed SWE-bench Verified subset with mini-SWE-agent.
Web-search BFCL tasks are excluded because their external state drifts.

Always report both aggregate and actually-triggered subsets. A benchmark with
fewer than 20% compressed requests is a no-regression control, not evidence for
compression quality.

## Pluto pilot

The Pluto entrypoints provision one H200 P1 node and one H200 P2 node. Because
the 450 GiB checkpoint transfer is longer than observed reclaim windows, its
partial shards and final `MODEL_READY` marker persist under
`/sensei-fs/users/zcai/rkv-sglang-eval-20260713/models`; a shared `flock` keeps
the P1/P2 download resumable and single-writer. Results live beside it under
the campaign root. The nodes run the CPU and H200 fused-kernel parity suites.
To reduce cold-start time, the default bootstrap uses the v0.5.14 wheel for
dependencies/native artifacts and overlays the exact patched `49e384ce` Python
tree for this HTTP-only campaign; a full Rust/protoc source build remains
available with `SGLANG_INSTALL_MODE=source`.
The nodes then execute a crossover so node/project effects do not masquerade as
R-KV effects:

- P1 order: `d-full`, `d-4k`, `p-full`, `p-4k`
- P2 order: `d-4k`, `d-full`, `p-4k`, `p-full`

Each decode arm runs short/tool/forced-long probes, AIME24 pilot questions, and
HumanEval+ through EvalPlus. Each prefill arm runs the same health probes plus a
deterministic BFCL V4 long-context/memory pilot with all required memory setup
dependencies. The server context cap is 65,536 tokens for these pilot slices.

For a reclaimed P2 allocation, `MODEL_RSYNC_HOST=<P1 pod IP>` and optional
`MODEL_RSYNC_PORT` stage the already-verified public checkpoint from P1 over the
pod network instead of repeating the 450 GiB Hugging Face transfer.

Results are written to the shared filesystem and, when pod credentials allow,
incrementally mirrored to
`s3://phidias/zcai/codex/rkv-sglang-eval-20260713/results/`.

The bundled BFCL patch only registers the actual Qwen3-Coder checkpoint with
BFCL's existing Qwen3 FC handler; it does not change datasets or scoring. After
the crossover passes, expand to `d-8k`, `p-2k`, LiveCodeBench v6, and the fixed
SWE-bench Verified subset described above.
