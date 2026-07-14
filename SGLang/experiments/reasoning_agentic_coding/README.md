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

Pluto fixes the process stack hard limit at 10 MiB. EvalPlus 0.3.1 attempts to
raise that limit to its default 4 GiB memory cap, which makes every worker exit
before executing generated code. The harness therefore uses EvalPlus' `-1`
memory-limit sentinel on Pluto; its destructive-call guard and per-test
timeouts remain enabled. Validation rejects the characteristic all-timeout
artifact so evaluator failure cannot be reported as model pass@1=0.

Always report both aggregate and actually-triggered subsets. A benchmark with
fewer than 20% compressed requests is a no-regression control, not evidence for
compression quality.

## Pluto pilot

The Pluto entrypoints provision one H200 P1 node and one H200 P2 node. Because
the 450 GiB checkpoint transfer is longer than observed reclaim windows and
larger than one roughly 250 GiB Sensei project quota, finalized shards are
striped across
`/sensei-fs/users/zcai/rkv-sglang-eval-20260713/models` and
`/sensei-fs-3/users/zcai/rkv-sglang-eval-20260713/models`. The primary model
directory retains shards 1-24 and symlinks secondary shards 25-49 into one
loader-visible checkpoint. Package caches stay on node-local NVMe so the two
persistent quotas retain safety headroom. A
shared `flock` keeps the P1/P2 transfer resumable and single-writer, and the
final `MODEL_READY` marker is written only after all 49 shards validate.
Results live under the primary campaign root. The nodes run the CPU and H200
fused-kernel parity suites.
To reduce cold-start time, the default bootstrap uses the v0.5.14 wheel for
dependencies/native artifacts and overlays the exact patched `49e384ce` Python
tree for this HTTP-only campaign; a full Rust/protoc source build remains
available with `SGLANG_INSTALL_MODE=source`.
The nodes then execute a crossover so node/project effects do not masquerade as
R-KV effects:

- P1 order: `d-full`, `d-4k`, `p-full`, `p-4k`
- P2 order: `d-4k`, `d-full`, `p-4k`, `p-full`

Each decode arm runs short/tool/forced-long probes, all 30 bundled AIME24
questions, and HumanEval+ through EvalPlus. Each prefill arm runs the same
health probes plus a deterministic BFCL V4 long-context/memory pilot with all
required memory setup dependencies. The server context cap is 65,536 tokens
for these pilot slices.

The earlier whole-checkpoint `MODEL_RSYNC_HOST` shortcut is intentionally
disabled: it would place all 449 GiB on the primary quota and reproduce the
failure. Either lane instead resumes the same deterministic 24/25-shard split.

Results are written to the shared filesystem and, when pod credentials allow,
incrementally mirrored to
`s3://phidias/zcai/codex/rkv-sglang-eval-20260713/results/`.

The bundled BFCL patch only registers the actual Qwen3-Coder checkpoint with
BFCL's existing Qwen3 FC handler; it does not change datasets or scoring. After
the crossover passes, expand to `d-8k`, `p-2k`, LiveCodeBench v6, and the fixed
SWE-bench Verified subset described above.
