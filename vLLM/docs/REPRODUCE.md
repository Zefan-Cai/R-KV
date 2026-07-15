# Reproducing / regenerating the R-KV patch

The wiring patch [`patch/rkv-vllm-0.25.1.patch`](../patch/rkv-vllm-0.25.1.patch)
is generated against a pinned upstream vLLM commit. Anyone can rebuild the
patched tree, verify the patch applies cleanly, or regenerate it after editing
the wiring.

## Pinned reference

| | |
| --- | --- |
| Repo | `https://github.com/vllm-project/vllm.git` |
| Tag | `v0.25.1` |
| Commit | `752a3a504485790a2e8491cacbb35c137339ad34` |

## Build the patched tree

```bash
scripts/apply_rkv.sh            # clone v0.25.1 + copy rkv/ + apply patch
# or, to overwrite an existing ./vllm-src:
scripts/apply_rkv.sh --force
```

`vllm-src/` is git-ignored — it is a build artifact, never vendored into this
repository.

## Verify the patch applies cleanly (non-destructive)

```bash
cd vllm-src
git stash                                   # park any local edits
git apply --check ../patch/rkv-vllm-0.25.1.patch
git stash pop
```

`git apply --check` exits 0 when the patch applies cleanly to the pristine
v0.25.1 tree.

## Regenerate the patch after editing the wiring

The source of truth for the *algorithm* is `rkv/`; the source of truth for the
*wiring* is the patch. If you change the wiring, edit the files inside
`vllm-src/` and regenerate:

```bash
cd vllm-src
# ... make wiring edits to the 9 tracked files ...
git diff > ../patch/rkv-vllm-0.25.1.patch
```

Do **not** hand-edit the patch file. Do **not** rely on edits inside `vllm-src/`
persisting — they are invisible to this repo's git and are wiped by
`apply_rkv.sh --force`. Always fold wiring changes back into the patch.

## The 10 wired files

```
vllm/envs.py
vllm/config/vllm.py
vllm/v1/request.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/outputs.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/flash_attn.py
```
