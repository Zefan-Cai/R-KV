"""Fused Triton redundancy kernel for R-KV prefill (Stage 1: no retain-exemption).

Computes the R-KV redundancy signal::

    redundancy = softmax_j( (1/n) * sum_i simcos(i, j) )

where ``simcos`` is the normalized-key cosine similarity with the diagonal
zeroed, *without* the per-row "retain the most-recent above-threshold
neighbour" exemption. The ``n x n`` similarity matrix is never materialised:
each ``(row-block, col-block)`` tile is formed in registers via ``tl.dot`` and
reduced into a per-column running sum, so HBM traffic is ``O(n*d)`` reads of the
(pre-normalized) keys plus an ``O(n)`` write of the column sums — instead of the
``O(n^2)`` write+reread of the materialised matrix in ``cal_similarity_tiled``.

The kernel is **col-parallel**: one program owns ``BLOCK_N`` output columns and
loops over all rows, accumulating the column sum locally (no atomics). Stage 2
adds the retain exemption as a cheap ``O(n)``-atomics correction to reach
bit-parity with ``algo.cal_similarity`` / ``cal_similarity_tiled``.

This module imports ``triton`` at load time, so it is only imported on the CUDA
serving path; the CPU-by-path unit tests exercise ``cal_similarity_tiled``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _colsum_kernel(
    K_ptr,  # (H, N, D) normalized keys
    OUT_ptr,  # (H, N) fp32 column sums
    N,
    D,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_oh,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    col_pid = tl.program_id(1)

    cols = col_pid * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    col_mask = cols < N
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    # Columns this program owns: K[h, cols, :] -> (BLOCK_N, BLOCK_D)
    kcol = tl.load(
        K_ptr + h * stride_kh + cols[:, None] * stride_kn + d[None, :] * stride_kd,
        mask=col_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    kcol_t = tl.trans(kcol)  # (BLOCK_D, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row0 in range(0, N, BLOCK_M):
        rows = row0 + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        row_mask = rows < N
        krow = tl.load(
            K_ptr + h * stride_kh + rows[:, None] * stride_kn + d[None, :] * stride_kd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # (BLOCK_M, BLOCK_D)
        sim = tl.dot(krow, kcol_t, input_precision="ieee")  # (BLOCK_M, BLOCK_N) fp32
        # Zero the diagonal (self-similarity) and any padded rows/cols.
        diag = rows[:, None] == cols[None, :]
        drop = diag | (~row_mask[:, None]) | (~col_mask[None, :])
        sim = tl.where(drop, 0.0, sim)
        acc += tl.sum(sim, axis=0)  # (BLOCK_N,)

    tl.store(OUT_ptr + h * stride_oh + cols * stride_on, acc, mask=col_mask)


def cal_similarity_fused_noretain(
    key_states: torch.Tensor,
    block_n: int = 128,
    block_m: int = 64,
) -> torch.Tensor:
    """Fused redundancy WITHOUT the retain exemption.

    ``key_states``: ``(bsz, kv_heads, n, d)``. Returns ``(bsz, kv_heads, n)``.
    Matches a reference that zeroes only the diagonal (no ``similarity_retain``).
    """
    bsz, kv_heads, n, d = key_states.shape
    h = bsz * kv_heads
    k = key_states.reshape(h, n, d)
    k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-8)).contiguous()
    out = torch.empty((h, n), dtype=torch.float32, device=k.device)

    block_d = triton.next_power_of_2(d)
    grid = (h, triton.cdiv(n, block_n))
    _colsum_kernel[grid](
        k_norm,
        out,
        n,
        d,
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    redundancy = (out / n).softmax(dim=-1)
    return redundancy.reshape(bsz, kv_heads, n).to(key_states.dtype)


@triton.jit
def _colsum_retain_kernel(
    K_ptr,  # (H, N, D) normalized keys
    OUT_ptr,  # (H, N) fp32 column sums, pre-zeroed
    N,
    D,
    threshold,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_oh,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Row-parallel: one program owns BLOCK_M rows, sees the full row, so it can
    find the per-row retain column (largest column with cos > threshold). It
    atomically accumulates the column sums and, at the end, subtracts the single
    exempted entry per row — matching the reference's ``similarity_retain``.
    """
    h = tl.program_id(0)
    row_pid = tl.program_id(1)

    rows = row_pid * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    row_mask = rows < N
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    krow = tl.load(
        K_ptr + h * stride_kh + rows[:, None] * stride_kn + d[None, :] * stride_kd,
        mask=row_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # (BLOCK_M, BLOCK_D)

    retain_col = tl.zeros((BLOCK_M,), dtype=tl.int32)  # default column 0
    retain_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    has_any = tl.zeros((BLOCK_M,), dtype=tl.int32)
    val_at_col0 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for col0 in range(0, N, BLOCK_N):
        cols = col0 + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        col_mask = cols < N
        kcol = tl.load(
            K_ptr + h * stride_kh + cols[:, None] * stride_kn + d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # (BLOCK_N, BLOCK_D)
        sim = tl.dot(krow, tl.trans(kcol), input_precision="ieee")  # (BLOCK_M, BLOCK_N)
        diag = rows[:, None] == cols[None, :]
        drop = diag | (~row_mask[:, None]) | (~col_mask[None, :])
        sim = tl.where(drop, 0.0, sim)

        # Accumulate the (un-exempted) column sum.
        tl.atomic_add(
            OUT_ptr + h * stride_oh + cols * stride_on,
            tl.sum(sim, axis=0),
            mask=col_mask,
        )

        # Capture sim[:, 0] for the default (no-neighbour) retain. Column 0 lives
        # only in the first block, so this is nonzero there and zero elsewhere.
        val_at_col0 += tl.sum(tl.where(cols[None, :] == 0, sim, 0.0), axis=1)

        # Largest above-threshold column in this block (forward sweep => later
        # blocks always dominate), and its similarity value.
        over = sim > threshold
        cand = tl.where(over, cols[None, :].to(tl.int32), -1)
        blk_max_col = tl.max(cand, axis=1)  # (BLOCK_M,), -1 if none
        has_blk = blk_max_col >= 0
        is_bmc = cols[None, :].to(tl.int32) == blk_max_col[:, None]
        blk_val = tl.sum(tl.where(is_bmc, sim, 0.0), axis=1)  # (BLOCK_M,)
        retain_col = tl.where(has_blk, blk_max_col, retain_col)
        retain_val = tl.where(has_blk, blk_val, retain_val)
        has_any = tl.where(has_blk, 1, has_any)

    final_col = tl.where(has_any > 0, retain_col, 0)
    final_val = tl.where(has_any > 0, retain_val, val_at_col0)
    # Exempt the single retained entry per row.
    tl.atomic_add(
        OUT_ptr + h * stride_oh + final_col * stride_on,
        -final_val,
        mask=row_mask,
    )


def cal_similarity_fused(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    block_n: int = 64,
    block_m: int = 128,
) -> torch.Tensor:
    """Fused redundancy WITH the retain exemption (bit-parity target:
    ``algo.cal_similarity`` / ``cal_similarity_tiled`` for retain_direction='last').

    ``key_states``: ``(bsz, kv_heads, n, d)``. Returns ``(bsz, kv_heads, n)``.
    """
    bsz, kv_heads, n, d = key_states.shape
    h = bsz * kv_heads
    k = key_states.reshape(h, n, d)
    k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-8)).contiguous()
    out = torch.zeros((h, n), dtype=torch.float32, device=k.device)

    block_d = triton.next_power_of_2(d)
    grid = (h, triton.cdiv(n, block_m))
    _colsum_retain_kernel[grid](
        k_norm,
        out,
        n,
        d,
        float(threshold),
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    redundancy = (out / n).softmax(dim=-1)
    return redundancy.reshape(bsz, kv_heads, n).to(key_states.dtype)
