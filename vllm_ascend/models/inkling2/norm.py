# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU RMSNorm for Inkling 2.

Upstream counterpart: ``vllm/models/inkling2/nvidia/ops/norm.py``. The
``_rms_norm_kernel`` body is upstream's, unchanged. What differs is the
``rsqrt`` it calls: upstream's is inline PTX (``rsqrt.approx.f32``), which
triton-ascend does not support, here it is ``tl.rsqrt``; and the wrapper checks
that its inputs share one device. The unit test compares the result with the
fp32 PyTorch reference."""

import torch
import triton
import triton.language as tl


@triton.jit
def rsqrt(x):
    """Reciprocal square root, replacing upstream's inline PTX rsqrt."""
    return tl.rsqrt(x)


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    stride_x,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), N) + eps)
    y = x * rstd
    y = y * tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim of a row-major (possibly row-strided) tensor."""
    n = x.shape[-1]
    assert x.stride(-1) == 1
    # A tensor of another device would hand the kernel an invalid address.
    assert x.device == weight.device, "rms_norm inputs must share one device"
    x2d = x if x.dim() == 2 else x.reshape(-1, n)
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    if x2d.shape[0]:
        block = triton.next_power_of_2(n)
        _rms_norm_kernel[(x2d.shape[0],)](
            x2d,
            weight,
            out,
            x2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 512, 1), 16),
        )
    return out


@triton.jit
def _l2_clip(X, Scale, heads, cols, mask, threshold, eps, tau, HAS_TAU):
    x = tl.load(X + heads * cols.shape[1] + cols, mask, other=0).to(tl.float32)
    x = x * tl.load(Scale + heads * cols.shape[1] + cols, mask, other=0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=1)
    norm = tl.where(sum_sq > 0, sum_sq * rsqrt(sum_sq), 0.0)
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)[:, None]
    if HAS_TAU:
        y = y.to(X.dtype.element_ty).to(tl.float32) * tau
    return y.to(X.dtype.element_ty)


@triton.jit
def qr_clip_rel_tokens(
    t0,
    num_t: tl.constexpr,
    chunk,
    T,
    Q,
    R,
    QScale,
    RScale,
    Tau,
    RelProj,
    QOut,
    RelOut,
    SeqIdx,
    CuSeqlensQ,
    SeqLens,
    stride_q,
    stride_r,
    q_threshold,
    r_threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    E: tl.constexpr,
    W: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """For tokens `[t0, t0 + num_t)`: per-head L2 clipping of `q * q_scale`
    (`chunk < 0`) or of `r * r_scale` and the `BLOCK_E` columns `chunk`
    of the relative logits `r @ rel_proj` (fp32 accumulate; `rel_proj`
    read from the `shift_rel_proj` table `RelProj`). Clipped values are
    rounded, then scaled by the fp32 per-token `tau`.

    The logits are written sheared, as tml_fa4's `ShearingBias` does for
    128-token query tiles: `E + 256` columns per head, column `c` holding
    key `c + (n_block_max - 10) * 128`, where `n_block_max` bounds the key
    blocks of the token's tile, plus the table's padding (keys right of the
    diagonal, and for window layers those past the extent, are -inf)."""
    E_PAD: tl.constexpr = E + 256
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    q_cols = tl.arange(0, D)[None, :]
    r_cols = tl.arange(0, D_REL)[None, :]
    e_idx = tl.arange(0, BLOCK_E)[None, :]
    for i in tl.static_range(num_t):
        t = (t0 + i).to(tl.int64)
        if t < T and chunk < 0:
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            q = _l2_clip(
                Q + t * stride_q,
                QScale,
                heads,
                q_cols,
                h_mask,
                q_threshold,
                eps,
                tau,
                HAS_TAU,
            )
            tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)
        elif t < T:
            seq = tl.load(SeqIdx + t)
            q_start = tl.load(CuSeqlensQ + seq)
            q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
            q_idx = t - q_start
            pos = tl.load(SeqLens + seq) - q_len + q_idx
            tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
            # Column of the diagonal key (distance 0), in [E + 1, E + 255] for a
            # real token. CUDA-graph padding rows (seq_idx 0, q_idx past the
            # request) would land outside and read before the shifted table.
            diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
            diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
            tau = tl.load(Tau + t) if HAS_TAU else 1.0
            r = _l2_clip(
                R + t * stride_r,
                RScale,
                heads,
                r_cols,
                h_mask,
                r_threshold,
                eps,
                tau,
                HAS_TAU,
            )
            # Distance diag - c of column c is column (c - diag8 + E + 256) of
            # shifted table diag - diag8: 16-byte aligned loads.
            diag8 = diag // 8 * 8
            table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
            col = chunk * BLOCK_E + e_idx
            proj = tl.load(table + tl.trans(r_cols) * W + col)
            pad = tl.load(table + D_REL * W + col)
            rel = tl.dot(r, proj) + pad.to(tl.float32)
            tl.store(
                RelOut + t * H * E_PAD + heads * E_PAD + col,
                rel.to(RelOut.dtype.element_ty),
                h_mask,
            )


@triton.jit
def _l2_clip_rows_kernel(
    X,
    Scale,
    Tau,
    Y,
    stride_t,
    stride_h,
    stride_y_t,
    stride_y_h,
    stride_scale,
    H,
    D: tl.constexpr,
    threshold,
    eps,
    HAS_SCALE: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-head L2 clip of one row: ``x * scale`` in fp32, times
    ``min(threshold / (norm + eps), 1)``; with ``tau``, rounded to bf16 and
    then scaled by it (the rounding of ``attention.l2_clip``).

    One program per (token, head) row on a flat 1-D grid: on triton-ascend a
    second program dimension next to a scalar reduction fails GraphOptimize.
    """
    row = tl.program_id(0).to(tl.int64)
    head = row % H
    token = row // H
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X + token * stride_t + head * stride_h + cols, mask, other=0).to(tl.float32)
    if HAS_SCALE:
        x = x * tl.load(Scale + head * stride_scale + cols, mask, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x))
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)
    if HAS_TAU:
        y = y.to(Y.dtype.element_ty).to(tl.float32) * tl.load(Tau + token)
    tl.store(Y + token * stride_y_t + head * stride_y_h + cols, y.to(Y.dtype.element_ty), mask)


def l2_clip_triton(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    threshold: float,
    tau: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``attention.l2_clip`` as one Triton launch: ``x`` is ``[T, H, D]`` bf16
    (row- and head-strided views of the fused projection are fine), ``scale``
    ``[H, D]`` bf16, ``tau`` fp32 ``[T]`` or None."""
    tokens, heads, width = x.shape
    out = torch.empty_like(x)
    _l2_clip_rows_kernel[(tokens * heads,)](
        x,
        scale if scale is not None else x,
        tau if tau is not None else x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        scale.stride(0) if scale is not None else 0,
        heads,
        width,
        threshold,
        eps,
        HAS_SCALE=scale is not None,
        HAS_TAU=tau is not None,
        BLOCK_D=triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
