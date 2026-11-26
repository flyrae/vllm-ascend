# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-residual mixing for Inkling 2 on NPU.

Upstream counterpart: ``vllm/models/inkling2/nvidia/ops/attn_res.py``, with
the same math. For each token, the sources (stored blocks, then the partial
block plus an optional addend, all read in fp32) are scored as
``(q . x_n) * rsqrt(mean(x_n^2) + eps)`` per pseudo-query; the output is the
softmax-weighted sum, with the sources RMS-normalized when ``normalize_values``
is set. The fp32 mix then goes through the following sublayer's RMSNorm in the
same kernel, or is returned as is (fp32) without a norm weight.

What changed for NPU:

- ``_attn_res_kernel`` and its helpers ``_load_source`` / ``_mix`` keep
  upstream's code, without the PDL wait (``tl.extra.cuda.gdc_wait``), with
  ``tldevice.exp`` replaced by ``tl.math.exp`` and ``rsqrt`` by the
  ``tl.rsqrt`` of :mod:`.norm`
- upstream's kernel for small batches (``_attn_res_row_kernel``) is not ported:
  every batch runs ``_attn_res_kernel``, with ``BLOCK_D`` 512 instead of 1024
- the launch has no ``launch_pdl``; the wrapper checks that the inputs share
  one device"""

import torch
import triton
import triton.language as tl

from .norm import rsqrt


@triton.jit
def _load_source(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    n,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
):
    """Source `n` in fp32: a stored block, else `last (+ addend)`."""
    if n < N_STACK:
        x = tl.load(Stack + n * stride_s + row_off + offs, mask, other=0)
        x = x.to(tl.float32)
    else:
        x = tl.load(Last + row_off + offs, mask, other=0).to(tl.float32)
        if HAS_ADDEND:
            x = x + tl.load(Addend + row_off + offs, mask, other=0).to(tl.float32)
    return x


@triton.jit
def _mix(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    probs,
    n_idx,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """The fp32 probability-weighted sum of the sources of one block."""
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        x = _load_source(Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND)
        acc = tl.fma(tl.sum(tl.where(n_idx == n, probs, 0.0)), x, acc)
    return acc


@triton.jit
def _attn_res_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    n_idx = tl.arange(0, BLOCK_N)
    d_idx = tl.arange(0, BLOCK_D)

    # Pass 1: per-source inverse RMS and query dot products.
    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        sum_sq = tl.zeros((BLOCK_D,), tl.float32)
        dots = tl.zeros((BLOCK_D,), tl.float32)
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + d_idx
            mask = offs < D
            x = _load_source(Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND)
            q = tl.load(Queries + q_off + offs, mask, other=0)
            sum_sq = tl.fma(x, x, sum_sq)
            dots = tl.fma(q, x, dots)
        rstd = rsqrt(tl.fma(tl.sum(sum_sq), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        logits = tl.where(n_idx == n, tl.sum(dots) * rstd, logits)

    probs = tl.math.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    # Pass 2: RMS of the probability-weighted sum.
    out_rstd = 1.0
    if APPLY_NORM:
        sum_sq = 0.0
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + d_idx
            mask = offs < D
            acc = _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            acc = acc * inv_sum
            sum_sq += tl.sum(acc * acc)
        out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)

    # Pass 3: normalized sum.
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        mask = offs < D
        acc = _mix(
            Stack,
            stride_s,
            Last,
            Addend,
            row_off,
            offs,
            mask,
            probs,
            n_idx,
            N_STACK,
            HAS_ADDEND,
            BLOCK_D,
        )
        acc = acc * inv_sum
        if APPLY_NORM:
            w = tl.load(NormW + q_off + offs, mask, other=0).to(tl.float32)
            acc = acc * out_rstd * w
        tl.store(out + offs, acc.to(Out.dtype.element_ty), mask)


def _same_device(*tensors: torch.Tensor | None) -> bool:
    devices = {tensor.device for tensor in tensors if tensor is not None}
    return len(devices) == 1


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mix `[*stack, last (+ addend)]` with each pseudo-query.

    stack: (N, T, D) bf16 completed blocks; last, addend: (T, D) bf16
    queries: fp32 (Q, D), already scaled by `D**-0.5`
    norm_weight: bf16 (Q, D), row `q` applied after the mix's RMSNorm, or
        None for the fp32 mixes without the norm
    Returns (Q, T, D) in `last.dtype`, or fp32 without `norm_weight`.
    """
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    assert last.is_contiguous() and queries.is_contiguous()
    assert queries.dtype == torch.float32
    # A tensor of another device would hand the kernel an invalid address.
    assert _same_device(last, stack, queries, addend, norm_weight), "attn_res inputs must share one device"
    assert stack.stride(1) == d and stack.stride(2) == 1
    if addend is not None:
        assert addend.is_contiguous() and addend.shape == last.shape
    apply_norm = norm_weight is not None
    if norm_weight is None:
        norm_weight = queries  # dummy

    out_dtype = last.dtype if apply_norm else torch.float32
    out = torch.empty(num_q, num_tokens, d, dtype=out_dtype, device=last.device)
    if num_tokens:
        block_n = triton.next_power_of_2(num_stack + 1)
        block_d = min(triton.next_power_of_2(d), 512)
        _attn_res_kernel[(num_q, num_tokens)](
            stack,
            last,
            addend if addend is not None else last,
            queries,
            norm_weight,
            out,
            stack.stride(0),
            num_tokens,
            eps,
            norm_eps,
            D=d,
            N_STACK=num_stack,
            HAS_ADDEND=addend is not None,
            NORMALIZE=normalize_values,
            APPLY_NORM=apply_norm,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
        )
    return out
