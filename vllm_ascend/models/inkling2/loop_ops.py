# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loop message-controller helpers for Inkling 2 on NPU.

Upstream counterpart: ``vllm/models/inkling2/nvidia/ops/loop.py``, with the
same math. ``_pair_rms_norm_kernel`` keeps upstream's code except for the
``rsqrt`` it calls (:mod:`.norm`'s ``tl.rsqrt`` instead of inline PTX);
``_blend_rms_norm_kernel`` also takes ``tl.math.tanh`` in place of
``tldevice.tanh``. The wrappers check that their inputs share one device."""

import torch
import triton
import triton.language as tl

from .norm import rsqrt


def _same_device(*tensors: torch.Tensor) -> bool:
    devices = {tensor.device for tensor in tensors}
    return len(devices) == 1


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0)."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    # A tensor of another device would hand the kernel an invalid address.
    assert _same_device(dest, message), "pair_rms_norm inputs must share one device"
    out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=torch.bfloat16)
    if num_tokens:
        block = triton.next_power_of_2(d)
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=block,
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + row * D + cols, mask, other=0)
    gain = tl.math.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
    forget = tl.math.tanh(tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0))
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert _same_device(dest, message, gain_forget, norm_weight), "blend_rms_norm inputs must share one device"
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        block = triton.next_power_of_2(d)
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=block,
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out
