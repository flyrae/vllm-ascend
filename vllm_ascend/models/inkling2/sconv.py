# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short causal convolutions for Inkling 2 on NPU (sequence form).

Upstream counterpart: ``vllm/models/inkling2/nvidia/ops/sconv.py``
(``_tap_sum``, ``_conv``, ``short_conv``), a Triton kernel over a paged
conv-state cache. This module is a PyTorch re-implementation of its formulas,
checked against the reference of the upstream tests. Static (fp32 accumulate,
oldest tap first)::

    y = (sum_k w[c, k] * x[t - (W - 1) + k] + x[t]) * (W + 1) ** -0.5

Dynamic (newest tap first, per-token per-head tap weights)::

    y = (sum_l (static_w[l, c] + dyn[t, head(c), l]) * x[t - l] + x[t])
        * (2 * W + 1) ** -0.5

``y`` is rounded to the input dtype. Upstream reads the taps of earlier steps
from its paged cache; here the caller keeps the last ``W - 1`` inputs of its
stream in a bf16 ``(W - 1, C)`` state tensor (zeros at the start of a
sequence) and passes it as ``state``: the state is advanced in place, so
successive calls continue one sequence (prefill chunks, then decode steps)."""

import torch


def _history(x: torch.Tensor, width: int, state: torch.Tensor | None) -> torch.Tensor:
    """The rows the ``T`` windows of ``x`` read: the state, or zeros (the taps
    before a sequence are zero), then ``x``."""
    if state is not None:
        return torch.cat((state, x))
    return torch.cat((x.new_zeros(width - 1, *x.shape[1:]), x))


def short_conv_static_seq(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
    """Static causal short convolution.

    x: [T, C] bf16 inputs
    weight: [C, W] bf16 taps, ``weight[c, k]`` multiplying ``x[t - (W - 1) + k]``
    state: [W - 1, C] bf16 inputs before ``x`` (updated in place), or None
    Returns: [T, C] bf16
    """
    width = weight.shape[1]
    history = _history(x, width, state)
    # history.unfold(0, W, 1)[t, c, k] = x[t - (W - 1) + k]
    taps = history.unfold(0, width, 1)
    y = (taps.float() * weight.float().unsqueeze(0)).sum(-1) + x.float()
    if state is not None:
        state.copy_(history[-(width - 1) :])
    return (y * (width + 1) ** -0.5).to(x.dtype)


def short_conv_dynamic_seq(
    x: torch.Tensor,
    static_w: torch.Tensor,
    deltas: torch.Tensor,
    num_heads: int,
    state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic causal short convolution.

    x: [T, C] bf16 inputs, C = num_heads * stream_width
    static_w: [W, C] bf16 taps, ``static_w[l, c]`` multiplying ``x[t - l]``
    deltas: [T, num_heads * W] bf16 per-token tap deltas, ``[t, head * W + l]``
        adding to the taps of ``x[t - l]`` in every channel of ``head``
    state: [W - 1, C] bf16 inputs before ``x`` (updated in place), or None
    Returns: [T, C] bf16
    """
    tokens, channels = x.shape
    width = static_w.shape[0]
    stream_width = channels // num_heads
    history = _history(x, width, state)
    taps = history.unfold(0, width, 1).view(tokens, num_heads, 1, stream_width, width)
    # Window index k holds x[t - (W - 1) + k]: reverse the lag-indexed weights.
    # [1, heads, 1, stream_width, W], broadcast over tokens and stream channels.
    static_taps = static_w.flip(0).view(width, num_heads, stream_width)
    static_taps = static_taps.permute(1, 2, 0).unsqueeze(1).unsqueeze(0)  # [1, heads, 1, sw, W]
    delta = deltas.view(tokens, num_heads, 1, 1, width).flip(-1)
    weights = static_taps.float() + delta.float()
    y = (taps.float() * weights).sum(-1).view(tokens, channels) + x.float()
    if state is not None:
        state.copy_(history[-(width - 1) :])
    return (y * (2 * width + 1) ** -0.5).to(x.dtype)


def dynamic_taps(x: torch.Tensor, weight_proj: torch.Tensor) -> torch.Tensor:
    """Per-token tap deltas ``[T, num_heads * W]`` projected from ``x`` ``[T, D]``."""
    return torch.nn.functional.linear(x, weight_proj)


__all__ = [
    "dynamic_taps",
    "short_conv_dynamic_seq",
    "short_conv_static_seq",
]
