# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 attention on NPU in plain PyTorch.

Upstream counterparts, all CUDA/Triton kernels: ``_l2_clip``
(``ops/norm.py``), ``qr_clip_rel_tokens`` and the K/V part of
``_kv_conv_tile`` (``ops/sconv.py``), ``_gate_sink_kernel`` and
``_log_scaling_tau_kernel`` (``ops/attention.py``), and the attention itself
(``rel_attention``, the FA4 kernel). None of that code runs on NPU: each piece
is re-implemented here from upstream's formulas and checked against the
reference implementations of the upstream tests.

What differs numerically from upstream:

- attention materializes the whole score matrix and takes one softmax, where
  upstream runs a flash-attention kernel with an online softmax
- upstream's approximate PTX ``rsqrt``, ``sigmoid`` and ``tanh`` are ``torch``
  functions here"""

import torch


def l2_clip(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    threshold: float,
    tau: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-head L2 clipping of ``x * scale`` (upstream ``_l2_clip``, and the
    key clip of ``_kv_conv_tile`` with ``scale=None``).

    x: [T, H, D] bf16; scale: [H, D] bf16 weights (upstream ``q_l2_clip.scale``)
    tau: fp32 [T] per-token log-scaling (full layers)
    Returns bf16: the clipped values, rounded, then scaled by ``tau``.
    """
    x = x.float() if scale is None else x.float() * scale.float()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    y = x * torch.clamp(threshold / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(torch.bfloat16).float() * tau.reshape(-1, 1, 1)
    return y.to(torch.bfloat16)


def relative_bias(
    r: torch.Tensor,
    rel_proj: torch.Tensor,
    positions: torch.Tensor,
    kv_length: int,
    is_local: bool,
) -> torch.Tensor:
    """Additive relative position bias (upstream ``qr_clip_rel_tokens``).

    r: [T, H, d_rel] bf16 clipped relative queries
    rel_proj: [d_rel, extent] bf16 relative projection
    positions: [T] absolute positions of the query tokens
    kv_length: number of key/value positions (cache length after this step)
    is_local: a sliding-window layer (keys beyond the extent are masked)

    Returns bf16 [T, H, kv_length]: ``bias[t, h, s] = r[t, h] . rel_proj[:, positions[t] - s]``,
    -inf for keys after the query and (local layers only) for distances beyond
    the extent, 0 for those distances on full layers.
    """
    tokens, heads, _ = r.shape
    extent = rel_proj.shape[1]
    logits = torch.einsum("thd,de->the", r.float(), rel_proj.float())
    logits = logits.to(torch.bfloat16)
    dist = positions.view(tokens, 1) - torch.arange(kv_length, device=r.device)
    index = dist.clamp(0, extent - 1).unsqueeze(1).expand(tokens, heads, kv_length)
    bias = logits.gather(2, index)
    if not is_local:
        # Full layers keep attending beyond the extent, without a bias.
        bias.masked_fill_((dist >= extent)[:, None, :], 0.0)
        return bias.masked_fill((dist < 0)[:, None, :], float("-inf"))
    # A local layer's extent is its window: only keys after the query within it.
    in_window = (dist >= 0) & (dist < extent)
    return bias.masked_fill(~in_window[:, None, :], float("-inf"))


def bmm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matmul of bf16 operands with an fp32 output: the tensor cores'
    fp32 accumulation on NPU (``out_dtype``), elsewhere the fp32 matmul of the
    same (exactly representable) values."""
    try:
        return torch.bmm(a, b, out_dtype=torch.float32)
    except (TypeError, NotImplementedError):  # no out_dtype on this backend
        return torch.bmm(a.float(), b.float())


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention with an additive bias.

    q: [T, H, D] bf16; k, v: [H_kv, S, D] bf16 (the KV state's layout);
    bias: [T, H, S] bf16 (-inf masked)
    Returns fp32 ``out`` [T, H, D] and the natural-log ``lse`` [T, H].
    """
    group = q.shape[1] // k.shape[0]
    if group > 1:
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
    scores = bmm_fp32(q.transpose(0, 1), k.transpose(1, 2))  # [H, T, S]
    scores = scores * scale + bias.permute(1, 0, 2).float()
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)  # [T, H]
    out = torch.bmm(torch.softmax(scores, dim=-1), v.float())
    return out.transpose(0, 1), lse


def gate_sink(out: torch.Tensor, lse: torch.Tensor, gate: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """bf16 ``out * sigmoid(gate) * sigmoid(lse - sink)`` (upstream ``gate_sink``).

    out: fp32 [T, H, D]; lse: fp32 [T, H]; gate: [T, H, D] bf16; sinks: [T, H] bf16
    """
    result = out * torch.sigmoid(gate.float())
    result = result * torch.sigmoid(lse - sinks.float()).unsqueeze(-1)
    return result.to(torch.bfloat16)
