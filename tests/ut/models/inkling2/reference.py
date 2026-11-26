# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference implementations of the Inkling 2 ops in plain PyTorch.

Transcribed from the reference code of the upstream vLLM tests, function by
function (not verbatim copies: variables are renamed and the code
restructured, and the CUDA-only parts are left out). The NPU implementations
are checked against them:

- ``tests/models/inkling2/test_core_ops.py``: ``_clip``, the ``ref_clip`` of
  ``test_qr_clip_rel_rounds_before_tau``, ``_dense_conv``, and the inline
  references of ``test_attn_res_matches_softmax_mix_then_norm``,
  ``test_attn_res_without_norm_returns_fp32_mix``,
  ``test_loop_norm_preserves_stacked_reference_and_graph`` and
  ``test_loop_blend_matches_per_stream_controllers`` (the controller itself is
  upstream's ``Inkling2MessageController.forward``)
- ``tests/models/inkling2/test_attention.py``: ``_ref_attention``; the sinks
  follow ``eager_attention_forward`` of the transformers implementation
- ``tests/models/inkling2/test_moe.py``: ``dequant_nvfp4``, ``dequant_mxfp8``
  (with the outer scale of ``_moe_reference`` applied),
  ``_torch_select_experts`` and ``_moe_reference`` (without its activation
  quantization)

The references run wherever a test puts them (CPU, to keep float8 tensors away
from the NPU) and round to bf16 where the upstream kernels do.
"""

import torch

# e2m1 codes: sign in bit 3, two exponent bits, one mantissa bit.
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMSNorm, rounded once to bf16 (upstream ``rms_norm``)."""
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return bf16(ref * weight.float())


def dequant_nvfp4(
    values: torch.Tensor, scales: torch.Tensor, amax: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """[E, N, K/2] e2m1 pairs (low nibble first) and [E, N, K/16] e4m3 block
    scales to the fp32 weights ``e2m1 * scale * (amax / denominator)``."""
    lut = torch.tensor(E2M1)
    unpacked = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], dim=-1).flatten(-2)
    scaled = unpacked * scales.float().repeat_interleave(16, -1)
    return scaled * (amax / denominator).view(-1, 1, 1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K] e4m3 weights and [E, N, K/32] ue8m0 scales to fp32 weights."""
    return values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def clip(x: torch.Tensor, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    """Per-head L2 clipping (upstream ``_clip``)."""
    norm = x.norm(dim=-1, keepdim=True)
    return x * (threshold / (norm + eps)).clamp(max=1.0)


def clip_scale_tau(
    x: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x * scale`` clipped per head, rounded to bf16, then scaled by the
    per-token ``tau`` and rounded again (upstream ``ref_clip``)."""
    y = bf16(clip(x.float() * scale.float(), threshold))
    if tau is None:
        return y
    return bf16(y.float() * tau[:, None, None])


def short_conv(x: torch.Tensor, weight: torch.Tensor, deltas: torch.Tensor | None = None):
    """Causal depthwise convolution over the whole sequence; x: (L, C).

    Static with ``weight`` (C, W) (oldest tap first), or dynamic with the
    (W, C) static taps (lag order) and per-token deltas (L, heads, W). The
    current input is added unweighted, then all is scaled (upstream
    ``_dense_conv``).
    """
    width = weight.shape[0] if deltas is not None else weight.shape[1]
    padded = torch.cat([x.new_zeros(width - 1, x.shape[1]), x]).float()
    acc = torch.zeros_like(x, dtype=torch.float32)
    for lag in range(width):
        value = padded[width - 1 - lag : width - 1 - lag + x.shape[0]]
        if deltas is None:
            w = weight[:, width - 1 - lag].float()
        else:
            head_dim = x.shape[1] // deltas.shape[1]
            w = weight[lag].float() + deltas[:, :, lag].float().repeat_interleave(head_dim, dim=1)
        acc += value * w
    scale = (width * (2 if deltas is not None else 1) + 1) ** -0.5
    return bf16((acc + x.float()) * scale)


def relative_bias(
    rel_logits: torch.Tensor, kv_len: int, q_offset: int, extent: int, window_left: int | None
) -> torch.Tensor:
    """The dense additive bias of upstream ``_ref_attention``: [H, T, kv_len].

    ``rel_logits``: (T, H, extent) bf16, the tokens starting at position
    ``q_offset`` of a sequence with ``kv_len`` keys. Zero outside
    ``0 <= distance < extent``; masked (-inf) for keys after the query and
    beyond ``window_left``.
    """
    q_len = rel_logits.shape[0]
    q_pos = torch.arange(q_offset, q_offset + q_len)
    dist = q_pos[:, None] - torch.arange(kv_len)[None]
    rel = rel_logits.float().transpose(0, 1)
    index = dist.clamp(0, extent - 1).expand(rel.shape[0], -1, -1)
    bias = rel.gather(2, index).masked_fill(~((dist >= 0) & (dist < extent)), 0)
    mask = dist < 0
    if window_left is not None:
        mask = mask | (dist > window_left)
    return bias.masked_fill(mask, float("-inf"))


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    sinks: torch.Tensor | None = None,
):
    """``softmax(q k^T * scale + bias (+ one sink logit with a zero value)) v``
    of upstream ``_ref_attention`` / ``eager_attention_forward``.

    q: (T, H, D); k, v: (H_kv, S, D); bias: (H, T, S); sinks: (T, H) or None.
    Returns the fp32 output (T, H, D), with the sinks absorbing probability
    mass, and the natural-log LSE (T, H) of the scores without the sinks.
    """
    group = q.shape[1] // k.shape[0]
    if group > 1:
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
    qk = torch.einsum("thd,hsd->hts", q.float(), k.float())
    scores = (qk * scale + bias).float()
    lse = scores.logsumexp(-1).t()
    if sinks is not None:
        scores = torch.cat([scores, sinks.float().t().unsqueeze(-1)], dim=-1)
    probs = scores.softmax(-1)
    if sinks is not None:
        probs = probs[..., :-1]
    out = torch.einsum("hts,hsd->thd", probs, v.float())
    return out, lse


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    weight: torch.Tensor | None,
    normalize: bool,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Softmax-weighted mix of ``[*stack, last (+ addend)]``, then the RMSNorm
    with ``weight``'s rows, or the fp32 mix without it (upstream reference)."""
    values = torch.cat([stack, last[None]]).float()
    if addend is not None:
        values[-1] = values[-1] + addend.float()
    inv_rms = torch.rsqrt(values.square().mean(-1) + 1e-6)
    probs = torch.einsum("ntd,qd,nt->qtn", values, queries, inv_rms).softmax(-1)
    if normalize:
        values = values * inv_rms[..., None]
    mix = torch.einsum("qtn,ntd->qtd", probs, values)
    if weight is None:
        return mix
    ref = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + norm_eps)
    return bf16(ref * weight.float()[:, None])


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 ``[rms(dest), rms(message)]`` of fp32 (S, T, D) inputs."""
    pair = torch.stack((dest, message), dim=2)
    return bf16(pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)).flatten(2)


def controller(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain: torch.Tensor,
    forget: torch.Tensor,
) -> torch.Tensor:
    """fp32 ``(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message`` with
    ``h = down(rms([dest, message]))`` of one stream ``(T, D)``
    (upstream ``Inkling2MessageController``)."""
    pair = torch.stack((dest, message), dim=1)
    pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
    hidden = bf16(pair.to(torch.bfloat16).flatten(1) @ down.t())
    gain_out = hidden.float() @ gain.float().t()
    forget_out = hidden.float() @ forget.float().t()
    return (1.0 - torch.tanh(forget_out)) * dest + torch.tanh(gain_out) * message


def select_experts(logits: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float):
    """Upstream ``_torch_select_experts``: selected by ``sigmoid(logit) + bias``,
    weights ``softmax(logsigmoid(logit)) * route_scale``."""
    ids = torch.topk(logits.sigmoid() + bias, topk, sorted=True).indices
    return torch.nn.functional.logsigmoid(logits.gather(1, ids)).softmax(-1) * route_scale, ids


def moe(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    route_scale: float,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Per-token loop over BF16 experts (upstream ``_moe_reference``, without
    activation quantization): interleaved (gate, up) columns, softcapped
    SwiGLU with the sigmoid of the raw gate, routing weight before the down
    projection, fp32 combine."""
    weights, ids = select_experts(x.float() @ gate_weight.float().t(), gate_bias, topk, route_scale)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(topk):
            expert = ids[token, slot]
            gate_up = bf16(x[token : token + 1] @ w13[expert].t()).float()
            gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            act = (
                gate_cap
                * torch.tanh(gate / gate_cap)
                * torch.sigmoid(gate)
                * up_cap
                * torch.tanh(up / up_cap)
                * weights[token, slot]
            )
            down = bf16(bf16(act) @ w2[expert].t()).float()
            out[token] += down[0]
    return bf16(out)
