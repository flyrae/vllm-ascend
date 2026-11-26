# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU Inkling 2 decoder layers against references built
from the upstream test references: the attention layer (local and full, with
its KV and conv state) and the loop controllers."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.configs import Inkling2ModelConfig
from vllm_ascend.models.inkling2.layers import LOOP_STREAMS, AscendInkling2DecoderLayer

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
D_REL = 8
EXTENT = 16
WIDTH = 4
KV_SIZE = KV_HEADS * HEAD_DIM


def _config(local_layer_ids: list[int]) -> Inkling2ModelConfig:
    return Inkling2ModelConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=12,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        d_rel=D_REL,
        rel_extent=EXTENT,
        local_layer_ids=local_layer_ids,
        sliding_window_size=EXTENT,
        sconv_kernel_size=WIDTH,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
        n_layers_per_attn_res_block=1,
        loop_config=dict(core_start=1, core_end=3, num_passes=2, gate_rank=8, is_varloop=False),
    )


def _layer(is_local: bool) -> AscendInkling2DecoderLayer:
    torch.manual_seed(0)
    layer_id = 1 if is_local else 2
    layer = AscendInkling2DecoderLayer(_config([1]), layer_id).to("npu")
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if param.dtype == torch.float32:
                param.normal_()
            elif param.dim() < 2:
                param.zero_()
            else:
                param.normal_(0.0, param.shape[-1] ** -0.5)
    return layer


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight.cpu())


def _conv_reference(projection: torch.Tensor, dynamic_input: torch.Tensor, sconv, history: list) -> torch.Tensor:
    """The K or V short convolution of one step, continuing the sequence:
    ``history`` holds the previous ``W - 1`` projections and dynamic inputs."""
    inputs = torch.cat((history[0], projection))
    deltas = None
    if sconv.dynamic:
        deltas = _linear(torch.cat((history[1], dynamic_input)), sconv.weight_proj)
        deltas = deltas.view(-1, KV_HEADS, WIDTH)
        taps = sconv.static_w
    else:
        taps = sconv.weight.squeeze(1)
    history[0] = inputs[-(WIDTH - 1) :]
    history[1] = torch.cat((history[1], dynamic_input))[-(WIDTH - 1) :]
    return reference.short_conv(inputs, taps.cpu(), deltas)[WIDTH - 1 :]


def _attention_reference(
    layer: AscendInkling2DecoderLayer,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    tau: torch.Tensor | None,
    kv: list,
    history_k: list,
    history_v: list,
) -> torch.Tensor:
    """One attention visit on the CPU from the reference ops. ``kv`` holds the
    sequence's keys and values (H_kv, L, D) and its length so far."""
    attn = layer.attn
    sizes = attn.split_sizes
    q_side, k_end = sum(sizes[:4]), sum(sizes[:5])
    tokens = x_q.shape[0]
    q, r, gate, sinks = _linear(x_q, attn.qkvr.weight[:q_side]).split(sizes[:4], dim=-1)
    k = _linear(x_k, attn.qkvr.weight[q_side:k_end])
    v = _linear(x_v, attn.qkvr.weight[k_end:])

    k = _conv_reference(k, x_k, attn.k_sconv, history_k)
    v = _conv_reference(v, x_v, attn.v_sconv, history_v)
    k = reference.bf16(reference.clip(k.float().view(tokens, KV_HEADS, HEAD_DIM), attn.qk_clip))
    v = v.float().view(tokens, KV_HEADS, HEAD_DIM)
    v = reference.bf16(v * torch.rsqrt(v.square().mean(-1, keepdim=True) + attn.eps))
    start = kv.pop()
    keys, values = kv
    keys[:, start : start + tokens] = k.transpose(0, 1)
    values[:, start : start + tokens] = v.transpose(0, 1)
    kv.append(start + tokens)
    length = start + tokens

    clipped_q = reference.clip_scale_tau(q.view(tokens, HEADS, HEAD_DIM), attn.q_scale.cpu(), attn.qk_clip, tau)
    clipped_r = reference.clip_scale_tau(r.view(tokens, HEADS, D_REL), attn.r_scale.cpu(), attn.rel_clip, tau)
    rel_logits = torch.einsum("thd,de->the", clipped_r, attn.rel_logits_proj.cpu())
    window_left = attn.sliding_window - 1 if attn.is_local else None
    bias = reference.relative_bias(rel_logits, length, start, attn.rel_extent, window_left)
    out, _ = reference.attention(
        clipped_q, keys[:, :length], values[:, :length], bias, attn.scaling, sinks.view(tokens, HEADS)
    )
    out = reference.bf16(out * torch.sigmoid(gate.view(tokens, HEADS, HEAD_DIM).float()))
    return reference.bf16(torch.nn.functional.linear(out.reshape(tokens, HEADS * HEAD_DIM), attn.wo_ud.weight.cpu()))


@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("separate_kv", [False, True])
@pytest.mark.parametrize("steps", [[1], [37], [5, 1, 20, 1]])
def test_attention_layer_matches_reference(is_local, separate_kv, steps):
    """Local and full attention over prefill and decode steps: the output and
    the K/V and conv state written. Full layers always take separate q/k/v
    inputs; local layers only in the loop passes after the first."""
    torch.manual_seed(0)
    layer = _layer(is_local)
    total = sum(steps)
    separate = separate_kv or not is_local
    inputs = [(torch.randn(total, HIDDEN) * 0.5).to(torch.bfloat16) for _ in range(3 if separate else 1)]
    tau = 1 + torch.rand(total) * 0.5
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16)  # noqa: E731
    kv = [zeros(KV_HEADS, total, HEAD_DIM), zeros(KV_HEADS, total, HEAD_DIM), 0]
    history_k = [zeros(WIDTH - 1, KV_SIZE), zeros(WIDTH - 1, HIDDEN)]
    history_v = [zeros(WIDTH - 1, KV_SIZE), zeros(WIDTH - 1, HIDDEN)]

    state = layer.new_state(total, "npu")
    start = 0
    for step in steps:
        rows = slice(start, start + step)
        xs = [x[rows].to("npu") for x in inputs]
        out = attn_forward(layer, xs, start, state, None if is_local else tau[rows].to("npu"))
        ref = _attention_reference(
            layer,
            *([x[rows] for x in inputs] if separate else [inputs[0][rows]] * 3),
            None if is_local else tau[rows],
            kv,
            history_k,
            history_v,
        )
        torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=2e-2)
        start += step

    # The K/V cache and the conv states hold the reference values.
    torch.testing.assert_close(state.key[:, :total].cpu(), kv[0], atol=0, rtol=1e-2)
    torch.testing.assert_close(state.value[:, :total].cpu(), kv[1], atol=0, rtol=1e-2)
    torch.testing.assert_close(state.conv["k"].cpu(), history_k[0], atol=0, rtol=0)
    torch.testing.assert_close(state.conv["v"].cpu(), history_v[0], atol=0, rtol=0)


def attn_forward(layer, xs, start, state, tau):
    positions = torch.arange(start, start + xs[0].shape[0], device="npu")
    if len(xs) == 1:
        return layer.attn(positions, xs[0], tau=tau, start=start, state=state)
    return layer.attn(positions, xs[0], xs[1], xs[2], tau, start=start, state=state)


@pytest.mark.parametrize("tokens", [1, 37])
def test_blend_matches_per_stream_controllers(tokens):
    """The stream-batched controllers (shared dest and norm weight, as a local
    layer's q/k/v) equal each stream's reference controller and RMSNorm."""
    torch.manual_seed(0)
    layer = _layer(True)
    dest = torch.randn(1, tokens, HIDDEN).expand(3, -1, -1)
    message = torch.randn(3, tokens, HIDDEN)
    norm_weight = torch.randn(1, HIDDEN).to(torch.bfloat16).expand(3, -1)

    out = layer.blend(
        slice(0, 3),
        dest.to("npu"),
        message.to("npu"),
        norm_weight.contiguous().to("npu"),
        1e-6,
    ).cpu()

    assert out.dtype == torch.bfloat16 and out.shape == (3, tokens, HIDDEN)
    down, gain_forget = layer.controller_down.cpu(), layer.controller_gain_forget.cpu()
    for stream in range(3):
        blended = reference.controller(
            dest[stream],
            message[stream],
            down[stream],
            gain_forget[stream, :HIDDEN],
            gain_forget[stream, HIDDEN:],
        )
        ref = reference.rms_norm(blended, norm_weight[stream].float(), 1e-6)
        torch.testing.assert_close(out[stream], ref, atol=2e-2, rtol=2e-2)


def test_only_core_layers_have_controllers():
    assert tuple(LOOP_STREAMS) == ("attn", "k", "v", "mlp")
    core = AscendInkling2DecoderLayer(_config([]), 2)
    assert core.controller_down.shape == (len(LOOP_STREAMS), 8, 2 * HIDDEN)
    assert core.controller_gain_forget.shape == (len(LOOP_STREAMS), 2 * HIDDEN, 8)
    assert not hasattr(AscendInkling2DecoderLayer(_config([]), 3), "controller_down")
