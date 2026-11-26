# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Inkling 2 attention pieces on NPU (L2 clipping,
relative position bias, softmax attention, the gate/sink epilogue), against
the upstream references (``tests/models/inkling2/test_core_ops.py`` and
``test_attention.py``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.attention import (
    attention,
    bmm_fp32,
    gate_sink,
    l2_clip,
    relative_bias,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

HEAD_DIM = 128
D_REL = 16
EXTENT = 1024
QK_CLIP = 22.6
REL_CLIP = 8.0


@pytest.mark.parametrize("with_tau", [True, False])
def test_l2_clip_rounds_before_tau(with_tau):
    torch.manual_seed(0)
    tokens, heads = 33, 12
    # q and r are strided column slices of the fused projection.
    proj = (torch.randn(tokens, heads * (HEAD_DIM + D_REL) + 64) * 4).to(torch.bfloat16)
    q = proj[:, : heads * HEAD_DIM].view(tokens, heads, HEAD_DIM)
    r = proj[:, heads * HEAD_DIM : heads * (HEAD_DIM + D_REL)].view(tokens, heads, D_REL)
    q_scale = (torch.rand(heads, HEAD_DIM) * 2).to(torch.bfloat16)
    r_scale = (torch.rand(heads, D_REL) * 2).to(torch.bfloat16)
    tau = 1 + torch.rand(tokens) if with_tau else None

    def to_npu(x):
        return x.to("npu") if x is not None else None

    out_q = l2_clip(to_npu(q), to_npu(q_scale), QK_CLIP, to_npu(tau)).cpu()
    out_r = l2_clip(to_npu(r), to_npu(r_scale), REL_CLIP, to_npu(tau)).cpu()
    torch.testing.assert_close(out_q, reference.clip_scale_tau(q, q_scale, QK_CLIP, tau), atol=0, rtol=1e-2)
    torch.testing.assert_close(out_r, reference.clip_scale_tau(r, r_scale, REL_CLIP, tau), atol=0, rtol=1e-2)


def test_l2_clip_without_scale_clips_the_keys():
    torch.manual_seed(0)
    k = (torch.randn(5, 8, HEAD_DIM) * 4).to(torch.bfloat16)
    out = l2_clip(k.to("npu"), None, QK_CLIP).cpu()
    torch.testing.assert_close(out, reference.clip_scale_tau(k, torch.ones(HEAD_DIM), QK_CLIP), atol=0, rtol=1e-2)


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("tokens,kv_len", [(300, 300), (64, 300), (1, 300)])
def test_relative_bias_matches_reference(is_local, tokens, kv_len):
    torch.manual_seed(0)
    heads = 12
    r = (torch.randn(tokens, heads, D_REL) * 4).to(torch.bfloat16)
    rel_proj = torch.randn(D_REL, EXTENT).to(torch.bfloat16)
    positions = torch.arange(kv_len - tokens, kv_len)
    # Local layers read a window of EXTENT keys, i.e. distances up to EXTENT - 1.
    window_left = EXTENT - 1 if is_local else None

    out = relative_bias(r.to("npu"), rel_proj.to("npu"), positions.to("npu"), kv_len, is_local)
    rel_logits = torch.einsum("thd,de->the", r, rel_proj)
    ref = reference.relative_bias(rel_logits, kv_len, kv_len - tokens, EXTENT, window_left)

    assert out.shape == (tokens, heads, kv_len) and out.dtype == torch.bfloat16
    # A 1-ulp difference in r shifts a row's logits by about ulp * |rel_proj|.
    torch.testing.assert_close(out.permute(1, 0, 2).float().cpu(), ref, atol=0.25, rtol=1e-2)


@pytest.mark.parametrize("window_left", [None, EXTENT - 1])
@pytest.mark.parametrize("tokens,kv_len", [(300, 300), (64, 300), (1, 300), (1, 1)])
def test_attention_matches_reference(window_left, tokens, kv_len):
    """Prefill, chunked prefill, decode, and the first token: q/k as
    L2-normalized in the model, GQA 24:8, with the sinks of the model."""
    torch.manual_seed(0)
    heads, kv_heads = 24, 8
    scale = HEAD_DIM**-0.5
    q = (torch.nn.functional.normalize(torch.randn(tokens, heads, HEAD_DIM), dim=-1) * 3).to(torch.bfloat16)
    k = (torch.nn.functional.normalize(torch.randn(kv_heads, kv_len, HEAD_DIM), dim=-1) * 2).to(torch.bfloat16)
    v = torch.randn(kv_heads, kv_len, HEAD_DIM).to(torch.bfloat16)
    rel_logits = torch.randn(tokens, heads, EXTENT).to(torch.bfloat16)
    gate = torch.randn(tokens, heads, HEAD_DIM).to(torch.bfloat16)
    sinks = torch.randn(tokens, heads).to(torch.bfloat16)

    bias = reference.relative_bias(rel_logits, kv_len, kv_len - tokens, EXTENT, window_left)
    ref_out, ref_lse = reference.attention(q, k, v, bias, scale, sinks)
    ref = reference.bf16(ref_out * torch.sigmoid(gate.float()))

    out, lse = attention(q.to("npu"), k.to("npu"), v.to("npu"), bias.permute(1, 0, 2).to("npu"), scale)
    out = gate_sink(out, lse, gate.to("npu"), sinks.to("npu")).cpu()

    assert lse.shape == (tokens, heads) and lse.dtype == torch.float32
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_bmm_fp32_of_transposed_views():
    """The bf16 batched matmul with an fp32 output (the q k^T of strided
    views) equals the fp32 matmul of the same values."""
    torch.manual_seed(0)
    tokens, length, heads = 37, 300, 24
    q = torch.randn(tokens, heads, HEAD_DIM).to(torch.bfloat16)
    k = torch.randn(heads, length, HEAD_DIM).to(torch.bfloat16)
    out = bmm_fp32(q.to("npu").transpose(0, 1), k.to("npu").transpose(1, 2)).cpu()
    ref = torch.einsum("thd,hsd->hts", q.float(), k.float())
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("is_local", [False, True])
def test_relative_bias_beyond_the_extent(is_local):
    """Full layers keep the keys farther than the extent, with a zero bias;
    local layers (the extent is their window) mask them."""
    torch.manual_seed(0)
    extent, tokens, heads = 16, 37, 4
    r = (torch.randn(tokens, heads, D_REL) * 4).to(torch.bfloat16)
    rel_proj = torch.randn(D_REL, extent).to(torch.bfloat16)
    positions = torch.arange(tokens)

    out = relative_bias(r.to("npu"), rel_proj.to("npu"), positions.to("npu"), tokens, is_local).cpu()
    rel_logits = torch.einsum("thd,de->the", r, rel_proj)
    ref = reference.relative_bias(rel_logits, tokens, 0, extent, extent - 1 if is_local else None)

    # Keys farther back than the extent: distance = query position - key position.
    positions = torch.arange(tokens)
    far = ((positions[:, None] - positions[None, :]) >= extent).unsqueeze(1).expand(-1, heads, -1)
    assert far.any()
    if is_local:
        assert torch.isinf(out.float()[far]).all()
    else:
        assert (out.float()[far] == 0).all()
    torch.testing.assert_close(out.permute(1, 0, 2).float(), ref, atol=0.25, rtol=1e-2)


@pytest.mark.parametrize("is_local", [False, True])
def test_decode_relative_bias_matches_reference(is_local):
    """The fixed-shape bias of a captured decode graph equals ``relative_bias``
    over each request's keys, and masks every key past them (and every key of
    a padding request, whose length is zero)."""
    from vllm_ascend.models.inkling2.attention_backend import MASK_VALUE, _decode_relative_bias

    torch.manual_seed(0)
    extent, heads = 64, 4
    kv_lens = [3, 70, 0, 1, 64]  # the second request reaches past the extent
    kv_cap = 96
    batch = len(kv_lens)
    r = (torch.randn(batch, heads, D_REL) * 4).to(torch.bfloat16)
    rel_proj = torch.randn(D_REL, extent).to(torch.bfloat16)
    positions = torch.tensor([max(length - 1, 0) for length in kv_lens])
    masked = torch.tensor(MASK_VALUE, dtype=torch.bfloat16)

    out = _decode_relative_bias(
        r.to("npu"),
        rel_proj.t().contiguous().to("npu"),
        positions.to("npu"),
        torch.tensor(kv_lens, dtype=torch.int32, device="npu"),
        kv_cap,
        is_local,
    ).cpu()
    assert out.shape == (batch, heads, 1, kv_cap)
    for b, kv_len in enumerate(kv_lens):
        got = out[b, :, 0]
        assert (got[:, kv_len:] == masked).all()
        if kv_len == 0:
            assert (got == masked).all()
            continue
        ref = relative_bias(
            r[b : b + 1].to("npu"),
            rel_proj.to("npu"),
            positions[b : b + 1].to("npu"),
            kv_len,
            is_local,
        ).cpu()[0]
        # A local layer masks the keys farther back than its window; a full
        # layer keeps them with a zero bias.
        finite = ~torch.isinf(ref.float())
        if is_local:
            assert (got[:, :kv_len][~finite] == masked).all()
        else:
            assert finite.all()
        torch.testing.assert_close(got[:, :kv_len].float()[finite], ref.float()[finite], atol=0, rtol=1e-2)


def test_gate_sink_fused_matches_torch():
    from vllm_ascend.models.inkling2.attention_backend import _gate_sink_fused

    torch.manual_seed(0)
    tokens, heads = 5, 4
    out = torch.randn(tokens, heads, HEAD_DIM, device="npu")
    lse = torch.randn(tokens, heads, device="npu")
    gate = (torch.randn(tokens, heads, HEAD_DIM) * 2).to(torch.bfloat16).to("npu")
    sinks = (torch.randn(tokens, heads) * 2).to(torch.bfloat16).to("npu")

    torch.testing.assert_close(
        _gate_sink_fused(out, lse, gate, sinks).cpu(),
        gate_sink(out.cpu(), lse.cpu(), gate.cpu(), sinks.cpu()),
        atol=0,
        rtol=1e-2,
    )
