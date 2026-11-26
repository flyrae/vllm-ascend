# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the attention-residual mixing on NPU and for the hidden
stream's short convolution fused with it (``conv_mix``), against the upstream
references (``tests/models/inkling2/test_core_ops.py``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.attn_res import attn_res
from vllm_ascend.models.inkling2.layers import AscendInkling2RMSNorm, AscendInkling2ShortConv

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

DIM = 2304


@pytest.mark.parametrize("num_stack,num_q,with_addend", [(1, 1, False), (5, 3, True)])
@pytest.mark.parametrize("normalize", [True, False])
# Small batches keep rows in registers; large ones loop over D chunks.
@pytest.mark.parametrize("tokens", [29, 300])
def test_attn_res_matches_softmax_mix_then_norm(num_stack, num_q, with_addend, normalize, tokens):
    torch.manual_seed(0)
    stack = torch.randn(num_stack, tokens, DIM).to(torch.bfloat16)
    last = torch.randn(tokens, DIM).to(torch.bfloat16)
    addend = torch.randn(tokens, DIM).to(torch.bfloat16) if with_addend else None
    queries = torch.randn(num_q, DIM) * DIM**-0.5
    weight = torch.randn(num_q, DIM).to(torch.bfloat16)

    out = attn_res(
        stack.to("npu"),
        last.to("npu"),
        addend.to("npu") if addend is not None else None,
        queries.to("npu"),
        weight.to("npu"),
        1e-6,
        normalize,
    ).cpu()
    ref = reference.attn_res(stack, last, addend, queries, weight, normalize, 1e-6)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize("num_stack,with_addend", [(1, False), (3, True)])
@pytest.mark.parametrize("tokens", [29, 300])
def test_attn_res_without_norm_returns_fp32_mix(num_stack, with_addend, tokens):
    """Without a norm weight: the fp32 softmax mix (the loop's sublayer inputs
    before their controllers, and its two-block message reads)."""
    torch.manual_seed(0)
    stack = torch.randn(num_stack, tokens, DIM).to(torch.bfloat16)
    last = torch.randn(tokens, DIM).to(torch.bfloat16)
    addend = torch.randn(tokens, DIM).to(torch.bfloat16) if with_addend else None
    queries = torch.randn(4, DIM) * DIM**-0.5

    out = attn_res(
        stack.to("npu"),
        last.to("npu"),
        addend.to("npu") if addend is not None else None,
        queries.to("npu"),
        None,
        0.0,
        True,
    ).cpu()
    ref = reference.attn_res(stack, last, addend, queries, None, True)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


# Chunked prefill and decode steps of one sequence.
STEPS = [3, 150, 2, 140, 1, 1]


@pytest.mark.parametrize(
    "num_stack,num_q,with_last,with_residual",
    [(1, 1, True, False), (2, 1, False, False), (3, 1, False, True), (8, 3, False, True)],
)
@pytest.mark.parametrize("with_norm", [True, False])
def test_conv_mix_matches_conv_then_attn_res(num_stack, num_q, with_last, with_residual, with_norm):
    """The short conv (with the residual update) followed by the AttnRes mix
    over chunked steps equals the reference over the whole sequence."""
    torch.manual_seed(0)
    conv = AscendInkling2ShortConv(DIM, 4, normalize_values=True).to("npu")
    with torch.no_grad():
        conv.weight.normal_()
    norm = AscendInkling2RMSNorm(DIM, 1e-6, num_inputs=num_q).to("npu")
    with torch.no_grad():
        norm.weight.normal_()
    queries = (torch.randn(num_q, DIM) * DIM**-0.5).to("npu")
    total = sum(STEPS)
    x = torch.randn(total, DIM).to(torch.bfloat16)
    addend = torch.randn(total, DIM).to(torch.bfloat16)
    residual = torch.randn(total, DIM).to(torch.bfloat16)

    conv_out = reference.short_conv(x, conv.weight.squeeze(1).cpu())
    if with_residual:
        conv_out = reference.bf16(residual.float() + addend.float() + conv_out.float())

    state = torch.zeros(3, DIM, dtype=torch.bfloat16, device="npu")
    start = 0
    for step in STEPS:
        rows = slice(start, start + step)
        xs, last = x[rows].to("npu"), None
        stack = torch.randn(num_stack, step, DIM).to(torch.bfloat16)
        if with_last:
            last = torch.randn(step, DIM).to(torch.bfloat16).to("npu")
        out = residual[rows].to("npu").clone() if with_residual else None
        mixes, out = conv.conv_mix(
            xs,
            stack.to("npu"),
            last,
            queries,
            norm if with_norm else None,
            state,
            addend=addend[rows].to("npu") if with_residual else None,
            residual=residual[rows].to("npu") if with_residual else None,
            out=out,
        )

        torch.testing.assert_close(out.cpu(), conv_out[rows], atol=0, rtol=1e-2)
        # With ``last``, the sources are [*stack, last + out]; else [*stack, out].
        ref = reference.attn_res(
            stack,
            last.cpu() if with_last else conv_out[rows],
            conv_out[rows] if with_last else None,
            queries.cpu(),
            norm.weight.cpu() if with_norm else None,
            True,
            norm.eps,
        )
        torch.testing.assert_close(mixes.cpu(), ref, atol=1e-2, rtol=1e-2)
        start += step
