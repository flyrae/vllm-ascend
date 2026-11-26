# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the BF16 MoE forward on NPU, against the upstream reference
(``tests/models/inkling2/test_moe.py``): the routing
(``_torch_select_experts``) and a per-token loop over the experts
(``_moe_reference``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.moe import moe_forward, moe_route

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

ROUTE_SCALE = 8.0
GATE_CAP = 4.0
UP_CAP = 25.0


def _gate(experts, hidden):
    """Router weight and bias (CPU)."""
    torch.manual_seed(0)
    return dict(
        gate_weight=(torch.randn(experts, hidden) * 0.02).to(torch.bfloat16),
        gate_bias=torch.randn(experts) * 0.02,
    )


def _experts(experts, hidden, intermediate, device):
    """BF16 expert projections; on the NPU, which generates them much faster
    than the CPU for the real shapes."""
    torch.manual_seed(0)
    w13 = (torch.randn(experts, 2 * intermediate, hidden, device=device) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(experts, hidden, intermediate, device=device) * 0.02).to(torch.bfloat16)
    return dict(w13=w13, w2=w2)


@pytest.mark.parametrize("tokens", [1, 5, 200])
def test_moe_route_matches_reference(tokens):
    torch.manual_seed(0)
    experts, hidden, topk = 128, 2304, 8
    x = (torch.randn(tokens, hidden) * 0.02).to(torch.bfloat16)
    weights = _gate(experts, hidden)

    perm, gammas, offsets, src2dst = moe_route(
        x.to("npu"),
        weights["gate_weight"].to("npu"),
        weights["gate_bias"].to("npu"),
        topk,
        ROUTE_SCALE,
    )
    perm, gammas, offsets, src2dst = (t.cpu() for t in (perm, gammas, offsets, src2dst))
    ref_weights, ref_ids = reference.select_experts(
        x.float() @ weights["gate_weight"].float().t(), weights["gate_bias"], topk, ROUTE_SCALE
    )

    # Rows are sorted by expert: the offsets count them and perm maps each row
    # to its token's slot.
    assert offsets.shape == (experts + 1,) and int(offsets[0]) == 0
    assert int(offsets[-1]) == tokens * topk
    assert torch.equal(src2dst, perm // topk)
    ids = ref_ids.reshape(-1)[perm.long()]
    assert (ids[1:] >= ids[:-1]).all()
    assert torch.equal(offsets.diff(), torch.bincount(ids, minlength=experts).to(torch.int32))

    # A token's experts may come in another order: compare each one's weight.
    dense = torch.zeros(tokens, experts)
    dense.scatter_(1, ref_ids, ref_weights)
    got = torch.zeros(tokens, experts)
    got[src2dst.long(), ids] = gammas
    torch.testing.assert_close(got, dense, atol=0, rtol=1e-5)
    # The renormalized sigmoid weights sum to the route scale.
    torch.testing.assert_close(dense.sum(-1), torch.full((tokens,), ROUTE_SCALE), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("tokens", [1, 5, 33])
def test_moe_forward_matches_reference(tokens):
    torch.manual_seed(0)
    experts, hidden, intermediate, topk = 16, 256, 128, 8
    x = torch.randn(tokens, hidden).to(torch.bfloat16)
    weights = {**_gate(experts, hidden), **_experts(experts, hidden, intermediate, "cpu")}

    out = moe_forward(
        x.to("npu"),
        weights["gate_weight"].to("npu"),
        weights["gate_bias"].to("npu"),
        weights["w13"].to("npu"),
        weights["w2"].to("npu"),
        topk,
        ROUTE_SCALE,
        GATE_CAP,
        UP_CAP,
    ).cpu()
    ref = reference.moe(
        x,
        weights["gate_weight"],
        weights["gate_bias"],
        weights["w13"],
        weights["w2"],
        topk,
        ROUTE_SCALE,
        GATE_CAP,
        UP_CAP,
    )

    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("tokens", [1, 5])
def test_moe_routing_weights_follow_the_sorted_rows(tokens):
    """Each expert-sorted row gets its own token's routing weight (applied
    before the down projection, in fp32).  The CANN path then matches the
    per-token reference to GEMM rounding, far inside the 0.4% that a bf16
    weight applied after the down projection costs."""
    torch.manual_seed(0)
    experts, hidden, intermediate, topk = 16, 256, 128, 8
    x = (torch.randn(tokens, hidden) * 0.5).to(torch.bfloat16)
    weights = {**_gate(experts, hidden), **_experts(experts, hidden, intermediate, "npu")}

    out = moe_forward(
        x.to("npu"),
        weights["gate_weight"].to("npu"),
        weights["gate_bias"].to("npu"),
        weights["w13"],
        weights["w2"],
        topk,
        ROUTE_SCALE,
        GATE_CAP,
        UP_CAP,
    ).cpu()
    ref = reference.moe(
        x,
        weights["gate_weight"],
        weights["gate_bias"],
        weights["w13"].cpu(),
        weights["w2"].cpu(),
        topk,
        ROUTE_SCALE,
        GATE_CAP,
        UP_CAP,
    )

    scale = ref.abs().max().item()
    torch.testing.assert_close(out, ref, atol=2e-3 * scale, rtol=2e-3)


def test_moe_forward_real_shapes():
    """The Inkling-2-Mini shapes (128 experts, top-8, hidden 2304,
    intermediate 1152): finite output of the routing weights' scale."""
    torch.manual_seed(0)
    experts, hidden, intermediate, topk = 128, 2304, 1152, 8
    x = (torch.randn(33, hidden) * 0.5).to(torch.bfloat16)
    gate, experts = _gate(experts, hidden), _experts(experts, hidden, intermediate, "npu")

    out = moe_forward(
        x.to("npu"),
        gate["gate_weight"].to("npu"),
        gate["gate_bias"].to("npu"),
        experts["w13"],
        experts["w2"],
        topk,
        ROUTE_SCALE,
        GATE_CAP,
        UP_CAP,
    ).cpu()

    assert out.shape == (33, hidden) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("rows", [1, 8, 264])
def test_moe_swiglu_fused_matches_torch(rows):
    """The one-launch SwiGLU equals the PyTorch reference (fp32 math, bf16
    rounding) up to sigmoid/tanh implementation differences."""
    from vllm_ascend.models.inkling2.moe import moe_swiglu, moe_swiglu_fused

    torch.manual_seed(0)
    intermediate = 1152
    gate_up = (torch.randn(rows, 2 * intermediate, device="npu") * 2).to(torch.bfloat16)
    gammas = torch.rand(rows, device="npu") * 8.0

    out = moe_swiglu_fused(gate_up, gammas, GATE_CAP, UP_CAP).cpu()
    ref = moe_swiglu(gate_up, gammas, GATE_CAP, UP_CAP).cpu()

    scale = ref.float().abs().max().item()
    torch.testing.assert_close(out, ref, atol=1e-3 * scale, rtol=1e-3)
