# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU RMSNorm used by Inkling 2."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.inkling2.norm import rms_norm

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")


def test_rms_norm_matches_pytorch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2304, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(2304, dtype=torch.bfloat16, device="npu")
    eps = 1e-6

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(torch.bfloat16)

    out = rms_norm(x, weight, eps)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.01, f"rms_norm max_err={err}"


@pytest.mark.parametrize("with_tau", [True, False])
def test_l2_clip_triton_matches_torch(with_tau):
    from vllm_ascend.models.inkling2.attention import l2_clip
    from vllm_ascend.models.inkling2.norm import l2_clip_triton

    torch.manual_seed(0)
    tokens, heads, width = 5, 4, 16
    q_scale = (torch.rand(heads, width) * 2).to(torch.bfloat16).to("npu")
    x = (torch.randn(tokens, heads, width) * 4).to(torch.bfloat16).to("npu")
    tau = 1 + torch.rand(tokens, device="npu") if with_tau else None

    out = l2_clip_triton(x, q_scale, 22.6, tau).cpu()
    ref = l2_clip(x, q_scale, 22.6, tau).cpu()
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


def test_l2_clip_triton_without_scale():
    from vllm_ascend.models.inkling2.attention import l2_clip
    from vllm_ascend.models.inkling2.norm import l2_clip_triton

    torch.manual_seed(0)
    x = (torch.randn(3, 2, 32) * 4).to(torch.bfloat16).to("npu")
    torch.testing.assert_close(l2_clip_triton(x, None, 22.6).cpu(), l2_clip(x.cpu(), None, 22.6), atol=0, rtol=1e-2)
