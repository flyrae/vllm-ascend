# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LM head of a few decode rows on NPU, against the fp32
GEMM of the same bf16 values."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.inkling2.lm_head import lm_head_fp32

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")


@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("hidden,vocab", [(2304, 1000), (64, 300)])
def test_lm_head_matches_fp32_gemm(rows, hidden, vocab):
    """fp32 accumulation of exact bf16 products: the fp32 GEMM's result up to
    the order of the sum.  The second shape is the test models' (a K that is
    not a multiple of the block)."""
    torch.manual_seed(0)
    x = (torch.randn(rows, hidden) * 0.5).to(torch.bfloat16)
    weight = (torch.randn(vocab, hidden) * 0.02).to(torch.bfloat16)

    out = lm_head_fp32(x.to("npu"), weight.to("npu")).cpu()

    assert out.shape == (rows, vocab) and out.dtype == torch.float32
    reference = torch.nn.functional.linear(x.float(), weight.float())
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-4)


def test_lm_head_rejects_more_rows_than_a_decode_batch():
    from vllm_ascend.models.inkling2.lm_head import MAX_ROWS

    x = torch.zeros(MAX_ROWS + 1, 64, dtype=torch.bfloat16, device="npu")
    with pytest.raises(AssertionError):
        lm_head_fp32(x, torch.zeros(8, 64, dtype=torch.bfloat16, device="npu"))
