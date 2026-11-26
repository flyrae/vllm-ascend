# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NVFP4 and MXFP8 expert dequantization on NPU, against
the upstream reference (``tests/models/inkling2/test_moe.py``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.dequantize import (
    dequantize_mxfp8,
    dequantize_nvfp4,
    e2m1_to_float,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

# The checkpoint's outer scale denominator: 448 * 6.
DENOMINATOR = torch.tensor([2688.0])


def _nvfp4(experts: int, rows: int, width: int):
    values = torch.randint(0, 256, (experts, rows, width // 2), dtype=torch.uint8)
    scales = (torch.rand(experts, rows, width // 16) * 3 + 0.5).to(torch.float8_e4m3fn)
    return values, scales, torch.rand(experts) + 0.5


def _mxfp8(experts: int, rows: int, width: int):
    values = torch.randn(experts, rows, width).to(torch.float8_e4m3fn)
    return values, torch.randint(120, 135, (experts, rows, width // 32), dtype=torch.uint8)


def test_e2m1_codes():
    # e2m1: code 0 = +0.0, 1 = 0.5, 2 = 1.0, 3 = 1.5, 7 = 6.0; bit 3 is the sign.
    codes = torch.arange(16, dtype=torch.uint8)
    values = e2m1_to_float(codes.to("npu")).cpu()
    torch.testing.assert_close(values, torch.tensor(reference.E2M1), atol=0, rtol=0)


@pytest.mark.parametrize("experts,rows,width", [(4, 64, 128), (3, 2304, 2304), (2, 2304, 1152)])
def test_dequantize_nvfp4_matches_reference(experts, rows, width):
    torch.manual_seed(0)
    values, scales, amax = _nvfp4(experts, rows, width)
    # chunk=2 exercises several chunks; the float8 scales travel as raw bytes.
    out = dequantize_nvfp4(
        values.to("npu"),
        scales.view(torch.uint8).to("npu"),
        amax.to("npu"),
        DENOMINATOR.to("npu"),
        chunk=2,
    ).cpu()
    ref = reference.dequant_nvfp4(values, scales, amax, DENOMINATOR).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=0, rtol=2**-8)


def test_dequantize_nvfp4_writes_into_out():
    torch.manual_seed(0)
    values, scales, amax = _nvfp4(4, 32, 64)
    out = torch.empty(4, 32, 64, dtype=torch.bfloat16, device="npu")
    returned = dequantize_nvfp4(
        values.to("npu"),
        scales.view(torch.uint8).to("npu"),
        amax.to("npu"),
        DENOMINATOR.to("npu"),
        out=out,
    )
    assert returned.data_ptr() == out.data_ptr()


@pytest.mark.parametrize("experts,rows,width", [(4, 64, 128), (3, 2304, 2304), (2, 2304, 1152)])
def test_dequantize_mxfp8_is_exact(experts, rows, width):
    torch.manual_seed(0)
    values, scales = _mxfp8(experts, rows, width)
    out = dequantize_mxfp8(values.view(torch.uint8).to("npu"), scales.to("npu"), chunk=2).cpu()
    # Power-of-two scales and 3-bit mantissas: lossless in bf16.
    ref = reference.dequant_mxfp8(values, scales).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_e4m3_scales_all_exponents():
    """Every finite e4m3 byte decodes as torch's cast, subnormals included."""
    from vllm_ascend.models.inkling2.dequantize import e4m3_to_float

    bytes_ = torch.arange(256, dtype=torch.uint8)
    values = torch.arange(256, dtype=torch.uint8).to(torch.uint8).view(torch.float8_e4m3fn)
    decoded = e4m3_to_float(bytes_.to("npu")).cpu()
    ref = values.float()
    finite = torch.isfinite(ref)
    torch.testing.assert_close(decoded[finite], ref[finite], atol=0, rtol=0)
    assert torch.isnan(decoded[~finite]).all()
