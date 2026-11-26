# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantize Inkling 2 Mini's NVFP4 and MXFP8 expert weights to BF16.

Checkpoint layouts:
  NVFP4 (layers 0-19):  w13_weight U8 [E,N,K/2] (two e2m1 per byte, the first
                        one in the low nibble),
                        w13_weight_scale F8_E4M3 [E,N,K/16],
                        w13_weight_amax F32 [E],
                        w13_weight_outer_scale_denominator F32 [1]
  MXFP8 (layers 20-23): w13_weight F8_E4M3 [E,N,K],
                        w13_weight_scale U8 (ue8m0) [E,N,K/32]

Dequantization, as the upstream expert GEMMs scale the weights
(``ops/nvfp4.py``: ``(e2m1 * scale) * outer_scale``; ``ops/mxfp8.py``):
  NVFP4:  bf16 = e2m1_values * block_scale * (amax / outer_scale_denominator)
  MXFP8:  bf16 = e4m3_values * 2 ** (scale - 127)

The denominator is 448 * 6 in the checkpoint, so the outer scale is
``amax / 448 / 6``. MXFP8 is lossless in BF16 (power-of-two scales); NVFP4
weights are rounded to BF16 (relative error below 2 ** -9). Experts are
dequantized ``chunk`` at a time, into ``out`` when given, to bound the fp32
temporaries.

Upstream has no counterpart in the model code (its expert GEMMs read the
quantized formats directly); the formulas above are the ones those GEMMs
apply, and the unit tests compare the result with the reference
dequantization of upstream's tests (``tests/models/inkling2/test_moe.py``).
"""

import torch


def e2m1_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of uint8 e2m1 codes to float32 values."""
    # e2m1: 1 sign bit, 2 exponent bits, 1 mantissa bit
    sign = torch.where(codes & 8 != 0, -1.0, 1.0)
    exp = (codes >> 1) & 3
    man = codes & 1
    # value = (-1)^sign * 2^(exp-1) * (1 + man/2) for exp > 0
    # value = (-1)^sign * man/2 for exp == 0 (subnormal)
    return sign * torch.where(
        exp > 0,
        (1.0 + man.float() / 2.0) * torch.exp2(exp.float() - 1.0),
        man.float() / 2.0,
    )


def e4m3_to_float(scale: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn values as float32, decoded from their bits.

    ``scale`` is the storage of the e4m3 tensor: its uint8 bytes, or the
    float8 tensor itself, which is read through ``view(torch.uint8)``. The
    checkpoint's float8 tensors are only ever stored and moved as raw bytes
    (NPU has no float8 support); every value the model computes with is BF16.

    The decode is exact and uses float arithmetic only: integer shifts are
    very slow on NPU, and its ``exp2`` is off by about 1e-7. The fields come
    from the byte read as an integer; a power of two ``2 ** n`` is built from
    its fp32 exponent field ``n + 127``, written as the exact integer
    ``(n + 127) * 2 ** 23`` and reinterpreted.
    """
    value = scale.view(torch.uint8).float()
    exp = torch.floor(value / 8.0) - 16.0 * torch.floor(value / 128.0)  # bits 3-6
    man = value - 8.0 * torch.floor(value / 8.0)  # bits 0-2
    power = ((exp + 120.0) * 8388608.0).to(torch.int32).view(torch.float32)
    # Normal: 2^(exp - 7) * (1 + man/8); subnormal (exp == 0): man/8 * 2^-6.
    mantissa = torch.where(exp > 0, 1.0 + man / 8.0, man * 2.0**-9)
    number = torch.where(exp > 0, mantissa * power, mantissa)
    number = torch.where(value >= 128.0, -number, number)
    # No infinities: the largest exponent with every mantissa bit is NaN.
    return torch.where((exp == 15) & (man == 7), float("nan"), number)


def ue8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """uint8 ue8m0 scales as float32: ``2 ** (scale - 127)``, the byte read as
    the exponent field of an fp32 number (see ``e4m3_to_float``)."""
    exponent = scale_u8.float()
    scale = (exponent * 8388608.0).to(torch.int32).view(torch.float32)
    return torch.where(exponent == 0, torch.tensor(2.0**-127), scale)


def dequantize_nvfp4(
    weight_u8: torch.Tensor,
    scale_fp8: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
    chunk: int = 8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize NVFP4 weights to BF16.

    weight_u8: [E, N, K/2] uint8 (two e2m1 per byte, low nibble first)
    scale_fp8: [E, N, K/16] float8_e4m3fn block scales (uint8 bytes or float8)
    amax: [E] float32 per-expert amax
    denominator: [1] float32 outer scale denominator
    chunk: experts dequantized at a time
    out: [E, N, K] bfloat16 destination (allocated when omitted)
    Returns: [E, N, K] bfloat16
    """
    num_experts, rows, half = weight_u8.shape
    width = half * 2
    if out is None:
        out = torch.empty((num_experts, rows, width), dtype=torch.bfloat16, device=weight_u8.device)
    outer = (amax.float() / denominator.float()).view(num_experts, 1, 1, 1)
    for start in range(0, num_experts, chunk):
        end = min(start + chunk, num_experts)
        packed = weight_u8[start:end]
        # Unpack two e2m1 values per byte: low nibble first, high nibble second.
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
        values = e2m1_to_float(codes.reshape(end - start, rows, width))
        scale = e4m3_to_float(scale_fp8[start:end]).unsqueeze(-1)  # [e, N, K/16, 1]
        # (value * block_scale) * outer_scale, as the upstream GEMM.
        values = values.view(end - start, rows, width // 16, 16) * scale
        dequantized = values * outer[start:end]
        out[start:end] = dequantized.reshape(end - start, rows, width).to(torch.bfloat16)
    return out


def dequantize_mxfp8(
    weight_fp8: torch.Tensor,
    scale_u8: torch.Tensor,
    chunk: int = 8,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize MXFP8 weights to BF16.

    weight_fp8: [E, N, K] float8_e4m3fn weights (uint8 bytes or float8)
    scale_u8: [E, N, K/32] uint8 ue8m0 exponents
    chunk: experts dequantized at a time
    out: [E, N, K] bfloat16 destination (allocated when omitted)
    Returns: [E, N, K] bfloat16
    """
    num_experts, rows, width = weight_fp8.shape
    if out is None:
        out = torch.empty((num_experts, rows, width), dtype=torch.bfloat16, device=weight_fp8.device)
    for start in range(0, num_experts, chunk):
        end = min(start + chunk, num_experts)
        values = e4m3_to_float(weight_fp8[start:end]).view(end - start, rows, width // 32, 32)
        scale = ue8m0_to_float(scale_u8[start:end]).unsqueeze(-1)
        dequantized = values * scale
        out[start:end] = dequantized.reshape(end - start, rows, width).to(torch.bfloat16)
    return out
