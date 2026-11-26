# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the loop message-controller helpers on NPU, against the
upstream references (``tests/models/inkling2/test_core_ops.py``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.loop_ops import blend_rms_norm, pair_rms_norm

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

DIM = 2304
RANK = 64


@pytest.mark.parametrize("tokens", [1, 37])
@pytest.mark.parametrize("broadcast", [False, True])
def test_pair_rms_norm_matches_reference(tokens, broadcast):
    """The bf16 ``[rms(dest), rms(message)]`` pairs; ``dest`` may broadcast
    over the streams (a local layer's shared q/k/v controller input)."""
    torch.manual_seed(42)
    streams = 3
    source = torch.randn(1 if broadcast else streams, tokens, DIM)
    dest = source.expand(streams, -1, -1)
    message = torch.randn(streams, tokens, DIM)

    out = pair_rms_norm(dest.to("npu"), message.to("npu")).cpu()
    ref = reference.pair_rms_norm(dest, message)

    assert out.dtype == torch.bfloat16 and out.shape == (streams, tokens, 2 * DIM)
    # tl.rsqrt may differ from torch.rsqrt in the last bf16 ulp.
    torch.testing.assert_close(out, ref, atol=0, rtol=2**-7)


@pytest.mark.parametrize("tokens", [1, 37])
@pytest.mark.parametrize("broadcast", [False, True])
def test_blend_rms_norm_matches_controller_then_norm(tokens, broadcast):
    """``(1 - tanh(forget)) * dest + tanh(gain) * message`` per stream, then
    the RMSNorm with the stream's row of the norm weight (which may broadcast)."""
    torch.manual_seed(0)
    streams = 3
    source = torch.randn(1 if broadcast else streams, tokens, DIM)
    dest = source.expand(streams, -1, -1)
    message = torch.randn(streams, tokens, DIM)
    gain = torch.randn(streams, tokens, DIM) * 0.5
    forget = torch.randn(streams, tokens, DIM) * 0.5
    gain_forget = torch.cat((gain, forget), dim=-1).contiguous()
    norm_weight = torch.randn(1 if broadcast else streams, DIM).to(torch.bfloat16).expand(streams, -1)

    out = blend_rms_norm(
        dest.to("npu"),
        message.to("npu"),
        gain_forget.to("npu"),
        norm_weight.contiguous().to("npu"),
        1e-6,
    ).cpu()

    blended = (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message
    ref = reference.rms_norm(blended, norm_weight.float().unsqueeze(1), 1e-6)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


@pytest.mark.parametrize("tokens", [1, 37])
def test_streamed_controllers_match_the_per_stream_reference(tokens):
    """The upstream controller reference, end to end: ``h = down(rms([dest,
    message]))``, the blend, and the RMSNorm, per stream."""
    torch.manual_seed(0)
    streams = 4
    down = (torch.randn(streams, RANK, 2 * DIM) * 0.02).to(torch.bfloat16)
    gain_forget = (torch.randn(streams, 2 * DIM, RANK) * 0.1).to(torch.bfloat16)
    dest = torch.randn(streams, tokens, DIM)
    message = torch.randn(streams, tokens, DIM)
    norm_weight = torch.randn(streams, DIM).to(torch.bfloat16)

    pair = pair_rms_norm(dest.to("npu"), message.to("npu"))
    hidden = torch.bmm(pair, down.to("npu").transpose(1, 2))
    gain_forget_out = torch.bmm(hidden.float(), gain_forget.to("npu").transpose(1, 2).float())
    out = blend_rms_norm(dest.to("npu"), message.to("npu"), gain_forget_out, norm_weight.to("npu"), 1e-6).cpu()

    for stream in range(streams):
        blended = reference.controller(
            dest[stream],
            message[stream],
            down[stream],
            gain_forget[stream, :DIM],
            gain_forget[stream, DIM:],
        )
        ref = reference.rms_norm(blended, norm_weight[stream].float(), 1e-6)
        torch.testing.assert_close(out[stream], ref, atol=2e-2, rtol=2e-2)
