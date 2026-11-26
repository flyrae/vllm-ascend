# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the short causal convolutions on NPU (static and dynamic),
against the upstream reference (``tests/models/inkling2/test_core_ops.py``)."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.ops_sconv import short_conv
from vllm_ascend.models.inkling2.sconv import (
    dynamic_taps,
    short_conv_dynamic_seq,
    short_conv_static_seq,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

WIDTH = 4


@pytest.mark.parametrize("tokens", [1, 5, 300])
def test_short_conv_static_matches_reference(tokens):
    torch.manual_seed(0)
    channels = 64
    x = torch.randn(tokens, channels, device="npu").to(torch.bfloat16)
    weight = torch.randn(channels, WIDTH, device="npu").to(torch.bfloat16)
    out = short_conv_static_seq(x, weight).cpu()
    ref = reference.short_conv(x.cpu(), weight.cpu())
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


def test_short_conv_static_state_continues_the_sequence():
    """Chunked prefill and decode steps through the state equal one pass."""
    torch.manual_seed(0)
    tokens, channels = 23, 64
    x = torch.randn(tokens, channels, device="npu").to(torch.bfloat16)
    weight = torch.randn(channels, WIDTH, device="npu").to(torch.bfloat16)
    full = short_conv_static_seq(x, weight)

    state = torch.zeros(WIDTH - 1, channels, dtype=torch.bfloat16, device="npu")
    chunks = []
    start = 0
    for length in (3, 7, 1, 1, 11):
        chunks.append(short_conv_static_seq(x[start : start + length], weight, state))
        start += length
    torch.testing.assert_close(torch.cat(chunks), full, atol=0, rtol=0)


def test_short_conv_static_leaves_the_last_inputs_in_the_state():
    torch.manual_seed(0)
    tokens, channels = 9, 8
    x = torch.randn(tokens, channels, device="npu").to(torch.bfloat16)
    weight = torch.randn(channels, WIDTH, device="npu").to(torch.bfloat16)
    state = torch.zeros(WIDTH - 1, channels, dtype=torch.bfloat16, device="npu")
    short_conv_static_seq(x, weight, state)
    torch.testing.assert_close(state, x[-(WIDTH - 1) :], atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 5, 300])
def test_short_conv_dynamic_matches_reference(tokens):
    torch.manual_seed(0)
    heads, head_dim = 2, 16
    channels = heads * head_dim
    x = torch.randn(tokens, channels, device="npu").to(torch.bfloat16) * 3
    static_w = torch.randn(WIDTH, channels, device="npu").to(torch.bfloat16)
    dynamic_input = torch.randn(tokens, 32, device="npu").to(torch.bfloat16)
    weight_proj = torch.randn(heads * WIDTH, 32, device="npu").to(torch.bfloat16)

    deltas = dynamic_taps(dynamic_input, weight_proj)
    out = short_conv_dynamic_seq(x, static_w, deltas, heads).cpu()

    # The deltas are a bf16 linear of the dynamic input, as the layer computes them.
    deltas_cpu = deltas.view(tokens, heads, WIDTH).cpu()
    ref = reference.short_conv(x.cpu(), static_w.cpu(), deltas_cpu)
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


def test_short_conv_dynamic_state_continues_the_sequence():
    torch.manual_seed(0)
    tokens, heads, head_dim = 23, 2, 16
    channels = heads * head_dim
    x = torch.randn(tokens, channels, device="npu").to(torch.bfloat16)
    static_w = torch.randn(WIDTH, channels, device="npu").to(torch.bfloat16)
    deltas = torch.randn(tokens, heads * WIDTH, device="npu").to(torch.bfloat16)
    full = short_conv_dynamic_seq(x, static_w, deltas, heads)

    state = torch.zeros(WIDTH - 1, channels, dtype=torch.bfloat16, device="npu")
    chunks = []
    start = 0
    for length in (3, 7, 1, 1, 11):
        chunks.append(
            short_conv_dynamic_seq(
                x[start : start + length],
                static_w,
                deltas[start : start + length],
                heads,
                state,
            )
        )
        start += length
    torch.testing.assert_close(torch.cat(chunks), full, atol=0, rtol=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_paged_short_conv_matches_sequence_across_chunks(dynamic):
    """The engine's paged cache preserves taps across prefill and decode."""
    torch.manual_seed(7)
    heads, stream_width, width, block_size = 2, 4, WIDTH, 4
    channels = heads * stream_width
    total = 8
    x = (torch.randn(total, channels, device="npu") * 0.4).to(torch.bfloat16).contiguous()
    static_weight = (torch.randn(channels, width, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    dynamic_weight = (torch.randn(width, channels, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    deltas = (torch.randn(total, heads * width, device="npu") * 0.1).to(torch.bfloat16).contiguous()
    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="npu")
    slots = torch.tensor(
        [int(block_table[0, p // block_size]) * block_size + p % block_size for p in range(total)],
        dtype=torch.int64,
        device="npu",
    )
    cache = torch.zeros(8, heads, block_size, 16, dtype=torch.bfloat16, device="npu")
    chunks = []
    for start, end in ((0, 3), (3, 7), (7, 8)):
        length = end - start
        chunks.append(
            short_conv(
                x[start:end].contiguous(),
                dynamic_weight if dynamic else static_weight,
                cache,
                torch.arange(start, end, dtype=torch.int32, device="npu"),
                block_table,
                torch.zeros(length, dtype=torch.int32, device="npu"),
                slots[start:end],
                torch.zeros(length, dtype=torch.int32, device="npu"),
                0,
                stream_width,
                dynamic_taps=deltas[start:end].contiguous() if dynamic else None,
            )
        )
    got = torch.cat(chunks)
    want = (
        short_conv_dynamic_seq(x, dynamic_weight, deltas, heads) if dynamic else short_conv_static_seq(x, static_weight)
    )
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_paged_short_conv_batch_matches_sequence(dynamic):
    """What the engine hands the kernel: a strided prefill of three requests
    (K and V are column slices of the fused projection), then decode batches
    whose taps come from the paged cache, with a padding token.  Each request
    equals its own whole-sequence convolution."""
    from vllm_ascend.models.inkling2.ops_sconv import lag_taps

    torch.manual_seed(11)
    heads, stream_width, width, block_size = 2, 4, WIDTH, 4
    channels = heads * stream_width
    prompt = 5
    lengths = [8, 7, 5]  # the third request never decodes
    starts = [0, lengths[0], lengths[0] + lengths[1]]
    proj = (torch.randn(sum(lengths), channels + 6, device="npu") * 0.4).to(torch.bfloat16)
    x = proj[:, :channels]
    static_weight = (torch.randn(channels, width, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    dynamic_weight = (torch.randn(width, channels, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    weight = dynamic_weight if dynamic else static_weight
    deltas = (torch.randn(sum(lengths), heads * width, device="npu") * 0.1).to(torch.bfloat16).contiguous()
    block_table = torch.tensor([[1, 2], [4, 5], [7, 8]], dtype=torch.int32, device="npu")
    cache = torch.zeros(10, heads, block_size, 16, dtype=torch.bfloat16, device="npu")
    outputs = [[], [], []]

    def forward(rows, requests, padding=False):
        """One forward over ``rows`` (global rows of ``x``) of ``requests``.
        The inputs keep the projection's row stride, as K and V do."""
        real = len(rows)
        if padding:
            rows, requests = [*rows, 0], [*requests, 0]
        rows_t = torch.tensor(rows, device="npu")
        requests_t = torch.tensor(requests, dtype=torch.int32, device="npu")
        positions = torch.tensor([row - starts[r] for row, r in zip(rows, requests)], device="npu")
        slots = block_table[requests_t.long(), positions // block_size].long() * block_size + positions % block_size
        if padding:
            slots = torch.cat((slots, torch.tensor([-1], device="npu")))
        first_seen = {}
        query_start = torch.tensor(
            [first_seen.setdefault(r, i) for i, r in enumerate(requests)], dtype=torch.int32, device="npu"
        )
        out = short_conv(
            proj[rows_t][:, :channels],
            weight,
            cache,
            positions,
            block_table,
            requests_t,
            slots,
            query_start,
            0,
            stream_width,
            dynamic_taps=deltas[rows_t].contiguous() if dynamic else None,
            taps=lag_taps(weight, dynamic),
        )
        for i, r in enumerate(requests[:real]):
            outputs[r].append(out[i].clone())

    forward(
        [starts[r] + row for r in range(len(lengths)) for row in range(prompt)],
        [r for r in range(len(lengths)) for _ in range(prompt)],
    )
    for step in range(max(lengths) - prompt):
        active = [r for r, length in enumerate(lengths) if length > prompt + step]
        forward([starts[r] + prompt + step for r in active], active, padding=step == 1)

    for r, length in enumerate(lengths):
        rows = x[starts[r] : starts[r] + length].contiguous()
        want = (
            short_conv_dynamic_seq(rows, dynamic_weight, deltas[starts[r] : starts[r] + length], heads)
            if dynamic
            else short_conv_static_seq(rows, static_weight)
        )
        torch.testing.assert_close(torch.stack(outputs[r]), want, atol=0, rtol=0)


def test_paged_short_conv_residual_update():
    """With ``addend`` (and ``residual``), the result is
    ``(residual + addend) + y`` in fp32, ``y`` being the rounded convolution."""
    torch.manual_seed(3)
    heads, stream_width, width = 2, 4, WIDTH
    channels = heads * stream_width
    x = (torch.randn(1, channels, device="npu") * 0.4).to(torch.bfloat16)
    weight = (torch.randn(channels, width, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    addend = torch.randn(1, channels, device="npu").to(torch.bfloat16)
    residual = torch.randn(1, channels, device="npu").to(torch.bfloat16)
    cache = torch.zeros(4, heads, 4, 16, dtype=torch.bfloat16, device="npu")
    # The first token of the sequence stored at slot 4.
    args = (
        x,
        weight,
        cache,
        torch.zeros(1, dtype=torch.int64, device="npu"),
        torch.tensor([[1]], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.tensor([4], dtype=torch.int64, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        0,
        stream_width,
    )
    y = short_conv(*args)
    out = short_conv(*args, addend=addend, residual=residual)
    torch.testing.assert_close(out, (residual.float() + addend.float() + y.float()).to(torch.bfloat16))


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 5, 9, 33])
def test_sconv_seq_metadata_matches_reference(num_reqs):
    """Each token's request and its first row, in one launch into persistent
    buffers; the padding rows are zero."""
    import itertools

    from vllm_ascend.models.inkling2.ops_sconv import sconv_seq_metadata

    torch.manual_seed(0)
    lengths = torch.randint(1, 9, (num_reqs,)).tolist()
    starts = [0, *itertools.accumulate(lengths)]
    query_start_loc = torch.tensor(starts, dtype=torch.int32)
    num_actual_tokens = starts[-1]
    padding = 5
    seq_idx = torch.full((num_actual_tokens + padding,), -1, dtype=torch.int32, device="npu")
    query_start = torch.full((num_actual_tokens + padding,), -1, dtype=torch.int32, device="npu")

    sconv_seq_metadata(
        query_start_loc.to("npu"),
        num_reqs,
        num_actual_tokens,
        seq_idx,
        query_start,
        num_actual_tokens + padding,
    )

    ref_seq_idx, ref_query_start = [], []
    for token in range(num_actual_tokens):
        owner = max(j for j in range(num_reqs) if starts[j] <= token)
        ref_seq_idx.append(owner)
        ref_query_start.append(starts[owner])
    torch.testing.assert_close(seq_idx.cpu(), torch.tensor(ref_seq_idx + [0] * padding, dtype=torch.int32))
    torch.testing.assert_close(query_start.cpu(), torch.tensor(ref_query_start + [0] * padding, dtype=torch.int32))


@pytest.mark.parametrize("dynamic", [False, True])
def test_short_conv_kv_epilogue_matches_separate_stages(dynamic):
    """short_conv_kv equals short_conv + the eager epilogue: the K rows
    L2-clipped, the V rows RMS-normalized, both written at the KV slots."""
    from vllm_ascend.models.inkling2.ops_sconv import lag_taps, short_conv, short_conv_kv

    torch.manual_seed(5)
    heads, stream_width, width, block_size = 2, 4, WIDTH, 4
    channels = heads * stream_width
    total = 6
    proj = (torch.randn(total, channels + 8, device="npu") * 4).to(torch.bfloat16)
    x = proj[:, :channels]
    static_weight = (torch.randn(channels, width, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    dynamic_weight = (torch.randn(width, channels, device="npu") * 0.2).to(torch.bfloat16).contiguous()
    weight = dynamic_weight if dynamic else static_weight
    deltas = (torch.randn(total, heads * width, device="npu") * 0.1).to(torch.bfloat16).contiguous()
    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="npu")
    slots = torch.arange(total, dtype=torch.int64, device="npu")
    kv_slots = torch.tensor([8, 9, 10, 11, -1, 13], dtype=torch.int64, device="npu")
    threshold, eps = 22.6, 1e-6

    def reference():
        conv_cache = torch.zeros(4, heads, block_size, 16, dtype=torch.bfloat16, device="npu")
        y = short_conv(
            x,
            weight,
            conv_cache,
            slots,
            block_table,
            torch.zeros(total, dtype=torch.int32, device="npu"),
            slots,
            torch.zeros(total, dtype=torch.int32, device="npu"),
            0,
            stream_width,
            dynamic_taps=deltas if dynamic else None,
            taps=lag_taps(weight, dynamic),
        )
        # The epilogue works per head, as the eager path's [T, H, D] views do.
        per_head = y.view(total, heads, stream_width).float()
        keys = (per_head * torch.clamp(threshold / (per_head.norm(dim=-1, keepdim=True) + eps), max=1.0)).to(
            torch.bfloat16
        )
        vals = (per_head * torch.rsqrt(per_head.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        key_cache = torch.zeros(4, 4, heads, stream_width, dtype=torch.bfloat16, device="npu")
        value_cache = torch.zeros_like(key_cache)
        for t, slot in enumerate(kv_slots.tolist()):
            if slot >= 0:
                key_cache[slot // 4, slot % 4] = keys[t]
                value_cache[slot // 4, slot % 4] = vals[t]
        return key_cache, value_cache

    ref_k, ref_v = reference()
    got_k = torch.zeros_like(ref_k)
    got_v = torch.zeros_like(ref_v)
    for plane, is_k in ((got_k, True), (got_v, False)):
        conv_cache = torch.zeros(4, heads, block_size, 16, dtype=torch.bfloat16, device="npu")
        short_conv_kv(
            x,
            weight,
            conv_cache,
            slots,
            block_table,
            torch.zeros(total, dtype=torch.int32, device="npu"),
            slots,
            torch.zeros(total, dtype=torch.int32, device="npu"),
            kv_slots,
            plane,
            0,
            stream_width,
            is_k,
            threshold,
            eps,
            dynamic_taps=deltas if dynamic else None,
            taps=lag_taps(weight, dynamic),
        )
    torch.testing.assert_close(got_k, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(got_v, ref_v, atol=0, rtol=0)
