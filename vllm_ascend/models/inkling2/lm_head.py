# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The LM head of a few decode rows: fp32 logits from the bf16 head.

Upstream runs the unembedding with an fp32 output. The engine calls
``compute_logits`` outside the captured graph for every decode step, and an
fp32 GEMM would first copy the [201024, 2304] head to fp32 each time (3.7 ms
on NPU). ``lm_head_fp32`` reads the bf16 rows once and accumulates in fp32
on the Cube units instead: about 0.9 ms for up to ``MAX_ROWS`` rows. More
rows (a prefill) take the fp32 GEMM, where the copy amortizes.
"""

import torch
from vllm.triton_utils import tl, triton

MAX_ROWS = 16


@triton.jit
def _lm_head_kernel(
    x_ptr,  # [M, K] bf16 rows
    w_ptr,  # [N, K] bf16 head rows
    out_ptr,  # [M, N] fp32
    M,
    K,
    N,
    stride_x,
    stride_w,
    stride_o,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    r_mask = rows < N
    m = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        w = tl.load(w_ptr + rows[:, None] * stride_w + ks[None, :], mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        x = tl.load(x_ptr + m[:, None] * stride_x + ks[None, :], mask=(m[:, None] < M) & k_mask[None, :], other=0.0)
        # bf16 operands are exact in the products; the Cube units accumulate
        # in fp32, as an fp32 GEMM of the same values does.
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(out_ptr + m[:, None] * stride_o + rows[None, :], acc, mask=(m[:, None] < M) & r_mask[None, :])


def lm_head_fp32(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 ``hidden_states @ weight.T`` for up to ``MAX_ROWS`` rows.

    hidden_states: [M, K] bf16; weight: [N, K] bf16.
    """
    tokens, k_size = hidden_states.shape
    vocab = weight.shape[0]
    assert tokens <= MAX_ROWS
    assert hidden_states.stride(1) == 1 and weight.stride(1) == 1
    out = torch.empty((tokens, vocab), dtype=torch.float32, device=hidden_states.device)
    _lm_head_kernel[(triton.cdiv(vocab, 256),)](
        hidden_states,
        weight,
        out,
        tokens,
        k_size,
        vocab,
        hidden_states.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=MAX_ROWS,
        BLOCK_N=256,
        BLOCK_K=min(128, triton.next_power_of_2(k_size)),
    )
    return out
