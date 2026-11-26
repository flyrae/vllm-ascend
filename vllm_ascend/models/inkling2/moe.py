# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 MoE forward for Inkling 2 on NPU.

Upstream counterparts: ``_select_experts``, ``route`` and ``combine``
(``ops/moe.py``) and ``_situ`` (``ops/activation.py``). The routing, the
SwiGLU and the combine are re-implemented in PyTorch from their formulas for
the reference fallback; the NPU forward uses CANN routing, grouped GEMM and
unpermute. The fallback expert GEMMs call upstream's ``grouped_gemm`` Triton
kernel unchanged (it has no CUDA-only constructs). The gate/up columns are
interleaved (even columns gate, odd columns up) and the routing weight is
applied inside the SwiGLU.

Upstream runs the experts in NVFP4 / MXFP8 with quantized activations on
Blackwell (BF16 activations on Hopper); here they are BF16 weights
(:mod:`.dequantize`) and BF16 activations.

``grouped_gemm``'s ``out=None`` path allocates uninitialized memory
(``new_empty``); on triton-ascend some rows can keep NaN from that allocation.
Always pass a zero-initialized ``out`` (ACCUMULATE=True, so 0 + result is
exact)."""

import torch
from vllm.triton_utils import tl, triton

from .grouped_gemm import grouped_gemm


def moe_route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    topk: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens to experts (upstream ``ops.moe.select_experts``).

    x: [T, D] input tokens
    gate_weight: [E, D] router weight
    gate_bias: [E] router bias, steering only the selection
    topk: number of experts per token
    route_scale: scaling factor for routing weights

    Experts are selected by ``sigmoid(logit) + bias``. The routing weight of a
    selected expert is its sigmoid probability renormalized over the selected
    experts, computed from the logits without the bias.

    Returns:
        perm: [T * topk] row indices (token * topk + k), sorted by expert
        gammas: [T * topk] fp32 routing weights of the sorted rows
        offsets: [E + 1] row ranges per expert
        src2dst: [T * topk] token of each sorted row
    """
    num_tokens, _ = x.shape
    num_experts = gate_weight.shape[0]

    logits = x.float() @ gate_weight.float().t()  # [T, E]
    selection = torch.sigmoid(logits) + gate_bias.float()
    topk_indices = torch.topk(selection, topk, dim=-1).indices  # [T, K]
    log_probs = _log_sigmoid(logits.gather(1, topk_indices))
    max_log_probs = log_probs.max(dim=1).values
    shifted = torch.exp(log_probs - max_log_probs.unsqueeze(1))
    logsumexp = max_log_probs + torch.log(shifted.sum(1))
    weights = torch.exp(log_probs - logsumexp.unsqueeze(1)) * route_scale  # [T, K]

    # Expert-sorted rows: sort (token, k) pairs by expert ID.
    expert_ids = topk_indices.reshape(-1)  # [T * K]
    perm = torch.argsort(expert_ids).to(torch.int32)
    sorted_experts = expert_ids[perm.long()]

    gammas = weights.reshape(-1)[perm.long()]

    counts = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = counts.cumsum(0)

    src2dst = torch.div(perm, topk, rounding_mode="floor")

    return perm, gammas, offsets, src2dst


def _log_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.minimum(logits, torch.zeros_like(logits)) - torch.log1p(torch.exp(-torch.abs(logits)))


def moe_swiglu(
    gate_up: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """SwiGLU activation with softcap and routing weight (upstream ``_situ``).

    gate_up: [rows, 2 * I] with interleaved (gate, up) columns
    gammas: [rows] fp32 routing weights
    gate_cap: softcap for gate (tanh(gate / cap) * cap)
    up_cap: softcap for up projection

    Returns: [rows, I] activated values, rounded to bf16
    """
    # Upstream layout: interleaved (gate, up) columns -- even columns are gate,
    # odd columns are up (activation.py `_situ` loads bf16 pairs as one word).
    gate = gate_up[:, 0::2].float()
    up = gate_up[:, 1::2].float()

    # SwiGLU: sigmoid is computed on the ORIGINAL gate (before softcap),
    # then softcap is applied, then multiply: gate * sigmoid * up * gamma
    sigmoid = torch.sigmoid(gate)
    gate = torch.tanh(gate / gate_cap) * gate_cap
    up = torch.tanh(up / up_cap) * up_cap

    activated = ((gate * sigmoid) * up) * gammas.unsqueeze(-1)
    return activated.to(gate_up.dtype)


@triton.jit
def _moe_swiglu_kernel(
    gu_ptr,  # [rows, 2 * I] bf16, gate and up interleaved column pairs
    gamma_ptr,  # [rows] fp32 routing weights
    out_ptr,  # [rows, inter_size] bf16
    inter_size,
    gate_cap,
    up_cap,
    stride_gu,
    stride_out,
    BLOCK_I: tl.constexpr,
):
    """``tanh_softcap(gate) * sigmoid(gate) * tanh_softcap(up) * gamma`` in
    fp32, as ``moe_swiglu`` computes it, in one launch."""
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = i < inter_size
    base = gu_ptr + row * stride_gu
    gate = tl.load(base + 2 * i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + 2 * i + 1, mask=mask, other=0.0).to(tl.float32)
    # The sigmoid is of the original gate, before its softcap (moe_swiglu).
    sig = tl.sigmoid(gate)
    gate = tl.math.tanh(gate / gate_cap) * gate_cap
    up = tl.math.tanh(up / up_cap) * up_cap
    y = gate * sig * up * tl.load(gamma_ptr + row)
    tl.store(out_ptr + row * stride_out + i, y.to(tl.bfloat16), mask=mask)


def moe_swiglu_fused(
    gate_up: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """``moe_swiglu`` as one Triton launch (see the kernel for the formula)."""
    rows, two_i = gate_up.shape
    intermediate = two_i // 2
    out = torch.empty((rows, intermediate), dtype=gate_up.dtype, device=gate_up.device)
    _moe_swiglu_kernel[(rows, triton.cdiv(intermediate, 128))](
        gate_up,
        gammas,
        out,
        intermediate,
        gate_cap,
        up_cap,
        gate_up.stride(0),
        out.stride(0),
        BLOCK_I=128,
        num_warps=4,
    )
    return out


def moe_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    route_scale: float,
    gate_cap: float,
    up_cap: float,
    gate_weight_fp32: torch.Tensor | None = None,
    gate_bias_fp32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Complete MoE forward: route -> w13 GEMM -> SwiGLU -> w2 GEMM -> combine.

    x: [T, D] input tokens
    gate_weight: [E, D] router weight
    gate_bias: [E] router bias
    w13: [E, 2*I, D] gate/up projection weights (BF16)
    w2: [E, D, I] down projection weights (BF16)
    topk: experts per token
    route_scale: routing weight scale
    gate_cap, up_cap: softcap values

    Returns: [T, D] MoE output
    """
    num_tokens, hidden = x.shape

    if x.device.type == "npu":
        return _cann_moe_forward(
            x,
            gate_weight,
            gate_bias,
            w13,
            w2,
            topk,
            route_scale,
            gate_cap,
            up_cap,
            gate_weight_fp32,
            gate_bias_fp32,
        )

    # 1. Route
    perm, gammas, offsets, src2dst = moe_route(x, gate_weight, gate_bias, topk, route_scale)
    num_rows = perm.shape[0]

    # 2. w13 GEMM: gate_up = x @ w13.T (grouped by expert)
    # grouped_gemm expects: x [tokens, K], weight [E, N, K], perm [rows], offsets [E+1]
    gate_up = grouped_gemm(x, w13, perm, offsets, num_rows, topk, gather=True)
    # gate_up: [num_rows, 2*I]

    # 3. SwiGLU
    activated = moe_swiglu(gate_up, gammas, gate_cap, up_cap)
    # activated: [num_rows, I]

    # 4. w2 GEMM: down = activated @ w2.T (grouped by expert)
    # grouped_gemm's out=None path allocates uninitialized memory (new_empty);
    # on triton-ascend some rows can keep NaN from that allocation. Always pass
    # a zero-initialized out (ACCUMULATE=True, so 0 + result is exact).
    down = grouped_gemm(
        activated,
        w2,
        perm,
        offsets,
        num_rows,
        topk,
        gather=False,
        out=torch.zeros(num_rows, hidden, dtype=activated.dtype, device=x.device),
    )
    # down: [num_rows, D]

    # 5. Combine: fp32 sum of each token's expert rows, rounded once to bf16.
    combined = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    combined.index_add_(0, src2dst, down.float())
    return combined.to(x.dtype)


def _cann_moe_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    route_scale: float,
    gate_cap: float,
    up_cap: float,
    gate_weight_fp32: torch.Tensor | None = None,
    gate_bias_fp32: torch.Tensor | None = None,
) -> torch.Tensor:
    """CANN grouped MoE path used by vllm-ascend's token dispatcher.

    The transposed expert weights are views, not copies: CANN accepts the
    ``[E, D, 2I]`` / ``[E, I, D]`` strides directly.  Routing weights are
    applied inside SwiGLU, before the down projection, as in upstream.
    """
    import torch_npu

    tokens, _ = x.shape
    experts = w13.shape[0]
    # The fp32 router logits need an fp32 copy of the gate weight; keep one
    # instead of casting 2304 x experts every layer visit.  It stays
    # untransposed and is used through a ``.t()`` view, so the GEMM sees the
    # same strides as the per-call cast it replaces (a contiguous transposed
    # copy picks another kernel and shifts the logits by an ulp).
    weight = gate_weight_fp32 if gate_weight_fp32 is not None else gate_weight.float()
    bias = gate_bias_fp32 if gate_bias_fp32 is not None else gate_bias.float()
    logits = x.float() @ weight.t()
    weights, ids, _ = torch_npu.npu_moe_gating_top_k(
        logits,
        k=topk,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=1,
        out_flag=False,
        routed_scaling_factor=route_scale,
        eps=1e-20,
        bias=bias,
    )
    sorted_x, row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids.to(torch.int32),
        active_num=tokens * topk,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        quant_mode=-1,
    )
    group_list = expert_tokens.to(torch.int64)
    gate_up = torch_npu.npu_grouped_matmul(
        x=[sorted_x],
        weight=[w13.transpose(1, 2)],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
    )[0]
    # row_idx[f] is the sorted position of expanded row f = token * k; the
    # source row of every sorted position is its argsort.  Sorted as float32
    # the sort runs on the AI Core in ~60 us at any size; as int32 it falls
    # back to the AI CPU and grows to ~1 ms at 16k rows.  The values are
    # distinct indices below 2**24, so the cast is exact and the order does
    # not depend on stability.  vllm-ascend's MoE LoRA and GDN builder sort
    # the same way.
    source_row = torch.argsort(row_idx.to(torch.float32))
    sorted_weights = weights.reshape(-1)[source_row]
    activated = moe_swiglu_fused(gate_up, sorted_weights, gate_cap, up_cap)
    down = torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=[w2.transpose(1, 2)],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
    )[0]
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=down,
        sorted_indices=row_idx,
        probs=torch.ones_like(weights, dtype=torch.bfloat16),
    )
