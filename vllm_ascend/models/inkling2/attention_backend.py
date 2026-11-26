# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side attention for Inkling 2 on NPU.

Two cache-owning kinds, following first-generation Inkling
(``vllm/models/inkling/nvidia/sconv_swa_attn.py``):

- ``AscendInkling2Attention``: one decoder layer's KV cache
  (``FullAttentionSpec`` or ``SlidingWindowSpec``), with the prologue
  (K/V short convolutions, clipping, KV-cache write) and the CANN fused
  attention (TND + additive relative bias + LSE) in ``forward``.  The extra
  loop passes own ``AscendInkling2AttentionPassCache`` each.
- ``AscendInkling2ConvState``: one layer's short-convolution state cache
  (``SlidingWindowSpec`` with ``block_size = sconv_kernel_size``), with a
  dummy backend that only provides the metadata builder.

Decode steps can replay a captured ACL graph (``cudagraph_mode=
FULL_DECODE_ONLY`` on the uncompiled model).  Everything the graph reads
then lives in persistent buffers the runner refreshes per step: the input
positions, the metadata builders' tensors, the block tables and the caches.
The one shape a replay cannot change is the relative bias, so a captured
attention builds it over the fixed key range ``[0, decode_graph_kv_len)``
and registers its fused-attention task with vllm-ascend, which rebinds the
lengths and block table before every replay.  Prefill and any batch without
a graph take the eager path.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, SlidingWindowSpec

from .norm import l2_clip_triton

CLIP_EPS = 1e-6  # the clip epsilon of attention.l2_clip and the K epilogue
# Value of a masked key in the additive bias handed to the fused attention op.
MASK_VALUE = -30000.0
# Longest key range a captured decode graph's fixed bias buffer can cover.
DECODE_GRAPH_MAX_KV_LEN = 32768


def decode_graph_kv_len(max_model_len: int) -> int:
    """32-aligned key range of a captured decode graph: every key a request
    can hold, so one fixed shape serves every decode step."""
    return -(-max_model_len // 32) * 32


@dataclass
class AscendInkling2ConvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks] physical blocks per req
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


@dataclass(frozen=True, slots=True)
class Inkling2FIAParamProvider:
    """Replay-time parameters of one attention cache owner's fused attention.

    vllm-ascend binds these to the captured attention task before every
    replay: the runner's persistent block table and the step's host lengths.
    The mask-free, pse-carried layout is fixed at capture and never rebound.
    """

    layer_name: str

    def resolve(self, attn_metadata) -> dict:
        md = attn_metadata[self.layer_name]
        return {
            "actual_seq_lengths": md.actual_seq_lengths_q,
            "actual_seq_lengths_kv": md.seq_lens_list,
            "block_table": md.block_tables,
        }


class AscendInkling2ConvMetadataBuilder(AttentionMetadataBuilder[AscendInkling2ConvMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        self.query_start_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False) -> AscendInkling2ConvMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
        num_padded_tokens = common_attn_metadata.slot_mapping.shape[0]
        assert num_padded_tokens >= num_actual_tokens

        # One device launch into the persistent buffers (no host round trip), so
        # a captured graph reads the refreshed values at the same addresses.
        from .ops_sconv import sconv_seq_metadata

        sconv_seq_metadata(
            common_attn_metadata.query_start_loc,
            num_reqs,
            num_actual_tokens,
            self.seq_idx_buffer,
            self.query_start_buffer,
            num_padded_tokens,
        )

        return AscendInkling2ConvMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:num_padded_tokens],
            seq_idx=self.seq_idx_buffer[:num_padded_tokens],
            query_start=self.query_start_buffer[:num_padded_tokens],
        )


class AscendInkling2ConvBackend(AttentionBackend):
    """Dummy backend for the sconv sliding-window cache management."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INKLING2_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling 2 sconv has no attention impl; the conv runs out-of-band.")

    @staticmethod
    def get_builder_cls() -> type[AscendInkling2ConvMetadataBuilder]:
        return AscendInkling2ConvMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class AscendInkling2ConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner emitting one sliding-window conv-state spec."""

    def __init__(self, *, num_kv_heads: int, head_dim: int, hidden_size: int, kernel_size: int, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self._kv_cache = torch.tensor([])  # bound by bind_kv_cache
        self.num_kv_heads = num_kv_heads
        hidden_per_head = hidden_size // num_kv_heads
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges = (
            (0, head_dim),  # K
            (head_dim, head_dim),  # V
            (2 * head_dim, hidden_per_head),  # attn-output
            (2 * head_dim + hidden_per_head, hidden_per_head),  # mlp-output
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, f"sconv cache supports bfloat16 only, got {self._dtype}"
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def kv_cache(self) -> torch.Tensor:
        return self._kv_cache

    @property
    def cache_block_size(self) -> int:
        """Physical token slots per conv-cache block after page unification."""
        return self._kv_cache.shape[2] if self._kv_cache.numel() else self.block_size

    @kv_cache.setter
    def kv_cache(self, value) -> None:
        """Bind Ascend's K/V allocation as the conv kernel's ``[B, H, N, C]``.

        The legacy Ascend allocator returns separate K/V tensors.  Depending
        on the entry point, the K tensor is either still a flat int8 buffer or
        has already been viewed as bf16 ``[B, N, H, C]``.  The conv state is
        packed entirely in K; V is only an allocation placeholder.
        """
        k_tensor = value[0] if isinstance(value, (list, tuple)) else value
        if k_tensor.numel() == 0:
            self._kv_cache = k_tensor
            return

        if k_tensor.dtype == torch.int8:
            # Convert the byte buffer before applying the logical shape.
            k_tensor = k_tensor.reshape(-1).view(self._dtype)
        if k_tensor.ndim == 4:
            # Page unification can expose a larger physical N than the
            # declared conv-state spec (for example 16 instead of 4).
            if k_tensor.shape[2:] == (self.num_kv_heads, self.head_size) and k_tensor.shape[1] > 0:
                self._kv_cache = k_tensor.permute(0, 2, 1, 3)
                return
            if k_tensor.shape[1] == self.num_kv_heads and k_tensor.shape[3] == self.head_size and k_tensor.shape[2] > 0:
                self._kv_cache = k_tensor
                return
            raise ValueError(f"Unexpected conv-state cache shape {tuple(k_tensor.shape)} for {self.prefix}")

        page_elements = self.block_size * self.num_kv_heads * self.head_size
        if k_tensor.numel() % page_elements:
            raise ValueError(
                f"Conv-state cache for {self.prefix} has {k_tensor.numel()} elements, "
                f"not a whole number of {page_elements}-element pages"
            )
        num_blocks = k_tensor.numel() // page_elements
        self._kv_cache = k_tensor.reshape(num_blocks, self.block_size, self.num_kv_heads, self.head_size).permute(
            0, 2, 1, 3
        )

    def bind_kv_cache(self, kv_cache) -> None:
        self.kv_cache = kv_cache

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendInkling2ConvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,  # nonzero for the Ascend allocator
            dtype=self._dtype,
            sliding_window=self.sliding_window,
        )


class AscendInkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """KV cache owner for one extra loop pass of an attention layer."""

    def __init__(
        self,
        *,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        is_local: bool,
        sliding_window: int,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_local = is_local
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        return AscendAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _attention_cache_spec(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.is_local,
            self.sliding_window,
        )


def _attention_cache_spec(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    is_local: bool,
    sliding_window: int,
) -> KVCacheSpec:
    kwargs = dict(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
    )
    if is_local:
        return SlidingWindowSpec(**kwargs, sliding_window=sliding_window)
    return FullAttentionSpec(**kwargs)


class AscendInkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling 2 attention for one decoder layer, engine-side.

    Does the prologue (K/V short convs, clipping, KV-cache write) and calls
    the CANN fused attention with the relative position bias. Uses
    ``AscendAttentionBackend``'s metadata (block tables, seq lengths, etc.)
    and runs its own computation instead of going through an impl.
    """

    def __init__(
        self,
        config,
        is_local: bool,
        conv_state: AscendInkling2ConvState,
        prefix: str,
        pass_conv_states: tuple[AscendInkling2ConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.num_heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.split_sizes = [q_size, self.num_heads * self.d_rel, q_size, self.num_heads, kv_size, kv_size]
        hidden = config.hidden_size
        from .layers import _KVConv, _Linear

        self.qkvr = _Linear(sum(self.split_sizes), hidden)
        self.wo_ud = _Linear(hidden, q_size)
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=torch.bfloat16), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.r_scale = nn.Parameter(torch.empty(self.num_heads, self.d_rel, dtype=torch.bfloat16), requires_grad=False)
        dynamic_input = None if is_local else hidden
        self.k_sconv = _KVConv(kv_size, config.sconv_kernel_size, dynamic_input, self.num_kv_heads)
        self.v_sconv = _KVConv(kv_size, config.sconv_kernel_size, dynamic_input, self.num_kv_heads)

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self._rel_rows = None  # built by process_weights_after_loading
        self.pass_caches = nn.ModuleList(
            AscendInkling2AttentionPassCache(
                prefix=f"{prefix}.pass{p}",
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                is_local=self.is_local,
                sliding_window=self.sliding_window,
            )
            for p in range(1, 1 if getattr(config, "loop_share_cache", False) else len(self.conv_states))
        )
        vllm_config = get_current_vllm_config()
        self.kv_cache_torch_dtype = torch.bfloat16
        self.kv_cache = torch.tensor([])  # bound by bind_kv_cache
        # A captured decode graph builds its relative bias over a fixed range.
        self.decode_graph_kv_len = decode_graph_kv_len(vllm_config.model_config.max_model_len)
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        return AscendAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _attention_cache_spec(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.is_local,
            self.sliding_window,
        )

    @property
    def rel_rows(self) -> torch.Tensor:
        """The relative projection row-major ``[extent, d_rel]``, one distance
        per row, for the bias kernel."""
        if self._rel_rows is None:
            self.process_weights_after_loading()
        return self._rel_rows

    def process_weights_after_loading(self) -> None:
        self._rel_rows = self.rel_logits_proj.detach().t().contiguous()

    def _pass_cache(self, pass_idx: int):
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def shard_slice(self, shard: int) -> slice:
        start = sum(self.split_sizes[:shard])
        return slice(start, start + self.split_sizes[shard])

    def _project(self, x_q, x_k, x_v):
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            return list(self.qkvr(x_q).split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = torch.nn.functional.linear(x_q, weight[:q_side])
        k = torch.nn.functional.linear(x_k, weight[q_side:k_end])
        v = torch.nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.split_sizes[:4], dim=-1), k, v]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """bf16 `(T, hidden)` attention output for one visit of one decoder layer."""
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        tokens = x_q.shape[0]
        heads, head_dim = self.num_heads, self.head_dim
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)

        conv_state = self.conv_states[pass_idx]
        conv_md = _conv_metadata(conv_state)
        if conv_md is None:
            # Profile run without conv-state cache: return zeros of the output shape.
            return torch.zeros(tokens, self.wo_ud.weight.shape[0], dtype=q.dtype, device=q.device)
        layer_name, kv_cache = self._pass_cache(pass_idx)
        attn_md = _attention_metadata(layer_name, kv_cache)
        if attn_md is None:
            raise RuntimeError(f"Missing attention metadata for {layer_name}")

        # K/V short convolutions with their epilogue: the K rows L2-clipped
        # and the V rows RMS-normalized, written straight into the KV cache.
        k_dynamic = self.k_sconv.dynamic_taps(x_k) if self.k_sconv.dynamic else None
        v_dynamic = self.v_sconv.dynamic_taps(x_v) if self.v_sconv.dynamic else None
        _conv_kv_apply(
            k,
            self.k_sconv,
            conv_state,
            conv_md,
            positions,
            k_dynamic,
            kv_cache[0],
            attn_md.slot_mapping,
            "K",
            self.qk_clip,
        )
        _conv_kv_apply(
            v, self.v_sconv, conv_state, conv_md, positions, v_dynamic, kv_cache[1], attn_md.slot_mapping, "V", self.eps
        )

        clipped_q = l2_clip_triton(q.view(tokens, heads, head_dim), self.q_scale, self.qk_clip, tau)
        clipped_r = l2_clip_triton(r.view(tokens, heads, self.d_rel), self.r_scale, self.rel_clip, tau)
        out, lse = _fused_attention_with_bias(
            clipped_q,
            clipped_r,
            self.rel_rows,
            positions,
            kv_cache,
            attn_md,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rel_extent,
            self.is_local,
            self.sliding_window if self.is_local else None,
            layer_name=layer_name,
            kv_cap=self.decode_graph_kv_len,
        )
        out = _gate_sink_fused(out, lse, gate.view(tokens, heads, head_dim), sinks.view(tokens, heads))
        return self.wo_ud(out.reshape(tokens, heads * head_dim))


def _conv_metadata(owner: AscendInkling2ConvState) -> AscendInkling2ConvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return attn_metadata.get(owner.prefix)


def _attention_metadata(prefix: str, kv_cache):
    """Return the Ascend attention metadata for a cache owner, if bound."""
    if isinstance(kv_cache, (list, tuple)):
        empty = len(kv_cache) == 0 or kv_cache[0].numel() == 0
    else:
        empty = kv_cache.numel() == 0
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or empty:
        return None
    return attn_metadata.get(prefix)


def _conv_kv_apply(
    x, sconv, owner, md, positions, dynamic_taps, kv_plane, kv_slot_mapping, stream_name, epilogue_param
):
    """One K or V short convolution whose output goes straight into the paged
    KV cache: the K rows L2-clipped (``epilogue_param`` the threshold) and
    the V rows RMS-normalized (``epilogue_param`` the epsilon)."""
    from .ops_sconv import short_conv_kv

    offset, width = owner.stream_ranges[{"K": 0, "V": 1}[stream_name]]
    short_conv_kv(
        x,
        sconv.conv_weight,
        owner.kv_cache,
        positions,
        md.block_table,
        md.seq_idx,
        md.slot_mapping,
        md.query_start,
        kv_slot_mapping,
        kv_plane,
        offset,
        width,
        stream_name == "K",
        epilogue_param,  # K: the clip threshold; V: unused
        epilogue_param if stream_name == "V" else 0.0,  # V: the norm epsilon
        dynamic_taps=dynamic_taps,
        taps=sconv.paged_taps,
    )


def _fused_attention_with_bias(
    q,
    r,
    rel_rows,
    positions,
    kv_cache,
    md,
    num_heads,
    num_kv_heads,
    head_dim,
    extent,
    is_local,
    sliding_window,
    layer_name,
    kv_cap,
):
    """CANN fused attention with the relative position bias.

    Returns (out [T, H, D] fp32, lse [T, H] fp32).

    Both paths call the operator with one query row per request: the bias is
    ``[T, H, 1, kv_pad]`` with the mask inside it, no causal template.  A
    multi-row request (one ``actual_seq_length`` per chunk, ``q`` up to the
    chunk size) crashes the operator's kernel for some ``(q, kv)`` shapes --
    ``q=32`` with ``kv`` of 96..192 among them, which every chunked prefill
    past two chunks hits.  Per-row calls are safe at any ``kv`` the dense sweep
    covered (q=1, kv 1..2048), and the graph path already used them.  Masked
    keys carry ``MASK_VALUE`` instead of ``-inf``: ``exp(MASK_VALUE - lse)``
    underflows to exactly 0 in fp32, so the softmax is unchanged, and ``-inf``
    entries in the bias leave inf/nan in the operator's LSE output.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    if _EXTRA_CTX.capturing:
        return _captured_decode_attention(
            q, r, rel_rows, positions, kv_cache, md, num_heads, num_kv_heads, head_dim, is_local, layer_name, kv_cap
        )

    tokens = q.shape[0]
    kv_pad = _kv_pad_from_md(md)
    # The per-token expansion of the batch (one request per query row) depends
    # only on the batch composition, not on the layer: the metadata object is
    # rebuilt every step, so caching on it computes it once per step instead
    # of once per layer visit.
    expanded = getattr(md, "inkling2_per_token", None)
    if expanded is None or expanded[0].shape[0] != tokens:
        kv_lens = _actual_kv_lengths(md)
        q_lens = kv_lens.new_tensor(_q_lens_from_md(md))
        expanded = (
            kv_lens.repeat_interleave(q_lens),
            md.block_tables.repeat_interleave(q_lens, dim=0),
            torch.arange(1, tokens + 1, dtype=torch.int32, device=q.device),
        )
        md.inkling2_per_token = expanded
    kv_per_token, block_rows, seq_q = expanded
    pse = _decode_relative_bias(r, rel_rows, positions, kv_per_token, kv_pad, is_local)
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    if key_cache.ndim == 4:
        # CANN's TND + pse_shift path requires the paged cache's H/D axes
        # flattened.  This is a view of the same storage used by the writer.
        key_cache = key_cache.reshape(key_cache.shape[0], key_cache.shape[1], -1)
        value_cache = value_cache.reshape(value_cache.shape[0], value_cache.shape[1], -1)
    # ``actual_seq_lengths`` is cumulative (one boundary per request), as
    # vllm-ascend always passes it; per-request 1s misassign the query rows
    # and poison the LSE.
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q,
        key_cache,
        value_cache,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        scale=head_dim**-0.5,
        block_table=block_rows,
        block_size=key_cache.shape[1],
        actual_seq_lengths=seq_q,
        actual_seq_lengths_kv=kv_per_token,
        pse_shift=pse,
        softmax_lse_flag=True,
        sparse_mode=0,
    )
    # TND single-layout LSE is [B, N, 1]: request-major, then head.
    return out.float(), lse.reshape(tokens, num_heads).float()


def _captured_decode_attention(
    q, r, rel_rows, positions, kv_cache, md, num_heads, num_kv_heads, head_dim, is_local, layer_name, kv_cap
):
    """Record one decode attention in the graph being captured.

    Only uniform single-token decode batches are captured (``FULL_DECODE_ONLY``):
    each request brings one query, so its bias is ``[B, H, 1, kv_cap]`` for the
    graph's whole life and is masked from the device sequence lengths at every
    replay.  The attention task is registered with vllm-ascend, which rebinds
    the lengths and the block table before each replay.
    """
    from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task

    tokens = q.shape[0]
    if md.max_query_len != 1 or len(md.actual_seq_lengths_q) != tokens:
        raise NotImplementedError(
            "Inkling 2 captures ACL graphs for single-token decode batches only; use "
            "cudagraph_mode=FULL_DECODE_ONLY (or NONE) instead of FULL"
        )
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    if key_cache.ndim == 4:
        key_cache = key_cache.reshape(key_cache.shape[0], key_cache.shape[1], -1)
        value_cache = value_cache.reshape(value_cache.shape[0], value_cache.shape[1], -1)
    pse = _decode_relative_bias(r, rel_rows, positions, md.seq_lens_gpu[:tokens], kv_cap, is_local)
    out = torch.empty((tokens, num_heads, head_dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((tokens, num_heads, 1), dtype=torch.float32, device=q.device)
    params = dict(
        query=q,
        key=key_cache,
        value=value_cache,
        pse_shift=pse,
        atten_mask=None,
        block_table=md.block_tables,
        num_heads=num_heads,
        scale=head_dim**-0.5,
        input_layout="TND",
        num_key_value_heads=num_kv_heads,
        sparse_mode=0,
        inner_precise=0,
        block_size=key_cache.shape[1],
        softmax_lse_flag=True,
        actual_seq_lengths=list(md.actual_seq_lengths_q),
        actual_seq_lengths_kv=list(md.seq_lens_list),
    )
    workspace = get_capture_resource(
        ("inkling2_fia_workspace", num_heads, num_kv_heads, head_dim),
        lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(**params),
    )
    register_task(
        torch_npu.npu_fused_infer_attention_score.out,
        {**params, "workspace": workspace, "out": [out, lse]},
        Inkling2FIAParamProvider(layer_name),
    )
    return out.float(), lse.reshape(tokens, num_heads)


def _decode_relative_bias(r, rel_rows, positions, kv_lens, kv_cap, is_local):
    """Single-token relative bias of a decode batch, as [B, H, 1, kv_cap].

    ``r`` is [B, H, d_rel] bf16 (clipped), ``rel_rows`` the relative
    projection in row-major ``[extent, d_rel]`` (``rel_proj.t().contiguous()``,
    cached by the layer), ``positions`` [B] the query positions, ``kv_lens``
    [B] the device sequence lengths.  Request ``b``'s key ``s`` gets
    ``r[b, h] . rel_rows[positions[b] - s]``, as ``attention.relative_bias``
    with the same bf16 rounding: zero past the extent on full layers and
    masked on local ones.  Keys at or past ``kv_lens[b]`` (and every key of a
    padding request) are masked, which keeps the shape fixed for a captured
    graph.  One Triton launch; the dot runs on the Cube units with an fp32
    accumulator, as the einsum it replaces.
    """
    tokens, heads, d_rel = r.shape
    extent = rel_rows.shape[0]
    bias = torch.empty((tokens, heads, 1, kv_cap), dtype=torch.bfloat16, device=r.device)
    _rel_bias_kernel[(tokens, triton.cdiv(kv_cap, 128))](
        r,
        rel_rows,
        positions,
        kv_lens,
        bias,
        heads,
        kv_cap,
        extent,
        MASK_VALUE,
        r.stride(0),
        bias.stride(0),
        IS_LOCAL=is_local,
        D_REL=d_rel,
        H_PAD=max(triton.next_power_of_2(heads), 16),
        BLOCK_K=128,
        num_warps=4,
    )
    return bias


@triton.jit
def _gate_sink_kernel(
    out_ptr,  # [T, H, D] fp32 attention output
    lse_ptr,  # [T, H] fp32 natural-log LSE
    gate_ptr,  # [T, H, D] bf16 gate (row- and head-strided views are fine)
    sink_ptr,  # [T, H] bf16 per-token attention sinks
    res_ptr,  # [T, H, D] bf16 result
    stride_o_t,
    stride_o_h,
    stride_g_t,
    stride_g_h,
    stride_l_t,
    stride_s_t,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``out * sigmoid(gate) * sigmoid(lse - sink)`` of one head, in fp32 with
    a bf16 result, as ``attention.gate_sink`` computes it."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    out = tl.load(out_ptr + token * stride_o_t + head * stride_o_h + cols, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + token * stride_g_t + head * stride_g_h + cols, mask=mask, other=0.0)
    # lse and sink are [T, H], row strided.
    lse = tl.load(lse_ptr + token * stride_l_t + head)
    sink = tl.load(sink_ptr + token * stride_s_t + head)
    y = out * tl.sigmoid(gate.to(tl.float32))
    y = y * tl.sigmoid(lse - sink.to(tl.float32))
    tl.store(res_ptr + token * stride_o_t + head * stride_o_h + cols, y.to(tl.bfloat16), mask=mask)


def _gate_sink_fused(out, lse, gate, sinks):
    """``gate_sink`` as one Triton launch (see the kernel)."""
    tokens, heads, width = out.shape
    res = torch.empty((tokens, heads, width), dtype=torch.bfloat16, device=out.device)
    _gate_sink_kernel[(tokens, heads)](
        out,
        lse,
        gate,
        sinks,
        res,
        out.stride(0),
        out.stride(1),
        gate.stride(0),
        gate.stride(1),
        lse.stride(0),
        sinks.stride(0),
        width,
        BLOCK_D=triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return res


@triton.jit
def _rel_bias_kernel(
    r_ptr,
    rows_ptr,
    pos_ptr,
    kv_lens_ptr,
    out_ptr,
    num_heads,
    kv_cap_,
    extent_,
    mask_value,
    stride_r_b,
    stride_out_b,
    IS_LOCAL: tl.constexpr,
    D_REL: tl.constexpr,
    H_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One query row's bias over ``[0, kv_cap)``: the dot of its clipped
    relative queries with each key's distance row of the projection."""
    query = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    d = tl.arange(0, D_REL)
    h = tl.arange(0, H_PAD)
    h_mask = h < num_heads
    pos = tl.load(pos_ptr + query)
    kv_len = tl.load(kv_lens_ptr + query)
    r = tl.load(r_ptr + query * stride_r_b + h[:, None] * D_REL + d[None, :], mask=h_mask[:, None], other=0.0)
    dist = pos - k
    # The projection row of each key's distance, clamped into range for the
    # load; the far keys' values are overwritten below.
    rows = tl.load(
        rows_ptr + tl.minimum(tl.maximum(dist, 0), extent_ - 1)[:, None] * D_REL + d[None, :],
        mask=k[:, None] < kv_cap_,
        other=0.0,
    )
    acc = tl.dot(r, tl.trans(rows))
    valid = (dist >= 0) & (k < kv_len) & (k < kv_cap_)
    far = dist >= extent_
    if IS_LOCAL:
        value = tl.where(valid & ~far, acc, mask_value)
    else:
        value = tl.where(valid, tl.where(far, 0.0, acc), mask_value)
    tl.store(
        out_ptr + query * stride_out_b + h[:, None] * kv_cap_ + k[None, :],
        value.to(tl.bfloat16),
        mask=h_mask[:, None] & (k[None, :] < kv_cap_),
    )


def _kv_pad_from_md(md) -> int:
    """Padded KV length for the bias tensor (max over the batch, 32-aligned)."""
    kv_lens = md.seq_lens_list
    if kv_lens is None:
        kv_lens = md.seq_lens.tolist()
    max_kv = max(kv_lens) if kv_lens else 128
    return -(-max_kv // 32) * 32


def _query_starts(md) -> list[int]:
    """Zero-prefixed cumulative query lengths from Ascend metadata."""
    cumulative = getattr(md, "actual_seq_lengths_q", None)
    if cumulative is not None:
        if isinstance(cumulative, torch.Tensor):
            cumulative = cumulative.tolist()
        return [0, *map(int, cumulative)]
    starts = getattr(md, "query_start_loc_cpu", None)
    if starts is None:
        starts = md.query_start_loc
    if isinstance(starts, torch.Tensor):
        starts = starts.tolist()
    return list(map(int, starts))


def _q_lens_from_md(md) -> list[int]:
    """Query lengths per request, from cumulative query start locations."""
    starts = _query_starts(md)
    return [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]


def _actual_kv_lengths(md):
    if md.seq_lens is not None:
        return md.seq_lens.to(device=md.block_tables.device, dtype=torch.int32)
    return torch.tensor(md.seq_lens_list, dtype=torch.int32, device=md.block_tables.device)
