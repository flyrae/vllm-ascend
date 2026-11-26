# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 decoder layers on NPU.

The pieces of upstream ``nvidia/model.py``, ``attention.py``, ``moe.py``,
``sconv.py`` and ``loop.py`` built from the NPU components of this package.
The weights are plain parameters with upstream's names (after its renames),
so ``load_weights`` shares upstream's mapping.

Differences from the NVIDIA layers:
  - tensor parallel is not supported (single device)
  - offline attention uses an explicit ``AscendInkling2LayerState`` (dense KV
    and conv state); engine attention uses the paged caches in
    ``attention_backend.py``
  - the experts are BF16 (dequantized from NVFP4 / MXFP8 at load time)

Method by method:

- ``AscendInkling2RMSNorm.forward``, ``AscendInkling2DecoderLayer.attention``
  and ``AscendInkling2DecoderLayer.blend`` keep upstream's code (the norm
  without its ``out_dtype`` argument, the blend without its SM12x branch)
- ``AscendInkling2Attention._project`` is upstream's generic branch (one
  linear for the query side, one each for k and v), without LoRA and the SM12x
  ``qkv_bmm``
- ``AscendInkling2ShortConv.conv_mix`` follows upstream's ``conv_mix``
  fallback for SM12x (a ``short_conv`` with the residual update, then
  ``attn_res``), not its fused kernel
- ``AscendInkling2Attention.forward`` is re-implemented from upstream's
  ``forward`` / ``_attention`` and the prologue kernels
"""

from dataclasses import dataclass

import torch
from torch import nn

from .attention import attention, bmm_fp32, gate_sink, l2_clip, relative_bias
from .attn_res import attn_res
from .configs import Inkling2ModelConfig
from .loop_ops import blend_rms_norm, pair_rms_norm
from .moe import moe_forward
from .norm import rms_norm
from .sconv import dynamic_taps, short_conv_dynamic_seq, short_conv_static_seq

# The loop message streams of a core layer, in the order the checkpoint's
# ``middle_cycle_cross_pass_queries`` and controllers stack them.
LOOP_STREAMS = ("attn", "k", "v", "mlp")


@dataclass
class AscendInkling2LayerState:
    """KV and conv state of one ``(layer, loop pass)``.

    key, value: [H_kv, L, D] bf16, the first ``start + T`` rows valid
    conv: per stream (``k``, ``v``, ``attn``, ``mlp``) its last ``W - 1`` inputs
    """

    key: torch.Tensor
    value: torch.Tensor
    conv: dict[str, torch.Tensor]


class AscendInkling2RMSNorm(nn.Module):
    """RMSNorm; ``num_inputs > 1`` stacks the weights of the per-input norms."""

    def __init__(self, hidden_size: int, eps: float, num_inputs: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps)


class AscendInkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), scaled by
    ``hidden_size**-0.5`` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros((1, hidden) if i in local_ids else (3, hidden), dtype=torch.float32),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False)
        self.scale = hidden**-0.5


class AscendInkling2Embedding(nn.Module):
    """An embedding table holding ``weight`` ``[vocab, hidden]``."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.weight)


class AscendInkling2ShortConv(nn.Module):
    """Hidden-stream static short convolution (attention or MLP output).

    Offline mode: the state is the last ``W - 1`` inputs ``(W - 1, C)`` bf16.
    Engine mode: the state is a ``(conv_state, conv_metadata)`` pair over the
    paged conv-state cache.
    """

    def __init__(
        self, dim: int, kernel_size: int, normalize_values: bool, engine: bool = False, stream_idx: int = 2
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(dim, 1, kernel_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.normalize_values = normalize_values
        self.engine = engine
        self._stream_idx = stream_idx
        self._paged_taps = None  # built by process_weights_after_loading

    @property
    def conv_weight(self) -> torch.Tensor:
        return self.weight.squeeze(1)

    @property
    def paged_taps(self) -> torch.Tensor:
        """The taps for the paged kernel, lag-indexed ``(W, C)``."""
        if self._paged_taps is None:
            self.process_weights_after_loading()
        return self._paged_taps

    def process_weights_after_loading(self) -> None:
        """Lay the taps out for the paged convolution kernel."""
        from .ops_sconv import lag_taps

        self._paged_taps = lag_taps(self.conv_weight)

    def conv_mix(
        self,
        x: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: AscendInkling2RMSNorm | None,
        state,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Short-convolve ``x``; with ``addend``, into
        ``(residual + addend) + y`` in fp32. Then mix
        ``[*stack, (last +) out]`` with each pseudo-query: through ``norm``
        (bf16 mixes), or the fp32 mixes without it.

        ``out`` (may alias ``residual``) receives the convolution's output
        when given. Returns ``(mixes (Q, T, D), out (T, D))``.
        """
        if self.engine:
            conv_state, conv_md = state
            if conv_md is None:
                # No conv state (profile run): skip the conv, keep the residual sum.
                if addend is not None:
                    acc = addend.float() if residual is None else residual + addend.float()
                    x = (acc + x.to(addend.dtype).float()).to(addend.dtype)
                y = x if out is None else out.copy_(x)
                mixes = attn_res(
                    stack,
                    y if last is None else last,
                    None if last is None else y,
                    queries,
                    None if norm is None else norm.weight,
                    0.0 if norm is None else norm.eps,
                    self.normalize_values,
                )
                return mixes, y
            stream_idx = self._stream_idx
            offset, width = conv_state.stream_ranges[stream_idx]
            from .ops_sconv import short_conv

            y = short_conv(
                x,
                self.conv_weight,
                conv_state.kv_cache,
                positions,
                conv_md.block_table,
                conv_md.seq_idx,
                conv_md.slot_mapping,
                conv_md.query_start,
                offset,
                width,
                taps=self.paged_taps,
            )
        else:
            y = short_conv_static_seq(x, self.conv_weight, state)
        if addend is not None:
            acc = addend.float()
            if residual is not None:
                acc = residual.float() + acc
            y = (acc + y.float()).to(y.dtype)
        if out is None:
            out = y
        elif out.data_ptr() != y.data_ptr():
            out.copy_(y)
        mixes = attn_res(
            stack,
            out if last is None else last,
            None if last is None else out,
            queries,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            self.normalize_values,
        )
        return mixes, out


class _Linear(nn.Module):
    """A bias-free bf16 linear layer holding ``weight`` ``[out, in]``, as
    upstream's ``qkvr`` / ``wo_ud`` (so the weights keep their names)."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight)


class _KVConv(nn.Module):
    """The K or V short convolution of a layer: static taps (local layers),
    or per-token taps ``static_w + weight_proj(x_k)`` (full layers)."""

    def __init__(
        self,
        width: int,
        kernel_size: int,
        dynamic_input_size: int | None,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dynamic = dynamic_input_size is not None
        if self.dynamic:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, width, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(width, 1, kernel_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
        self._paged_taps = None  # built by process_weights_after_loading

    @property
    def conv_weight(self) -> torch.Tensor:
        """Static taps in the layout expected by the convolution kernels."""
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    @property
    def paged_taps(self) -> torch.Tensor:
        """The static taps for the paged kernel, lag-indexed ``(W, C)``."""
        if self._paged_taps is None:
            self.process_weights_after_loading()
        return self._paged_taps

    def process_weights_after_loading(self) -> None:
        """Lay the static taps out for the paged convolution kernel."""
        from .ops_sconv import lag_taps

        self._paged_taps = lag_taps(self.conv_weight, self.dynamic)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        """Project per-token dynamic taps, or return ``None`` for static convs."""
        if not self.dynamic:
            return None
        return dynamic_taps(x, self.weight_proj)

    def forward(self, x: torch.Tensor, dynamic_input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.dynamic:
            deltas = self.dynamic_taps(dynamic_input)
            return short_conv_dynamic_seq(x, self.static_w, deltas, self.num_heads, state)
        return short_conv_static_seq(x, self.weight.squeeze(1), state)


class AscendInkling2Attention(nn.Module):
    """Inkling2 attention of one decoder layer.

    Local layers project q/k/v from one input and use static K/V convolutions.
    Full layers project the query side (q, r, gate, sink) from ``x_q``, k from
    ``x_k`` and v from ``x_v``, with dynamic K/V convolutions and log-scaled
    q/r. Local layers given separate ``x_k`` / ``x_v`` (loop passes after the
    first) project them as full layers do, without the log scaling.
    """

    def __init__(self, config: Inkling2ModelConfig, is_local: bool) -> None:
        super().__init__()
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
        # Query-side outputs first (q, r, gate, sink), then k and v.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        hidden = config.hidden_size
        self.qkvr = _Linear(sum(self.split_sizes), hidden)
        self.wo_ud = _Linear(hidden, q_size)
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel, dtype=torch.bfloat16),
            requires_grad=False,
        )
        dynamic_input = None if is_local else hidden
        self.k_sconv = _KVConv(kv_size, config.sconv_kernel_size, dynamic_input, self.num_kv_heads)
        self.v_sconv = _KVConv(kv_size, config.sconv_kernel_size, dynamic_input, self.num_kv_heads)

    def shard_slice(self, shard: int) -> slice:
        """Rows of ``qkvr`` holding the checkpoint shard ``shard`` (0..5)."""
        start = sum(self.split_sizes[:shard])
        return slice(start, start + self.split_sizes[shard])

    def _project(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> list[torch.Tensor]:
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            return list(self.qkvr(x_q).split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = nn.functional.linear(x_q, weight[:q_side])
        k = nn.functional.linear(x_k, weight[q_side:k_end])
        v = nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.split_sizes[:4], dim=-1), k, v]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        start: int = 0,
        state: AscendInkling2LayerState | None = None,
    ) -> torch.Tensor:
        """bf16 `(T, hidden)` attention output.

        ``positions`` are the tokens' absolute positions: the state's keys
        ``[0, start + T)`` are attended. With ``tau``, the fp32 per-token
        log-scaling of q and r (full layers).
        """
        assert state is not None
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        tokens = x_q.shape[0]
        heads, head_dim, kv_heads = self.num_heads, self.head_dim, self.num_kv_heads
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)

        # K/V short convolutions, then K's L2 clip and V's weightless RMSNorm,
        # both into the KV state (rounded to bf16, as the KV cache stores them).
        k = self.k_sconv(k, x_k, state.conv["k"]).view(tokens, kv_heads, head_dim)
        v = self.v_sconv(v, x_v, state.conv["v"]).view(tokens, kv_heads, head_dim)
        k = l2_clip(k, None, self.qk_clip)
        values = v.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        v = values.to(torch.bfloat16)
        state.key[:, start : start + tokens] = k.transpose(0, 1)
        state.value[:, start : start + tokens] = v.transpose(0, 1)

        clipped_q = l2_clip(q.view(tokens, heads, head_dim), self.q_scale, self.qk_clip, tau)
        clipped_r = l2_clip(r.view(tokens, heads, self.d_rel), self.r_scale, self.rel_clip, tau)
        length = start + tokens
        bias = relative_bias(clipped_r, self.rel_logits_proj, positions, length, self.is_local)
        out, lse = attention(clipped_q, state.key[:, :length], state.value[:, :length], bias, self.scaling)
        out = gate_sink(out, lse, gate.view(tokens, heads, head_dim), sinks.view(tokens, heads))
        return self.wo_ud(out.reshape(tokens, heads * head_dim))


class _Gate(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32), requires_grad=False)


class _Experts(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w13 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )


class AscendInkling2MoE(nn.Module):
    """Routed-experts MoE with BF16 experts."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        assert config.n_shared_experts == 0 and config.use_gate_bias
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.gate = _Gate(config.n_routed_experts, config.hidden_size)
        self.experts = _Experts(config.n_routed_experts, config.hidden_size, config.intermediate_size)

    def process_weights_after_loading(self) -> None:
        """Keep the fp32 router weight transposed and the fp32 bias, so the
        per-visit logits GEMM does not recast them."""
        self._gate_fp32 = self.gate.weight.detach().float()
        self._gate_bias_fp32 = self.gate.bias.detach().float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        experts = self.experts
        return moe_forward(
            x,
            self.gate.weight,
            self.gate.bias,
            experts.w13,
            experts.w2,
            self.topk,
            self.route_scale,
            self.gate_cap,
            self.up_cap,
            gate_weight_fp32=getattr(self, "_gate_fp32", None),
            gate_bias_fp32=getattr(self, "_gate_bias_fp32", None),
        )


class AscendInkling2DecoderLayer(nn.Module):
    """One decoder layer. A loop core layer also holds a message controller
    per stream (``LOOP_STREAMS``), stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        engine: bool = False,
        prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.engine = engine
        self.is_local = layer_id in config.local_layer_ids
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.normalize_values = config.attn_res_normalize_values
        if engine:
            from .attention_backend import AscendInkling2Attention as EngineAttention
            from .attention_backend import AscendInkling2ConvState

            self.conv_state = AscendInkling2ConvState(
                num_kv_heads=config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=hidden,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state",
            )
            # Loop core layers get a conv-state cache per pass.
            pass_conv_states = ()
            if layer_id in config.loop_core_layers:
                pass_conv_states = tuple(
                    AscendInkling2ConvState(
                        num_kv_heads=config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads,
                        head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                        hidden_size=hidden,
                        kernel_size=config.sconv_kernel_size,
                        prefix=f"{prefix}.conv_state.pass{p}",
                    )
                    for p in range(1, config.num_loop_passes)
                )
            self.attn = EngineAttention(
                config,
                self.is_local,
                self.conv_state,
                prefix=f"{prefix}.attn",
                pass_conv_states=pass_conv_states,
            )
        else:
            self.attn = AscendInkling2Attention(config, self.is_local)
        # Local layers: attn_norm; full layers: q/k/v input norms, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = AscendInkling2RMSNorm(hidden, eps, num_inputs=self.num_attn_inputs)
        self.mlp_norm = AscendInkling2RMSNorm(hidden, eps)
        self.mlp = AscendInkling2MoE(config)
        self.attn_sconv = AscendInkling2ShortConv(
            hidden, config.sconv_kernel_size, self.normalize_values, engine=engine, stream_idx=2
        )
        self.mlp_sconv = AscendInkling2ShortConv(
            hidden, config.sconv_kernel_size, self.normalize_values, engine=engine, stream_idx=3
        )
        self.is_core = layer_id in config.loop_core_layers
        if self.is_core:
            rank = config.loop_config["gate_rank"]
            self.controller_down = nn.Parameter(
                torch.empty(len(LOOP_STREAMS), rank, 2 * hidden, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.controller_gain_forget = nn.Parameter(
                torch.empty(len(LOOP_STREAMS), 2 * hidden, rank, dtype=torch.bfloat16),
                requires_grad=False,
            )

    @property
    def conv_kernel_size(self) -> int:
        return self.attn.k_sconv.kernel_size

    def new_state(self, max_length: int, device: torch.device | str) -> AscendInkling2LayerState:
        """Zeroed KV and conv state for up to ``max_length`` tokens."""
        attn = self.attn
        width, kernel = attn.num_kv_heads * attn.head_dim, self.conv_kernel_size
        hidden = self.attn_sconv.weight.shape[0]

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        return AscendInkling2LayerState(
            key=zeros(attn.num_kv_heads, max_length, attn.head_dim),
            value=zeros(attn.num_kv_heads, max_length, attn.head_dim),
            conv={
                "k": zeros(kernel - 1, width),
                "v": zeros(kernel - 1, width),
                "attn": zeros(kernel - 1, hidden),
                "mlp": zeros(kernel - 1, hidden),
            },
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: torch.Tensor,
        tau: torch.Tensor | None,
        start: int,
        state: AscendInkling2LayerState | int,
    ) -> torch.Tensor:
        """Attention on the normalized ``inputs`` `(Q, T, D)`: the one input
        of a local layer, or separate q/k/v inputs (rows 0, 1, 2).

        Offline mode: ``state`` is an ``AscendInkling2LayerState``.
        Engine mode: ``state`` is the loop pass index (int).
        """
        if self.engine:
            pass_idx = state
            if self.is_local and inputs.shape[0] == 1:
                return self.attn(positions, inputs[0], pass_idx=pass_idx)
            return self.attn(
                positions,
                inputs[0],
                inputs[1],
                inputs[2],
                None if self.is_local else tau,
                pass_idx=pass_idx,
            )
        if self.is_local and inputs.shape[0] == 1:
            return self.attn(positions, inputs[0], start=start, state=state)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            start=start,
            state=state,
        )

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The ``streams`` message controllers on fp32 `(S, T, D)` ``dest``
        (may broadcast over streams) and ``message``, each output
        RMS-normalized with its row of ``norm_weight`` `(S, D)`: bf16 `(S, T, D)`.

        ``(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message``, from
        ``h = down(rms([dest, message]))``.
        """
        pair = pair_rms_norm(dest, message)
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        hidden = torch.bmm(pair, down.transpose(1, 2))
        gain_forget = bmm_fp32(hidden, up.transpose(1, 2))
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)
