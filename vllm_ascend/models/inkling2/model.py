# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini (looped checkpoints) on Ascend NPU.

``AscendInkling2LoopForConditionalGeneration`` is a self-contained NPU
implementation with upstream-compatible interface declarations
(``packed_modules_mapping``, ``hf_to_vllm_mapper``, the weight renames and
shards). It replaces construction, the forward path and weight loading with
NPU implementations built from the kernels of this package. The weights keep
upstream's names; the NVFP4 / MXFP8 experts are dequantized to BF16 while
loading.

Supported today: a single device (TP = PP = 1), bfloat16, text-only prompts,
both an explicit engine-free cache and the vLLM v0.34.0 engine's paged KV and
conv-state caches. Engine prefill, decode, chunked prefill and multi-request
batches have been exercised, eagerly and with decode-only ACL graphs (the
uncompiled model with ``cudagraph_mode=FULL_DECODE_ONLY``; prefill stays
eager). Not supported: LoRA, speculative decoding (DSpark), images (the
vision encoder's forward is not ported yet), TP/PP, and ACL graphs for
prefill or piecewise/mixed batches.

What keeps upstream's code:

- the weight renames and shards (``_renames``, ``_shards``) and interface
  declarations of ``Inkling2LoopForConditionalGeneration``
- the loop schedule of ``AscendInkling2LoopModel.forward`` and ``_run_layer``
  (upstream ``Inkling2LoopModel``), without its all-reduce and fused-kernel
  branches or DSpark aux hidden states; engine passes use the registered
  paged-cache metadata, while offline runs use ``AscendInkling2DecodeCache``
- ``_max_loop_blocks`` and ``process_weights_after_loading``

What is new on NPU: the cache-backed state, ``load_weights`` and the checkpoint
mapping (with the expert dequantization), ``compute_logits`` (the fp32 head
and softcap of upstream's ``LogitsProcessor`` as it runs off CUDA), and the
engine attention/cache backend.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.utils import WeightsMapper, _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .attn_res import attn_res
from .cache import AscendInkling2DecodeCache, AscendInkling2LayerState
from .configs import Inkling2MMConfig, Inkling2ModelConfig
from .dequantize import dequantize_mxfp8, dequantize_nvfp4
from .layers import (
    LOOP_STREAMS,
    AscendInkling2AttnRes,
    AscendInkling2DecoderLayer,
    AscendInkling2Embedding,
    AscendInkling2RMSNorm,
)
from .lm_head import MAX_ROWS, lm_head_fp32
from .processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .vision import Inkling2VisionEncoder


def _layer_index(name: str) -> int | None:
    """Index of ``layers``, ``attn_res.attn_queries`` or ``attn_res.mlp_queries``
    in a renamed checkpoint name."""
    parts = name.split(".")
    for parent in ("layers", "attn_queries", "mlp_queries"):
        if parent in parts:
            index = parts.index(parent) + 1
            if index < len(parts) and parts[index].isdigit():
                return int(parts[index])
    return None


class AscendInkling2VisionEncoder(Inkling2VisionEncoder):
    """The upstream vision encoder's weights on NPU.

    ``load_weights`` is upstream's. The encoder's forward runs CUDA kernels
    (rotary, patch gathering, flash attention); the NPU forward is not ported
    yet, so only text-only prompts are supported.
    """

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "The Inkling 2 vision encoder is not ported to NPU yet: only text-only prompts are supported."
        )


class AscendInkling2LoopModel(nn.Module):
    """Looped Inkling 2 decoder (upstream ``Inkling2LoopModel``) on NPU.

    The core layers ``[core_start, core_end)`` run ``num_passes`` times: a
    prelude, then each pass from the AttnRes state after the prelude, then the
    coda from the last pass's state. Each pass of a core layer has its own KV
    and conv state. In passes after the first, every core sublayer input is
    blended with a message: a normalized-value AttnRes read of the previous
    pass's blocks, merged by a gated low-rank controller.
    """

    def __init__(self, config: Inkling2ModelConfig, prefix: str = "model", engine: bool = False) -> None:
        super().__init__()
        self.engine = engine
        if config.loop_config is None:
            raise ValueError("AscendInkling2LoopModel needs a checkpoint with a loop_config (model_type inkling2_loop)")
        config.check_loop_options()
        core = config.loop_core_layers
        block_size = config.n_layers_per_attn_res_block
        if core.start % block_size or len(core) != 2 * block_size:
            # The cross-pass messages mix the previous pass's two blocks.
            raise ValueError(
                f"The loop core layers [{core.start}, {core.stop}) must span two "
                f"AttnRes blocks of {block_size} layers from a block boundary, as "
                "the Inkling 2 loop checkpoints do"
            )
        self.config = config
        self.hidden_size = hidden = config.hidden_size
        self.block_size = block_size
        self.normalize_values = config.attn_res_normalize_values
        self.loop_core = core
        self.embed_tokens = AscendInkling2Embedding(config.padded_vocab_size, hidden)
        self.embed_norm = AscendInkling2RMSNorm(hidden, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            AscendInkling2DecoderLayer(config, i, engine=engine, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = AscendInkling2AttnRes(config)
        self.norm = AscendInkling2RMSNorm(hidden, config.rms_norm_eps)
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), hidden)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the LOOP_STREAMS queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(len(self.loop_core), len(LOOP_STREAMS), hidden, dtype=torch.float32),
            persistent=False,
        )
        self.queries_ready = False

    def process_weights_after_loading(self) -> None:
        """Stack the per-stream message queries, scaled by ``D**-0.5``."""
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(-1, 2, self.hidden_size)
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack((queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1)
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)
        self.queries_ready = True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def allocate_cache(self, max_length: int, device: torch.device | str | None = None) -> AscendInkling2DecodeCache:
        """KV and conv state of one sequence, for engine-free runs."""
        return AscendInkling2DecodeCache(
            self.layers,
            self.loop_core,
            self.config.num_loop_passes,
            max_length,
            device or self.embed_tokens.weight.device,
        )

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        """fp32 ``1 + alpha * log(max((positions + 1) / n_floor, 1))``."""
        floor = self.config.log_scaling_n_floor
        if floor is None:
            return None
        ratio = torch.clamp((positions.float() + 1.0) * (1.0 / floor), min=1.0)
        return 1.0 + self.config.log_scaling_alpha * torch.log(ratio)

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: AscendInkling2RMSNorm | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        start: int,
        state: AscendInkling2LayerState,
    ) -> tuple[int, torch.Tensor]:
        """Run decoder layer ``i`` in loop pass ``pass_idx``.

        ``inputs`` are its attention inputs ``(Q, T, D)``: for a visit with
        messages from ``bank``, the fp32 mixes they blend into. Returns the
        number of completed AttnRes blocks after it, and the mixes of its
        output with ``next_query`` through ``next_norm`` (fp32 without it),
        the next visit's inputs.
        """
        layer = self.layers[i]
        partial = stack[num_blocks]
        messages = None
        if bank is not None:
            assert inputs is not None
            # All four streams' messages: a mix of the bank's blocks.
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(slice(0, 3), dests, messages[:3], norm_weight, norm.eps)
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = torch.stack([layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)])
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, start, state)

        # The MLP input mixes the blocks with (partial +) attn_out. With
        # messages it is the fp32 mix the MLP stream blends into.
        first = i % self.block_size == 0
        if self.engine:
            conv_state = layer.attn.conv_states[pass_idx]
            from .attention_backend import _conv_metadata

            conv_md = _conv_metadata(conv_state)
            attn_sconv_state = (conv_state, conv_md)
            mlp_sconv_state = (conv_state, conv_md)
        else:
            attn_sconv_state = state.conv["attn"]
            mlp_sconv_state = state.conv["mlp"]
        mlp_in, attn_out = layer.attn_sconv.conv_mix(
            attn_partial,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
            attn_sconv_state,
            positions=positions,
        )
        if messages is not None:
            mlp_in = layer.blend(slice(3, 4), mlp_in, messages[3:], layer.mlp_norm.weight, layer.mlp_norm.eps)

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        outputs, _ = layer.mlp_sconv.conv_mix(
            layer.mlp(mlp_in[0]),
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
            mlp_sconv_state,
            addend=attn_out,
            residual=residual,
            out=stack[num_blocks],
            positions=positions,
        )
        return num_blocks, outputs

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        cache: AscendInkling2DecodeCache | None = None,
    ) -> torch.Tensor:
        """Hidden states ``(T, hidden)`` of the ``T`` tokens.

        Without ``cache``, the tokens are one sequence at ``positions`` (a
        stateless prefill). With the cache of ``allocate_cache``, they continue
        its sequence: they must sit at positions
        ``[cache.length, cache.length + T)`` and are appended to it.
        """
        tokens, hidden = inputs_embeds.shape
        assert self.queries_ready, "call process_weights_after_loading() first"
        if cache is not None:
            cache.check_append(tokens, positions)
        core = self.loop_core
        stack = inputs_embeds.new_empty((self._max_loop_blocks() + 1, tokens, hidden))
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        start = 0 if cache is None else cache.length
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(self.config.num_loop_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        self.attn_res.attn_queries[i - 1],
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            if cache is None and not self.engine:
                state = self.layers[i].new_state(tokens, inputs_embeds.device)
            elif self.engine:
                state = pass_idx
            else:
                state = cache.state(i, pass_idx)
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                start,
                state,
            )
        assert inputs is not None
        if cache is not None:
            cache.length += tokens
        return inputs[0]


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class AscendInkling2LoopForConditionalGeneration(nn.Module, SupportsMultiModal):
    """NPU implementation of Inkling 2 Mini (looped variant).

    Overrides the upstream architecture through the plugin model registry. The
    upstream ``__init__`` builds NVIDIA-specific layers; this class builds the
    NPU modules of :mod:`vllm_ascend.models.inkling2` instead and dequantizes
    the quantized experts to BF16 in ``load_weights``.

    The weight renames and shards below mirror upstream vLLM's
    ``Inkling2ForConditionalGeneration`` (``nvidia/model.py``); the checkpoint
    names are identical to upstream's, so weights load unchanged.
    """

    # LoRA adapters use the checkpoint's module names.
    packed_modules_mapping: dict[str, str] = {}
    embedding_modules: dict[str, str] = {}
    hf_to_vllm_mapper = WeightsMapper()

    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", engine: bool = True) -> None:
        # Not the upstream __init__: it builds NVIDIA layers and registers them
        # in the forward context. The NPU modules are built here instead.
        nn.Module.__init__(self)
        self.engine = engine
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1 or parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("Inkling 2 on NPU supports a single device (TP = PP = 1)")
        if getattr(model_config, "enable_prompt_embeds", False):
            raise ValueError("Inkling2 does not support prompt embeddings")
        if getattr(vllm_config, "lora_config", None) is not None:
            raise NotImplementedError("Inkling 2 LoRA is not supported on NPU")
        if getattr(model_config, "dtype", torch.bfloat16) != torch.bfloat16:
            raise NotImplementedError("Inkling 2 on NPU runs in bfloat16")
        # A captured decode graph sizes its relative-bias buffer for every key
        # a request can hold. Past that range only the eager path is correct,
        # and the runner picks graphs by batch shape alone, so turn them off.
        compilation_config = getattr(vllm_config, "compilation_config", None)
        if self.engine and compilation_config is not None and compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            from .attention_backend import DECODE_GRAPH_MAX_KV_LEN

            if model_config.max_model_len > DECODE_GRAPH_MAX_KV_LEN:
                logger.warning(
                    "Inkling 2 decode ACL graphs cover at most %d tokens of context, got max_model_len=%d. "
                    "Running without graphs; lower max_model_len to enable them.",
                    DECODE_GRAPH_MAX_KV_LEN,
                    model_config.max_model_len,
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                compilation_config.cudagraph_capture_sizes = []
                compilation_config.max_cudagraph_capture_size = 0
        config: Inkling2MMConfig = model_config.hf_config
        self.config = config
        self.text_config = text_config = config.text_config
        multimodal_config = getattr(model_config, "multimodal_config", None)
        self.visual = (
            AscendInkling2VisionEncoder(config.vision_config, text_config.hidden_size)
            if config.vision_config is not None
            and (multimodal_config is None or multimodal_config.get_limit_per_prompt("image"))
            else None
        )
        self.model = AscendInkling2LoopModel(text_config, f"{prefix}model", engine=engine)
        self.lm_head = AscendInkling2Embedding(text_config.padded_vocab_size, text_config.hidden_size)
        self.org_vocab_size = text_config.vocab_size
        self.soft_cap = text_config.final_logit_softcapping

    def get_language_model(self) -> nn.Module:
        return self

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers:
            raise NotImplementedError("Inkling 2 DSpark (aux hidden states) is not supported on NPU yet")

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raise NotImplementedError("Inkling 2 vision (multimodal) is not ported to NPU yet")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache: AscendInkling2DecodeCache | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Hidden states ``(T, hidden)`` of the ``T`` tokens.

        ``cache`` (from ``self.model.allocate_cache``) continues a sequence
        outside the engine; without it the tokens are a stateless prefill at
        ``positions``. ``positions`` default to the sequence's next ones.
        """
        if intermediate_tensors is not None:
            raise NotImplementedError("Inkling 2 on NPU supports a single pipeline stage")
        if kwargs:
            # The engine normally supplies attention metadata through the
            # forward context; reject unknown explicit kwargs rather than
            # silently running a stateless decode.
            raise NotImplementedError(
                "Inkling 2 on NPU received unsupported forward keyword arguments "
                f"{sorted(kwargs)}; use the engine forward context or model.allocate_cache"
            )
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        if positions is None:
            start = 0 if cache is None else cache.length
            positions = torch.arange(start, start + inputs_embeds.shape[0], device=inputs_embeds.device)
        return self.model(positions, inputs_embeds, cache=cache)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The checkpoint's unembedding runs with an fp32 output
        (``head_dtype=float32``), then the softcap, as upstream.  A few rows
        (every decode step) take the bf16-head kernel; more rows take the
        fp32 GEMM."""
        weight = self.lm_head.weight[: self.org_vocab_size]
        if hidden_states.shape[0] <= MAX_ROWS:
            logits = lm_head_fp32(hidden_states, weight)
        else:
            logits = torch.nn.functional.linear(hidden_states.float(), weight.float())
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        return logits

    def map_checkpoint_name(self, checkpoint_name: str) -> tuple[str, str | None]:
        """Map a checkpoint tensor to ``(parameter name, shard)``.

        The names follow upstream's renames and shards. ``shard`` is None for
        a whole parameter; else the checkpoint shard of a fused parameter: the
        output group of ``attn.qkvr`` (upstream ``split_sizes`` order), an
        ``attn_norm`` input, a controller stream (``{stream}.gain`` /
        ``{stream}.forget`` for ``controller_gain_forget``), or
        ``experts.{tensor}`` for the tensors of a quantized expert projection.
        """
        for old, new in self._renames:
            checkpoint_name = checkpoint_name.replace(old, new)
        name = checkpoint_name
        prefix, _, last = name.rpartition(".")
        if ".mlp.experts." in name:
            # ...mlp.experts.{w13|w2}_{weight|weight_scale|...}
            projection, _, tensor = last.partition("_")
            return f"{prefix}.{projection}", f"experts.{tensor}"
        if ".controllers." in name:
            # ...controllers.{stream}.{down|gain|forget}_weight
            stream = prefix.rsplit(".", 1)[-1]
            kind = last.split("_")[0]
            if stream not in LOOP_STREAMS or kind not in ("down", "gain", "forget"):
                raise ValueError(f"Unexpected checkpoint tensor: {checkpoint_name}")
            layer = name[: name.index(".controllers.")]
            if kind == "down":
                return f"{layer}.controller_down", stream
            return f"{layer}.controller_gain_forget", f"{stream}.{kind}"
        for old, (new, shard) in self._shards.items():
            if old in name:
                return name.replace(old, new), str(shard)
        return name, None

    def shard_target(self, params: dict[str, nn.Parameter], name: str, shard: str | None) -> torch.Tensor:
        """The view of parameter ``name`` that checkpoint shard ``shard``
        loads into (the whole parameter without a shard)."""
        param = params[name].data
        if shard is None:
            return param
        if name.endswith(".attn_norm.weight"):
            return param[int(shard)]
        layer = self.model.layers[_layer_index(name)]
        if name.endswith(".attn.qkvr.weight"):
            return param[layer.attn.shard_slice(int(shard))]
        stream, _, kind = shard.partition(".")
        index = LOOP_STREAMS.index(stream)
        if kind == "":
            return layer.controller_down[index]
        # [gain; forget]: the gain weights in the first hidden rows, the
        # forget weights in the last ones.
        rows = slice(0, self.text_config.hidden_size) if kind == "gain" else slice(self.text_config.hidden_size, None)
        return layer.controller_gain_forget[index][rows]

    def _expected_shards(self, name: str) -> set[str]:
        """The checkpoint shards a fused parameter is loaded from."""
        if name.endswith(".attn.qkvr.weight"):
            return {str(i) for i in range(6)}
        if name.endswith(".attn_norm.weight"):
            local = self.model.layers[_layer_index(name)].is_local
            return set() if local else {"0", "1", "2"}
        if name.endswith(".controller_down"):
            return set(LOOP_STREAMS)
        if name.endswith(".controller_gain_forget"):
            return {f"{s}.{k}" for s in LOOP_STREAMS for k in ("gain", "forget")}
        return set()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, dequantizing the NVFP4 / MXFP8 experts to BF16.

        Returns the names of the loaded parameters. Raises on unexpected or
        missing checkpoint tensors.
        """
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        loaded: set[str] = set()
        # Checkpoint shards seen so far of each fused parameter.
        shards: dict[str, set[str]] = {}
        visual_weights: list[tuple[str, torch.Tensor]] = []
        # Quantized tensors of the expert projections not complete yet.
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for checkpoint_name, weight in weights:
            if checkpoint_name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((checkpoint_name.removeprefix("model.visual."), weight))
                continue
            name, shard = self.map_checkpoint_name(checkpoint_name)
            index = _layer_index(name)
            # Truncated models (fewer layers via hf_overrides) skip the rest.
            if index is not None and index >= num_layers - ("attn_queries" in name):
                continue
            if shard is not None and shard.startswith("experts."):
                tensors = pending.setdefault(name, {})
                tensors[shard.removeprefix("experts.")] = weight
                if self._load_experts(name, tensors, params, loaded):
                    del pending[name]
                continue
            if name not in params:
                raise ValueError(f"Unexpected checkpoint tensor: {checkpoint_name}")
            target = self.shard_target(params, name, shard)
            if shard is None and ".attn_res." in name:
                # The pseudo-queries are scaled by hidden_size**-0.5 at load time.
                target.copy_(weight.reshape(target.shape)).mul_(self.model.attn_res.scale)
            else:
                target.copy_(weight.reshape(target.shape))
            if shard is None:
                loaded.add(name)
            else:
                shards.setdefault(name, set()).add(shard)
        for name, tensors in pending.items():
            raise ValueError(f"Incomplete expert tensors of {name}: {sorted(tensors)}")
        for name, seen in shards.items():
            if seen == self._expected_shards(name):
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        self.model.queries_ready = False
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded

    def _load_experts(
        self,
        name: str,
        tensors: dict[str, torch.Tensor],
        params: dict[str, nn.Parameter],
        loaded: set[str],
    ) -> bool:
        """Load the expert projection ``name`` once its quantized tensors are
        complete, dequantizing NVFP4 / MXFP8 into the BF16 parameter in place.

        Returns whether the projection was loaded.
        """
        weight = tensors.get("weight")
        if weight is None:
            return False  # the format follows the weights: wait for them
        target = params[name].data

        def part(tensor: torch.Tensor) -> torch.Tensor:
            # float8 tensors are moved as raw bytes: NPU has no float8 support.
            if tensor.dtype == torch.float8_e4m3fn:
                return tensor.view(torch.uint8).to(target.device)
            return tensor.to(target.device)

        if weight.dtype == torch.uint8:  # NVFP4: two e2m1 per byte
            needed = {"weight_scale", "weight_amax", "weight_outer_scale_denominator"}
        elif weight.dtype == torch.float8_e4m3fn:  # MXFP8
            needed = {"weight_scale"}
        else:
            needed = set()
        if not needed <= tensors.keys():
            return False
        if weight.dtype == torch.uint8:
            dequantize_nvfp4(
                part(weight),
                part(tensors["weight_scale"]),
                part(tensors["weight_amax"]),
                part(tensors["weight_outer_scale_denominator"]),
                out=target,
            )
        elif weight.dtype == torch.float8_e4m3fn:
            dequantize_mxfp8(part(weight), part(tensors["weight_scale"]), out=target)
        else:
            target.copy_(weight)
        loaded.add(name)
        return True
