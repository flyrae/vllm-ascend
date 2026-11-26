# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini support on Ascend NPU.

NPU implementations of the kernels and decoder layers of upstream
``vllm.models.inkling2``. The model class is registered as
``AscendInkling2LoopForConditionalGeneration`` and overrides the upstream
architecture through the plugin model registry.

The kernels follow upstream's math (see each module's docstring for the
upstream kernel it replaces); the unit tests check them against the reference
implementations of the upstream tests. The engine integration uses paged KV
and conv-state caches with CANN attention and grouped MoE on the vLLM v0.34.0
compatibility worktree. An explicit ``AscendInkling2DecodeCache`` remains
available for engine-free runs.
"""

from transformers import AutoConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY

from .attention import attention, bmm_fp32, gate_sink, l2_clip, relative_bias
from .attn_res import attn_res
from .cache import AscendInkling2DecodeCache, AscendInkling2LayerState
from .configs import Inkling2LoopMMConfig
from .dequantize import dequantize_mxfp8, dequantize_nvfp4
from .layers import AscendInkling2DecoderLayer
from .loop_ops import blend_rms_norm, pair_rms_norm
from .model import AscendInkling2LoopForConditionalGeneration, AscendInkling2LoopModel
from .moe import moe_forward, moe_route
from .norm import rms_norm
from .quant_config import Inkling2QuantConfig
from .sconv import dynamic_taps, short_conv_dynamic_seq, short_conv_static_seq

AutoConfig.register("inkling2_loop", Inkling2LoopMMConfig, exist_ok=True)
_CONFIG_REGISTRY["inkling2_loop"] = Inkling2LoopMMConfig

__all__ = [
    "AscendInkling2DecodeCache",
    "AscendInkling2DecoderLayer",
    "AscendInkling2LayerState",
    "AscendInkling2LoopForConditionalGeneration",
    "AscendInkling2LoopModel",
    "Inkling2QuantConfig",
    "attention",
    "attn_res",
    "blend_rms_norm",
    "bmm_fp32",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dynamic_taps",
    "gate_sink",
    "l2_clip",
    "moe_forward",
    "moe_route",
    "pair_rms_norm",
    "relative_bias",
    "rms_norm",
    "short_conv_dynamic_seq",
    "short_conv_static_seq",
]
