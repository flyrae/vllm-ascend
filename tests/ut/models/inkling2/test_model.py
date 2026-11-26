# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU Inkling 2 model on a tiny looped configuration: the
forward (one prefill vs. prefill + decode and chunked prefill through the
cache) and the checkpoint mapping of ``load_weights``, with every kind of
checkpoint tensor of the real model."""

import random
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.inkling2 import reference
from vllm_ascend.models.inkling2.configs import Inkling2LoopMMConfig
from vllm_ascend.models.inkling2.layers import LOOP_STREAMS
from vllm_ascend.models.inkling2.model import AscendInkling2LoopForConditionalGeneration

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")

HIDDEN = 64
BLOCK = 3
EXPERTS, INTERMEDIATE = 8, 32
# The real model's attention shards, in ``Inkling2Attention.split_sizes`` order.
ATTN_SHARDS = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")
NORM_SHARDS = ("q_input_norm", "k_input_norm", "v_input_norm")


def _config(num_layers: int = 12, core: tuple[int, int] = (BLOCK, 3 * BLOCK)):
    """The real model's shape at a smaller scale: as many AttnRes blocks as
    layers of three, every layer local but the last of its block, and a loop
    core of two blocks."""
    return Inkling2LoopMMConfig(
        text_config=dict(
            vocab_size=256,
            unpadded_vocab_size=250,
            hidden_size=HIDDEN,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            d_rel=8,
            rel_extent=16,
            local_layer_ids=[i for i in range(num_layers) if (i + 1) % BLOCK],
            sliding_window_size=16,
            sconv_kernel_size=4,
            rms_norm_eps=1e-6,
            model_max_length=512,
            n_routed_experts=EXPERTS,
            num_experts_per_tok=2,
            n_shared_experts=0,
            intermediate_size=INTERMEDIATE,
            log_scaling_n_floor=8,
            log_scaling_alpha=0.1,
            qk_l2_clip_threshold=22.6,
            rel_l2_clip_threshold=8.0,
            n_layers_per_attn_res_block=BLOCK,
            attn_res_normalize_values=True,
            final_logit_softcapping=30.0,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
            route_scale=8.0,
            use_gate_bias=True,
            loop_config=dict(core_start=core[0], core_end=core[1], num_passes=2, gate_rank=8, is_varloop=False),
        ),
        vision_config=None,
    )


def _vllm_config(config, tensor_parallel_size: int = 1, dtype: torch.dtype = torch.bfloat16, **model_config):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=dtype,
            multimodal_config=None,
            enable_prompt_embeds=False,
            **model_config,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=tensor_parallel_size),
        lora_config=None,
        quant_config=None,
    )


def _model(config, device: str = "npu", seed: int = 0, engine: bool = False, **options):
    torch.manual_seed(seed)
    with torch.device(device):
        model = AscendInkling2LoopForConditionalGeneration(vllm_config=_vllm_config(config, **options), engine=engine)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:  # the router bias and the pseudo-queries
                param.normal_(0.0, HIDDEN**-0.5 if "bias" not in name else 0.0)
            elif param.dim() < 2:
                param.zero_()
            elif "norm" in name:
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.3)
    model.model.process_weights_after_loading()
    return model


def test_forward_decode_and_chunked_prefill_match_one_prefill():
    model = _model(_config())
    ids = torch.randint(0, 250, (16,), device="npu")

    full = model(input_ids=ids)
    assert full.shape == (16, HIDDEN) and torch.isfinite(full).all()
    logits = model.compute_logits(full[-1:])
    assert logits.shape == (1, 250) and torch.isfinite(logits).all()
    assert logits.abs().max() <= 30.0  # the softcap

    # Prefill, then single-token decode steps through the cache.
    cache = model.model.allocate_cache(32)
    head = model(input_ids=ids[:9], cache=cache)
    tail = torch.cat([model(input_ids=ids[t : t + 1], cache=cache) for t in range(9, 16)])
    assert cache.length == 16
    torch.testing.assert_close(torch.cat((head, tail)), full, atol=0, rtol=2**-7)

    # Chunked prefill.
    cache.reset()
    chunks = [model(input_ids=ids[a:b], cache=cache) for a, b in ((0, 5), (5, 6), (6, 16))]
    torch.testing.assert_close(torch.cat(chunks), full, atol=0, rtol=2**-7)


def test_forward_rejects_engine_arguments():
    """Outside the engine there is no cache behind the engine-style
    arguments: they are rejected rather than run as a stateless prefill."""
    model = _model(_config())
    with pytest.raises(NotImplementedError, match="unsupported forward keyword arguments"):
        model(input_ids=torch.zeros(4, dtype=torch.int64, device="npu"), attn_metadata=None)


def test_forward_needs_processed_weights():
    """The message queries are stacked by ``process_weights_after_loading``."""
    model = _model(_config())
    model.model.queries_ready = False
    with pytest.raises(AssertionError, match="process_weights_after_loading"):
        model(input_ids=torch.zeros(4, dtype=torch.int64, device="npu"))


def _experts(layer: int, nvfp4: bool):
    """Quantized expert tensors of one layer in the checkpoint layout, and the
    BF16 weights they must load to."""
    rows, width = 2 * INTERMEDIATE, HIDDEN
    out = []
    expected = {}
    for projection, (n, k) in (("w13", (rows, width)), ("w2", (width, INTERMEDIATE))):
        prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
        target = f"model.layers.{layer}.mlp.experts.{projection}"
        if nvfp4:
            values = torch.randint(0, 256, (EXPERTS, n, k // 2), dtype=torch.uint8)
            scales = (torch.rand(EXPERTS, n, k // 16) * 3 + 0.5).to(torch.float8_e4m3fn)
            amax, denominator = torch.rand(EXPERTS) + 0.5, torch.tensor([2688.0])
            out += [(prefix, values), (f"{prefix}_scale", scales), (f"{prefix}_amax", amax)]
            out += [(f"{prefix}_outer_scale_denominator", denominator)]
            expected[target] = reference.dequant_nvfp4(values, scales, amax, denominator).to(torch.bfloat16)
        else:
            values = torch.randn(EXPERTS, n, k).to(torch.float8_e4m3fn)
            scales = torch.randint(120, 135, (EXPERTS, n, k // 32), dtype=torch.uint8)
            out += [(prefix, values), (f"{prefix}_scale", scales)]
            expected[target] = reference.dequant_mxfp8(values, scales).to(torch.bfloat16)
    return out, expected


def _checkpoint(model):
    """The checkpoint of ``model`` under upstream's names: its parameters as
    the tensors ``load_weights`` reads (the experts quantized), and the values
    every parameter must hold afterwards."""
    tensors, expected = [], {}
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            continue
        value = param.detach().clone()
        expected[name] = value * (HIDDEN**-0.5 if ".attn_res." in name else 1.0)
        parts = name.split(".")
        if name == "lm_head.weight":
            tensors.append(("model.llm.unembed.weight", value))
        elif name.startswith("model.layers."):
            layer, rest = parts[2], ".".join(parts[3:])
            prefix = f"model.llm.layers.{layer}"
            if rest == "attn.qkvr.weight":
                attention = model.model.layers[int(layer)].attn
                for shard, checkpoint in enumerate(ATTN_SHARDS):
                    tensors.append((f"{prefix}.attn.{checkpoint}.weight", value[attention.shard_slice(shard)]))
            elif rest == "attn_norm.weight":
                shards = NORM_SHARDS if value.shape[0] == 3 else ("attn_norm",)
                for shard, checkpoint in enumerate(shards):
                    tensors.append((f"{prefix}.{checkpoint}.weight", value[shard]))
            elif rest in ("controller_down", "controller_gain_forget"):
                for stream, stream_name in enumerate(LOOP_STREAMS):
                    if rest == "controller_down":
                        tensors.append((f"{prefix}.controllers.{stream_name}.down_weight", value[stream]))
                    else:
                        tensors.append((f"{prefix}.controllers.{stream_name}.gain_weight", value[stream, :HIDDEN]))
                        tensors.append((f"{prefix}.controllers.{stream_name}.forget_weight", value[stream, HIDDEN:]))
            else:
                rest = (
                    rest.replace(".q_scale", ".q_l2_clip.scale")
                    .replace(".r_scale", ".rel_l2_clip.scale")
                    .replace(".attn.rel_logits_proj", ".attn.rel_logits_proj.proj")
                )
                tensors.append((f"{prefix}.{rest}", value))
        else:
            tensors.append((f"model.llm.{name.removeprefix('model.')}", value))
    for layer in range(model.text_config.num_hidden_layers):
        # As the real checkpoint: NVFP4 experts, MXFP8 in the last layers.
        expert_tensors, expert_expected = _experts(layer, nvfp4=layer < 10)
        tensors += expert_tensors
        expected.update(expert_expected)
    return tensors, expected


def test_load_weights_maps_the_whole_checkpoint():
    """Every parameter of a model is restored from the checkpoint's tensors
    in upstream's layout (fused shards, renamed tensors, controllers, the
    quantized experts), whatever their order."""
    config = _config()
    source = _model(config, device="cpu", seed=1)
    tensors, expected = _checkpoint(source)
    random.Random(0).shuffle(tensors)

    target = _model(config, seed=2)
    loaded = target.load_weights(iter(tensors))

    assert loaded == {name for name, _ in target.named_parameters()}
    for name, value in expected.items():
        torch.testing.assert_close(target.get_parameter(name).cpu(), value, atol=0, rtol=0)
    target.model.process_weights_after_loading()
    torch.testing.assert_close(target.model.message_queries.cpu(), source.model.message_queries, atol=0, rtol=0)


@pytest.mark.parametrize(
    "drop, message",
    [
        ("attn.gate_attn.weight", "Unexpected|Missing"),  # a lost qkvr shard
        ("mlp.gate.bias", "Missing checkpoint tensors"),
    ],
)
def test_load_weights_rejects_incomplete_checkpoints(drop, message):
    config = _config()
    tensors, _ = _checkpoint(_model(config, device="cpu", seed=1))
    kept = [(name, value) for name, value in tensors if drop not in name]
    assert len(kept) < len(tensors)
    with pytest.raises(ValueError, match=message):
        _model(config, seed=2).load_weights(iter(kept))


def test_load_weights_rejects_incomplete_expert_tensors():
    config = _config()
    tensors, _ = _checkpoint(_model(config, device="cpu", seed=1))
    kept = [(name, value) for name, value in tensors if not name.endswith("w13_weight_amax")]
    with pytest.raises(ValueError, match="Incomplete expert tensors"):
        _model(config, seed=2).load_weights(iter(kept))


def test_load_weights_rejects_unexpected_tensors():
    model = _model(_config())
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        model.load_weights([("model.llm.layers.0.attn.typo.weight", torch.zeros(1))])


def test_loop_core_must_span_two_blocks():
    with pytest.raises(ValueError, match="two AttnRes blocks"):
        _model(_config(core=(4, 10)))
    with pytest.raises(ValueError, match="two AttnRes blocks"):
        _model(_config(core=(3, 6)))  # one block: no two-block message bank


def test_unsupported_options():
    config = _config()
    with pytest.raises(NotImplementedError, match="single device"):
        _model(config, tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="bfloat16"):
        _model(config, dtype=torch.float32)
    vllm_config = _vllm_config(config)
    vllm_config.lora_config = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="LoRA"), torch.device("npu"):
        AscendInkling2LoopForConditionalGeneration(vllm_config=vllm_config, engine=False)


def test_compute_logits_paths_agree():
    """A few rows (every decode step) take the bf16-head kernel and more rows
    the fp32 GEMM; both give the same softcapped logits."""
    model = _model(_config())
    hidden = (torch.randn(17, HIDDEN) * 0.5).to(torch.bfloat16).to("npu")
    few = model.compute_logits(hidden[:1])
    many = model.compute_logits(hidden)
    torch.testing.assert_close(few, many[:1], atol=1e-5, rtol=1e-4)
