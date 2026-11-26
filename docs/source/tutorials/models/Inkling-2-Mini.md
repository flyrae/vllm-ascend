# Inkling-2-Mini

## 1 Introduction

Inkling 2 Mini is a looped multimodal Mixture-of-Experts model from Thinking Machines (26B total / 3.5B active parameters, text and image in, text out). The 12 middle decoder layers run three passes with shared weights; each pass keeps its own KV and short-convolution state. The layers are tied together by block attention residuals instead of a single residual stream.

Upstream vLLM implements the model in `vllm/models/inkling2` for NVIDIA GPUs. `vllm_ascend/models/inkling2` replaces its CUDA/Triton kernels and layers with NPU implementations and registers `AscendInkling2LoopForConditionalGeneration` for the `Inkling2LoopForConditionalGeneration` architecture.

This document describes the current status of the port, how to run the model, and what was validated.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

**Experimental (🔵), text only, single A2.** The full looped model loads the released checkpoint and generates text on a single Atlas A2 NPU, both through the vLLM engine and through an explicit engine-free cache. The vLLM v0.34.0 engine path has been tested with prefill, decode, chunked prefill and multi-request batches, eagerly and with decode-only ACL graphs. Not available yet:

- Images (the vision encoder is not ported), LoRA, DSpark speculative decoding
- Tensor/pipeline parallel, prefix caching and production serving validation
- ACL graphs for prefill, or for batches that mix prefill with decode (those run eagerly)

The engine integration uses the vLLM v0.34.0 (`62d4bc3c`) compatibility worktree. The released checkpoint's `config.json` and `hf_quant_config.json` are both needed; the small vLLM loader compatibility branch for Inkling 2 reads the latter instead of treating the former as the complete quantization description.

## 3 Prerequisites

| Item | Requirement |
| --- | --- |
| Hardware | one Atlas A2 64 GB NPU: the BF16 weights take 49 GiB of HBM, 46 GiB of them experts |
| Model weight | the released checkpoint (BF16, with NVFP4 experts in layers 0-19 and MXFP8 in layers 20-23) |
| Software | vLLM v0.34.0 (`62d4bc3c`) compatibility worktree and vllm-ascend built from this source tree; no vLLM main Inkling 2 checkout is required |

NPU has no float8 support, so the quantized experts are dequantized to BF16 while loading. Use `quantization="inkling2"` for the engine path; this selects the self-contained metadata handler, while the actual expert dispatch follows the checkpoint tensor dtypes. The engine-free path needs no quantization argument.

## 4 Running the Model Outside the Engine

The model takes the pieces of a `VllmConfig` it needs (the HF config and the parallel options) and an explicit cache for the sequence:

```python
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm_ascend.models.inkling2.configs import Inkling2LoopMMConfig
from vllm_ascend.models.inkling2.model import AscendInkling2LoopForConditionalGeneration

MODEL = "/path/to/Inkling-2-Mini"
config = Inkling2LoopMMConfig.from_pretrained(MODEL)
vllm_config = SimpleNamespace(
    model_config=SimpleNamespace(
        hf_config=config,
        dtype=torch.bfloat16,
        multimodal_config=None,
        enable_prompt_embeds=False,
    ),
    parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=1),
    lora_config=None,
)

with torch.device("npu"):
    model = AscendInkling2LoopForConditionalGeneration(vllm_config=vllm_config, engine=False)
with safe_open(f"{MODEL}/model-00001-of-00001.safetensors", framework="pt") as f:
    model.load_weights((name, f.get_tensor(name)) for name in f.keys())
model.model.process_weights_after_loading()

cache = model.model.allocate_cache(max_length=4096)
hidden = model(input_ids=prompt_ids, cache=cache)  # prefill (chunks append the same way)
logits = model.compute_logits(hidden[-1:])
next_id = logits.argmax(-1)
hidden = model(input_ids=next_id, cache=cache)  # one decode step
logits = model.compute_logits(hidden)
```

The chat format follows the checkpoint's template, for example `<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>...<|end_message|><|message_model|>`. The model answers with a `<|content_thinking|>` and a `<|content_text|>` message and ends with `<|content_model_end_sampling|>`.

## 5 Running Through the vLLM Engine

The engine path uses the paged KV and physical conv-state caches. The following is the configuration used for the real-checkpoint smoke test; `prompt_ids` are the token IDs produced by the checkpoint tokenizer:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    quantization="inkling2",
    skip_tokenizer_init=True,
    tokenizer_mode="hf",
    limit_mm_per_prompt={"image": 0},
    enforce_eager=True,
    gpu_memory_utilization=0.94,
    max_num_batched_tokens=32,
    max_model_len=512,
    max_num_seqs=1,
    enable_prefix_caching=False,
)
outputs = llm.generate(
    [{"prompt_token_ids": prompt_ids}],
    SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True, detokenize=False),
)
```

For the v0.34.0 worktree, keep the Inkling 2 loader compatibility branch in `vllm/model_executor/model_loader/weight_utils.py`: the brief ModelOpt dictionary in `config.json` must fall through to `hf_quant_config.json`. The custom `inkling2` quantization method is registered by `vllm_ascend.models.inkling2`.

### 5.1 Decode-Only ACL Graphs

Single-token decode steps can replay a captured ACL graph of the uncompiled model. Drop `enforce_eager=True` and set:

```python
llm = LLM(
    ...,
    compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
)
```

How it works and what it needs:

- **Capture.** vLLM captures one graph per decode batch size (up to `max_num_seqs`) before serving; `vllm-ascend`'s `ModelAclGraphManager` owns capture and replay, as for every other model. No `torch.compile` is involved.
- **Fixed shapes.** A replay cannot change tensor shapes, so the graph reads only persistent buffers the runner refreshes each step (positions, block tables, slot mappings, sequence lengths, the conv-state metadata and the caches). The one shape that depends on the request is the relative position bias, which the fused attention op takes as a dense `[B, heads, 1, keys]` tensor: a captured attention builds it over the fixed key range `[0, max_model_len]` (rounded up to 32) and masks it from the device sequence lengths, so every decode step of any request fits the same graph.
- **Replay update.** Each captured attention is registered with `vllm-ascend` (`register_task`); before every replay the update stream rebinds the step's query/key lengths and block table, so a graph captured with 1-token dummy sequences serves sequences of any length.
- **Prefill stays eager.** Prefill, chunked prefill and mixed batches run the eager path; a batch that fits no captured graph falls back to it too.

Limits:

- `max_model_len` must not exceed 32768. The dense bias buffer is sized for every key a request can hold, and the runner selects graphs by batch shape alone, so with a longer context the model disables graphs at startup with a warning and runs eagerly.
- Graphs cover single-token decode batches only. Speculative decoding is not supported for this model anyway.

Measured on one Atlas A2 64 GB (910B2), greedy decoding, decode-step latency taken as the slope over extra generated tokens:

| Weights | Concurrent requests | Eager | Decode-only graph |
| --- | --- | --- | --- |
| Inkling-2-Mini checkpoint | 1 | 250 ms/step | 20.7 ms/step |
| Inkling-2-Mini checkpoint | 2 | 344 ms/step | 38 ms/step |
| 4-layer test model (dummy weights) | 1 | 56 ms/step | 6.7 ms/step |
| 4-layer test model (dummy weights) | 4 | 67 ms/step | 7.6 ms/step |

The eager step is host-bound and its latency moved by 10-15% between runs of the same code on this host; the graph timings repeat within about 2%.

A longer prompt, prefilled in 32-token chunks (eager in both modes; median of 3):

| Prompt + output | Mode | First token | Decode | End to end | Output rate |
| --- | --- | --- | --- | --- | --- |
| 1000 + 100 | eager | 10.0 s | 226 ms/step | 32.4 s | 3.1 tok/s |
| 1000 + 100 | decode-only graph | 10.4 s | 21 ms/step | 12.3 s | 8.1 tok/s |

The first token is the chunk prefills. A larger `max_num_batched_tokens` cuts their number; measured on one device (eager starts, 1000+100 needs a KV budget of about 0.95 GiB):

| `max_num_batched_tokens` | `gpu_memory_utilization` | Starts | KV tokens | 1000-token TTFT |
| --- | --- | --- | --- | --- |
| 128 | 0.94 / 0.96 / 0.97 | yes | 1,211 / 1,383 / 1,468 | 5.2 s (0.97) |
| 192 | 0.96 | no (needs 0.84 GiB, 0.82 free) | — | — |
| 192 | 0.97 | yes | 1,135 | 4.9 s |
| 256 | 0.96 / 0.97 | no (needs 1.03 GiB, 0.82/0.87 free) | — | — |
| 512 / 1024 | 0.97 | no (needs 1.78 / 1.87 GiB) | — | — |

So 192 tokens at utilization 0.97 is the edge on one device: 6 chunks instead of 32, TTFT 10.4 -> 4.9 s and end to end 12.3 -> 6.6 s at 14.5 tok/s output. Past that only smaller weights (TP or quantized experts) create room.

Capturing the checkpoint's two graphs (1 and 2 requests) takes about 3 s and 0.45 GiB once Triton has compiled the kernels; the first start after a kernel source changes also pays that compilation inside the capture warmups (about 40 s per graph on this host). The graph removes the host-side launch overhead. What remains is device time: in a decode step of the checkpoint (48 layer visits) about 6 ms are the MoE GEMMs, which read the BF16 experts at about 1 TB/s and are close to the memory bound; most of the rest is the per-kernel floor of the small operations around them (see 6.5).

## 6 Relationship with the Upstream Implementation

`vllm_ascend/models/inkling2` contains four kinds of code. Each module's docstring names its upstream counterpart and what changed.

### 6.1 Used from Upstream, Unchanged

| What | Upstream source |
| --- | --- |
| `grouped_gemm`, the reference/fallback expert GEMM | `vllm/models/inkling2/nvidia/ops/grouped_gemm.py`: a Triton kernel without CUDA-only constructs, compiled as is by triton-ascend; the production NPU MoE path uses CANN grouped GEMM instead |
| Config classes and quantization metadata | `vllm/models/inkling2/configs.py`, `quant_config.py`, vendored so the v0.34.0 worktree is self-contained |
| Model interface: the weight renames and shards (`_renames`, `_shards`) and declarations | `nvidia/loop.py`, `nvidia/model.py`; copied into the self-contained NPU class rather than inheriting its CUDA constructor |
| Vision encoder weights and its `load_weights` | `nvidia/vision.py`; only its `forward` is replaced, by a `NotImplementedError` |

### 6.2 Upstream Code with NPU Substitutions

The algorithm is upstream's. The results can differ from an NVIDIA run in the last bf16 bit.

| NPU code | Upstream | Substitution |
| --- | --- | --- |
| `_rms_norm_kernel` (`norm.py`) | `ops/norm.py` | inline-PTX `rsqrt.approx.f32` replaced by `tl.rsqrt` |
| `_pair_rms_norm_kernel` (`loop_ops.py`) | `ops/loop.py` | same |
| `_blend_rms_norm_kernel` (`loop_ops.py`) | `ops/loop.py` | same, and `tldevice.tanh` replaced by `tl.math.tanh` |
| `_attn_res_kernel` (`attn_res.py`) | `ops/attn_res.py` | same, `tldevice.exp` replaced by `tl.math.exp`, no PDL wait; upstream's kernel for small batches is not ported |
| Loop schedule: `AscendInkling2LoopModel.forward`, `_run_layer`, `_max_loop_blocks`, `process_weights_after_loading` | `loop.py`, `Inkling2LoopModel` | Dense KV/conv state offline or paged KV/conv-state metadata in the engine; no all-reduce or DSpark aux hidden states |
| `AscendInkling2RMSNorm.forward`, `AscendInkling2DecoderLayer.attention`, `AscendInkling2DecoderLayer.blend` | `model.py`, `loop.py` | no LoRA, no SM12x branches |
| `_sconv_unit_stride_kernel`, the paged short convolution over the K/V and hidden streams (`ops_sconv.py`) | `short_conv` (`ops/sconv.py`) | Same math and tap order (bit-identical in the tests). One program per (token, head) with unit-stride loads and scalar branches, where upstream's tile kernel addresses every element through computed strides and masks: the AI Core runs those loads one element at a time, about 650 us per launch against 17 us |

### 6.3 Re-implemented in PyTorch from Upstream's Formulas

None of upstream's code is used here: upstream's kernels are CUDA/Triton and do not run on NPU. The math is upstream's, and the unit tests check it against the reference implementations of the upstream tests.

| NPU code | Upstream kernel |
| --- | --- |
| L2 clipping with `tau` (`attention.l2_clip`) | `_l2_clip` (`ops/norm.py`) |
| Relative position bias | `qr_clip_rel_tokens` (`ops/sconv.py`) |
| K/V short convolutions, key clip, value norm (`sconv.py`) | `_kv_conv_tile` (`ops/sconv.py`) |
| Attention with a natural-log LSE | `rel_attention` (`ops/attention.py`, `attention_sm90.py`), the FA4 kernel |
| Gate/sink epilogue, log-scaling `tau` | `_gate_sink_kernel`, `_log_scaling_tau_kernel` (`ops/attention.py`) |
| `conv_mix` | upstream's SM12x fallback (`short_conv` with the residual, then `attn_res`), not its fused kernel |
| MoE routing, SwiGLU, combine (`moe.py`) | `_select_experts`, `route`, `combine` (`ops/moe.py`), `_situ` (`ops/activation.py`) | The formulas and reference fallback are retained; NPU forward uses `npu_moe_gating_top_k`, `npu_moe_init_routing_v2`, `npu_grouped_matmul` and `npu_moe_token_unpermute`, with the routing weight inside SwiGLU. The weight of each expert-sorted row is found with a float32 `argsort` of the routing index, as vllm-ascend's MoE LoRA path does |

### 6.4 New on NPU

- `attention_backend.py`: engine cache owners, physical conv-page binding, per-pass KV caches, CANN TND attention with relative bias and LSE, CANN grouped MoE dispatch, and the decode-graph path (fixed-shape bias, registered attention tasks). The `_seq_metadata_kernel` / `sconv_seq_metadata` it uses (in `ops_sconv.py`) are upstream's, unchanged.
- `dequantize.py`: the BF16 experts. Upstream has no counterpart in the model code; its expert GEMMs read NVFP4 / MXFP8 directly.
- `cache.py`: the dense KV and conv state of one sequence. Upstream gets these tensors from the engine.
- `load_weights` and the checkpoint mapping, including the expert dequantization.
- `lm_head.py` and `compute_logits`: the fp32 head and the softcap of upstream's `LogitsProcessor`, as it runs off CUDA. The engine calls `compute_logits` outside the captured graph on every decode step; for up to 16 rows a Triton kernel reads the bf16 head once and accumulates in fp32, instead of copying the head to fp32 first.

### 6.5 What Differs from an NVIDIA Run

- **Experts**: BF16 weights and activations, where Blackwell quantizes the activations too (Hopper keeps them in BF16). NVFP4 weights are rounded to BF16; MXFP8 is lossless.
- **Attention**: the engine uses CANN fused attention over paged KV with additive relative bias; the engine-free reference path materializes the full score matrix. Same math, other floating-point rounding.
- **Approximations**: upstream's PTX `rsqrt`, `sigmoid` and `tanh` approximations are `tl.rsqrt` and `torch` functions here.
- **State**: offline runs use one sequence with a dense cache; engine runs use paged KV and conv-state caches, including independent loop-pass caches. Prefix caching, graphs for prefill and tensor/pipeline parallel are not supported; decode-only ACL graphs are.

Pitfalls an NPU port has to get right, found while porting:

- **Integer shifts are very slow on NPU** (a `<<` on 42M int32 elements took 0.76 s) and `exp2` is off by about 1e-7. The float8 decodes use float arithmetic and build powers of two from the fp32 exponent field, which is exact.
- **`argsort` of an integer tensor runs on the AI CPU**, about 1 ms for 16k elements on this host; of a float32 tensor it runs on the AI Core in about 60 us at any size. Cast indices to float32 first (exact below 2**24), as vllm-ascend's MoE LoRA and GDN builder do. Both variants can be captured in an ACL graph here; the difference is speed, and how it scales. (`npu_moe_init_routing_v2` returns either direction of the routing permutation through `row_idx_type`, but the unpermute needs the original-to-sorted one, so the sorted-to-original index still takes one argsort.)
- **A load whose address is not affine in the lane is scalarized on the AI Core.** The ported convolution addressed weights by a stride of `W` and cache columns through `// WS` and `% WS`; that costs one load per element, and the kernel took about 650 us whatever its grid (1024 or 2304 channels, 1 or 4 warps). With one program per (token, head), a contiguous `[W, C]` tap layout and scalar branches for the per-token choices it takes 17 us, at the per-kernel floor.
- **Every kernel in a captured graph costs at least about 16 us** on this host, be it a Triton kernel, a torch elementwise op or a CANN op. After the convolution fix, most of a decode visit's time outside the MoE GEMMs is that floor times the number of small ops.
- **`npu_fused_infer_attention_score` crashes on multi-query requests with `pse_shift`.** With TND input, a paged KV cache and a dense additive bias, some `(q, kv)` shapes fault the kernel (``q=32`` with ``kv`` 96-192 among them, whatever the mask mode) — which every chunked prefill past two 32-token chunks hits. One query row per request (cumulative ``actual_seq_lengths``) is safe at any ``kv`` the dense sweep covered (q=1, kv 1-2048), so both the eager and the graph path call it that way, with the causal mask inside the bias. ``-inf`` entries in the bias leave inf/nan in the operator's LSE output; masked keys carry ``MASK_VALUE`` instead (``exp(MASK_VALUE - lse)`` underflows to exactly 0 in fp32, so the softmax is unchanged).
- **`npu_moe_init_routing_v2` needs `active_expert_range`**: with its default (empty) the op crashed the process for 1 token and 128 experts. The call sites pass `[0, num_experts]`.
- **A Triton kernel given a tensor of another device reads an invalid address** and takes the device down with an AI Core error. The kernel wrappers assert that their inputs share one device.
- **Full-attention layers keep attending beyond `rel_extent`** with a zero bias; only the sliding-window layers mask those keys.
- **The routing weight goes before the down projection.** `npu_moe_token_unpermute` can apply it through `probs` (the vllm-ascend AllGather dispatcher's way), but in bf16 and after the down projection; against the upstream reference that costs 0.4-0.8% of the output, against 0.1% or less with the fp32 weight inside SwiGLU.
- The checkpoint's controller weights are `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`; they load into the stacked `controller_down` / `controller_gain_forget` parameters.

## 7 Validation

Unit tests in `tests/ut/models/inkling2/` (skipped without an NPU) check the NPU implementations against the reference implementations of the upstream tests (`tests/models/inkling2/test_core_ops.py`, `test_attention.py`, `test_moe.py` in vLLM):

- Dequantization (NVFP4 and MXFP8) against the upstream LUT reference; every e4m3 byte decodes as `torch`'s cast
- Short convolutions (static and dynamic taps), including chunked prefill and decode through the state
- Attention: L2 clipping with `tau`, the relative position bias (also beyond the extent), softmax attention with sinks and window for prefill, chunked prefill and decode
- The attention layer (local and full) with its KV and conv state, the attention-residual mixing, `conv_mix`, the loop controllers
- MoE routing and forward against the per-token reference, including the CANN grouped path; the whole checkpoint mapping of `load_weights`
- The model on a tiny configuration: prefill + decode and chunked prefill through the cache equal one prefill
- Paged static and dynamic short convolution across `3 + 4 + 1` prefill/chunk/decode chunks, with zero observed error against the sequence reference; a three-request batch after a strided prefill, decode batches reading the paged cache and a padding token; the residual update
- The LM head for 1, 3 and 16 rows against the fp32 GEMM of the same bf16 values, at the checkpoint's hidden size and the test models'
- The conv-state metadata kernel (each token's request and its first row) and the fixed-shape decode bias against their references
- Engine smoke with a tiny dummy checkpoint: one-token generation, two-request three-token generation, and forced 16-token chunked prefill
- Real-checkpoint engine generation: 21 prompt tokens plus four greedy output tokens exactly matched the engine-free result (`[200008, 371, 150409, 171960]`), eagerly and with decode-only graphs, with one request and with two concurrent requests
- Decode-only graphs against eager decoding on the tiny model, comparing every step's greedy token and logprobs: four concurrent requests, and four requests finishing at different steps so the batch shrinks 4 → 3 → 2 → 1 (the graphs pad the batch); also each request decoded alone against the same request inside the batch. All identical (logprob gap 0.0).
- With `max_model_len` past the graphs' key range, the model logs its warning, skips graph capture and generates eagerly.

With the released checkpoint on one Atlas A2 64 GB (910B2), greedy decoding:

- 776 parameters load in about 28 s (page cache warm); the model uses 49.2 GiB of HBM (about 48.6 GiB reported by the engine loader)
- "What is the capital of France?" → a short thinking message, then `<|content_text|>Paris.<|end_message|>` and the end token
- A 2575-token prompt (a code at its start, more than a sliding window of unrelated text, a question about the code at its end): the code is recalled
- The engine-free eager PyTorch/Triton reference path is about 0.44 s per decode step. Through the engine, one request decodes at about 327 ms/step eagerly and 30 ms/step with decode-only graphs; no end-to-end throughput target has been established.

No numerical comparison with the NVIDIA implementation is available (no NVIDIA GPU was at hand); the references are the upstream test references and the quality of the generated text.

## 8 Next Steps

Where a decode step of the checkpoint goes with graphs (30 ms for 48 layer visits; per-visit device times from a graph that repeats each stage):

| Stage | Per visit | Per step | Note |
| --- | --- | --- | --- |
| MoE GEMMs (w13, w2) | 128 us | 6.1 ms | Reads 8 BF16 experts per visit at about 1 TB/s: close to the memory bound |
| Relative bias of the attention | 88 us | 4.2 ms | Five small ops |
| SwiGLU with the routing weight | 72 us | 3.4 ms | The fp32 argsort and gather, then about twelve elementwise ops |
| Attention, `qkvr` and `wo_ud` | 70 us | 3.4 ms | Near the floor |
| Router logits, gating, routing, unpermute | 69 us | 3.3 ms | Includes a cast of the router weight to fp32 on every call |
| The four short convolutions | 66 us | 3.2 ms | At the floor |
| q/r clip and gate/sink | 57 us | 2.7 ms | torch elementwise ops |
| Key clip, value norm, KV write | 52 us | 2.5 ms | Three ops after the K and V convolutions |

1. **Fuse the small operations into Triton kernels.** Done: the relative bias (one kernel over the row-major projection), the SwiGLU with its routing weight, the K/V convolutions' epilogue (L2 clip / RMS norm and the KV-cache write inside the convolution kernel), the q/r clip with `tau`, the gate/sink epilogue, an fp32 copy of the router weight kept once, and the per-token batch expansion computed once per step instead of per layer visit: 30 -> 20.7 ms. On this triton-ascend, a second program dimension next to a scalar reduction fails GraphOptimize, so the rows-clip kernel runs on a flat 1-D grid; and the KV-cache write needs the attention metadata's slot mapping (block size 128), not the conv cache's (16) -- using the wrong one passes engine-internal comparisons and only shows against the offline reference.
2. **The eager path is host-bound, so prefill is too.** An eager decode forward takes about 0.33 s although its arithmetic is that of the 30 ms graph: a Triton launch costs about 350 us of host time here and a torch or CANN op 50-100 us, with about 60 ops per layer visit. The same fusions help; beyond that, graphs for chunked prefill need a bias shape per token bucket.
3. **The MoE GEMMs (6 ms) only shrink with smaller experts.** BF16 experts are a choice forced by NPU's missing FP8 compute; INT8 weights would halve both this time and the 49 GiB of weights, at a precision cost to evaluate.
4. Measure the 48 attention task updates that vllm-ascend runs on its update stream before every replay; their host cost is not in the table above.

Then: broader long-context and multi-request numerical comparisons, `vllm serve` validation, the vision encoder (rotary, patch gathering, flash attention kernels), DSpark and the other unsupported features.
