# Inkling 2 NPU 后续优化方案

状态基线（真实 checkpoint，单卡 910B2，2026-11-26 实测）：decode-only graph 模式 20.7 ms/步；1000 入 / 100 出（chunk=192, util=0.97）TTFT 4.9 s、端到端 6.7 s、输出 14.5 tok/s；eager 一步约 0.25 s（host 启动开销主导）。全量单测 133 passed。

每项按「现状依据 → 改法 → 验证 → 预期收益 → 工作量与风险」组织。优先级排序在文末。

---

## L1 融合消息控制器 blend 链（4 kernel → 1）

**依据**：`_run_layer` 中每个 looped 访问（每步 24 次）执行 `pair_rms_norm → bmm → bmm → blend_rms_norm`，即 `layers.py` 的 `blend`。实测每个环节图内各约 16 µs（kernel 下限），合计 24 × 4 × 16 µs ≈ 1.6 ms，占 20.7 ms 的约 8%。控制器 rank 仅 64，两个 bmm 计算量极小，纯属启动次数浪费。

**改法**：在 `loop_ops.py` 新增一个 Triton kernel，每 program 处理一个 (stream, token) 行：
`[dest, message]` 拼接行的 RMS → `down`（64×4608，行内单位步长）→ `gain/forget`（4608，rank 64）→ `(1-tanh(forget))*dest + tanh(gain)*message` → RMSNorm + `norm_weight` → bf16。
权重 `controller_down` / `controller_gain_forget` 在 `process_weights_after_loading` 里转成行主序缓存（参照 `paged_taps` 的做法）。`pair_rms_norm`、`blend_rms_norm` 保留作为参考实现和单测对照。

**验证**：新增单测「融合 blend vs 分步 blend」逐位对照（含 local 层 3 流广播和 full 层 4 流）；全量单测；tiny eager/graph logprob 差值 0.0；真实 checkpoint 21-token 参照输出一致。

**预期**：−1.3 ms/步（20.7 → 约 19.4）；eager prefill 每访问省 3 次启动。

**工作量/风险**：约 1 天。纯数值等价（fp32 中间量顺序不变），风险低。注意 triton-ascend 上标量规约不能与第二 program 维并存（见教程坑列表），行内规约用 1-D。

## L2 物化 loop 访问调度表

**依据**：`model.py` 的 `forward` 每次动态构建 48 元素的 `visits` 列表，循环内每次做 `next_query`/`next_norm`/`bank`/`first` 等分支判断。Python 骨架本身仅 5 µs，但每次访问的属性查找、切片和分支在 eager 模式叠加进每次层调用之间的 host 间隙——eager 每步约 0.25 s、每 chunk 约 0.35 s 全部由 host 启动主导，任何 Python 侧削减直接落到 prefill。

**改法**：`__init__`（或首个 forward）把 48 次访问物化成元组表：`(layer_idx, pass_idx, is_looped, first, next_query 参数 id, next_norm 参数 id, 是否重启 pass)`。`forward` 循环只查表；`bank` 的克隆和 `stack` 重启也用预分配缓冲（`clone()` → `copy_` 到固定 tensor）替代分配。

**验证**：全量单测（含 tiny 的「prefill+decode = 一次 prefill」等价测试）；真实参照输出。

**预期**：eager prefill 每 chunk 约 −3~5 ms（48 次访问的查找/分配摊薄）；图模式无变化（已捕获）。

**工作量/风险**：半天。风险极低，行为完全一致。

## L3 `loop_share_cache` 支持与验证

**依据**：config 已有 `loop_share_cache`（`configs.py:60`），上游语义是 3 个 pass 共享同一份 KV/conv-state 而非各持一份。当前引擎按 pass 分配了 3 份（`attention_backend.py` 的 `pass_caches`、每 pass 一个 `conv_state`），真实模型下这占了 KV 预算的约 2/3。实测 192@0.97 边界的 KV 仅剩 1,135 token（1100 需求 + 35 余量），**没有并发余量**是当前单卡最紧的约束。

**改法**：
1. `configs.py` 已校验 `loop_share_cache` 需要 `is_varloop=true`（真实 checkpoint 是 `false`，因此这是一条需要谨慎评估的路径——先与离线参考对照确认在该 checkpoint 上输出是否可接受）。
2. 引擎侧：`pass_caches` 为空 + `conv_states` 只留 1 份时，`_pass_cache`/`_conv_metadata` 已天然回退到共享（代码已写好），缺的是引擎构造路径把它打开 + `_run_layer` 里 state 的对应处理。
3. 若数值不可接受（很可能，pass 间 KV 会互相覆盖），放弃此项，改列为「不可行，原因记录」。

**验证**：真实 checkpoint 21-token 参照 + 1000-token 生成质量人工检查；KV 容量前后对比。

**预期**：若可行，KV 容量 ×3 → chunk 512+ 单卡可行 → TTFT 进入 2 s 量级；并发余量恢复。

**工作量/风险**：1~2 天（含验证）。**语义风险高**：上游默认关闭即因 pass 间状态互相污染，此项必须以输出质量为门禁，不达标即弃。

## L4 prefill 图（按 token bucket 捕获）

**依据**：prefill 每个 eager chunk 约 0.35 s 而其中设备时间不足 10 ms（图模式同样形状的 decode 访问已证明），其余全是 host。chunk=192 时 TTFT 4.9 s ≈ 6 个 chunk × 0.35 s + 首块调度。若 prefill 也走图，每 chunk 可降到接近设备时间。

**改法**：decode 图的机制全部复用（`_captured_decode_attention` 已按 `kv_cap` 固定 bias 形状；prefill 的差异是 T>1）。关键点：
- bias 形状 `[T, H, 1, kv_cap]` 随 bucket 变化，需按 token bucket（如 192/128/64）分别捕获；
- `actual_seq_lengths` 为累计 1..T 的 per-row 形式已验证安全（正是 eager 修复后的路径）；
- MoE 的 routing 路径图内已验证可捕获。

**验证**：每 bucket 的 eager vs 图 logprob 对照；真实参照输出；TTFT 前后对比。

**预期**：TTFT 4.9 s → 约 1 s 量级（6 chunk × 设备时间 + 调度）；同时 eager 模式整体受益消失（图模式专属）。

**工作量/风险**：2~3 天。中风险：bucket 数量与显存的权衡（每 bucket 一套图内存），以及 prefill 与 decode 图共存时的调度选择逻辑。

## L5 MoE 专家 INT8 化（精度决策项）

**依据**：MoE 两个 grouped GEMM 6.1 ms/步，读 8 个 BF16 专家约 1 TB/s，已接近带宽上限。BF16 专家 46 GiB 也是单卡显存紧张（chunk 边界 192）的根因。

**改法**：加载时反量化目标从 BF16 改为 INT8（per-channel scale），`npu_grouped_matmul` 走 INT8 权重 + antiquant 路径（vllm-ascend 的 W8A8 已有成熟管线可参照）。

**验证**：MoE 单测对照 BF16 参考的误差上界；真实生成质量（与 BF16 输出逐 token 对比 + 人工检查）；这是精度决策，需要你拍板可接受的误差。

**预期**：MoE GEMM 约 −40~50%（6.1 → 约 3.5 ms/步）；权重 46 → 约 23 GiB，chunk 边界大幅后移（估计 512+ 单卡可行，与 L3 互补）。

**工作量/风险**：3~5 天。精度风险需评估；收益最大（性能 + 显存双收）。

## L6 TP=2（吞吐与容量项，非 TTFT 优先）

**依据**：TTFT 的 TP 收益已可由 L4/L5 以更低成本获得。TP2 的真实价值在多请求吞吐（两卡并行 decode）与 KV 容量翻倍。单卡只暴露 1 张 NPU，本机无法验证。

**改法**（按依赖顺序）：
1. 通用并行线性层：`qkvr` → `MergedColumnParallelLinear`（6 段各自 ×tp）、`wo_ud` → `RowParallelLinear`（`reduce_results=False`）；
2. attention 按 head 分片（含自研 bias/gate_sink kernel 的 head 维）；
3. MoE 切分 + all-reduce；
4. **AttnRes stack/bank 的跨卡聚合**（`tensor_model_parallel_all_gather(stacked[...])`）与 hidden 流卷积的 RS/AG（上游 `SeqParallel`/`all_reduce_conv_mix`，第一版用朴素 HCCL 路径）；
5. conv-state 双规则分片（K/V 随 KV head；attn/mlp hidden 流按 hidden/tp）；
6. `load_weights` 按类分片；
7. 嵌入/lm_head 词表并行（自研 lm_head kernel 需加 gather）。

**验证**：每步两卡 vs 单卡 BF16 输出对照；21-token 金标准贯穿全程。

**预期**：多请求吞吐约 2×；TTFT 从 L4/L5 之后的水位再有限改善（前向设备时间减半，通信抵消一部分）。

**工作量/风险**：2~4 周。第 4 步（AttnRes/hidden 流语义）是主要风险集中区。

---

## 优先级与依赖

| 顺序 | 项 | 收益 | 成本 | 依赖 |
|---|---|---|---|---|
| 1 | L1 blend 链融合 | −1.3 ms/步（−6%） | 1 天 | 无 |
| 2 | L2 调度表物化 | prefill 每 chunk −3~5 ms | 半天 | 无 |
| 3 | L4 prefill 图 | TTFT 4.9 s → 约 1 s | 2~3 天 | 建议先做 L1/L2 减少图内 kernel 数 |
| 4 | L5 MoE INT8 | −2.6 ms/步 + 显存减半（解锁大 chunk） | 3~5 天 | **需精度拍板** |
| 5 | L3 loop_share_cache | KV ×3（若可行） | 1~2 天 | 输出质量门禁，随时可弃 |
| 6 | L6 TP=2 | 吞吐 2× | 2~4 周 | 需第二张卡；在 L4/L5 之后立项 |

串联后的量级预期：L1+L2 → decode 约 19.4 ms、1000/100 端到端约 6.4 s；L4 → TTFT 约 1 s、端到端约 2.5 s；L5 → decode 约 17 ms 且 chunk 512+ 单卡可行。

所有项共同的金标准验证链（已在用）：133 个单测 + tiny eager/graph logprob 逐位对照 + 真实 checkpoint 21-token 参照输出 `[200008, 371, 150409, 171960]` + 交错 batch 一致性。
