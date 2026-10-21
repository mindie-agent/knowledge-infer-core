---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - Gemma-4 26B-A4B and Tencent Hy3 295B-A21B inference on 8-card / 16-rank NPU nodes
domain_path: infer-core/models/emerging-llm-and-rec/gemma4-and-hy3-agentic
evidence_strength: reproduced_empirical
id: unit-gemma4-and-hy3-agentic-four-stage-npu-optimization-recipe
inherits_from:
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/gemma_4/agentic/optimization_report.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/hy3/agentic/optimization_report.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/hy3/hy3_optimization.md
  source_digest: 78c58dc1a26f248e7811d6690e75f93d58e74321c3192ddf413509e252bff712
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:models/gemma_4/agentic/optimization_report.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Empirical 4-stage optimization playbook (Parallelism -> Static KV Cache ->
  Fused Operators -> Graph Mode) for Gemma-4 (dual-mode sliding/full attention with
  K==V and parallel Dense+MoE) and Hy3 (295B MoE on 16-rank Atlas A3 & Ascend 950).
tier: L2
title: Gemma-4 (26B-A4B) and Tencent Hy3 (295B-A21B) Four-Stage Agentic Optimization
  Playbook
triggers:
  error_signatures:
  - gemma 4 ascend optimization report
  - hy3 295b moe ascend optimization
  keywords:
  - gemma 4 ascend optimization report
  - hy3 295b moe ascend optimization
  - dual mode sliding full attention kvcache
  - attention k eq v shared kv projection
  - parallel dense mlp and moe addition
  - gemma-4
  - hy3
  - hunyuan
  - dual-mode-attention
  - k-eq-v
  - mc2
  - torchair
  - npugraph-ex
---
## 1. Gemma-4 (`26B-A4B`) Architectural Adaptation (`models/gemma_4/agentic/optimization_report.md`)

Gemma-4 (`30` layers, `128` experts `top-8`, `GEGLU`, `vocab=262144` tied with LM Head) introduces three non-standard mechanisms that require dedicated NPU adaptations:
1. **Dual-Mode Heterogeneous KV Cache (`25` Sliding Layers + `5` Full Layers)**:
   - Sliding layers use `head_dim=256`, `sliding_window=1024`; Full layers (every 6th layer) use `head_dim=512`.
   - **Rule**: Allocate two separate static KV Cache pools indexed by layer type (`sliding_kv_pool` with `D=256` and `full_kv_pool` with `D=512`) instead of padding all 30 layers to `D=512` (saves 41% KV memory and bandwidth).
2. **`attention_k_eq_v` on Full Attention Layers**:
   - Full layers share the exact same projection for Key and Value (`K == V`), omitting `v_proj` entirely.
   - Pass the same `key_cache` tensor reference as both `key` and `value` inputs to `npu_fused_infer_attention_score` without duplicating HBM writes during `scatter_update_`.
3. **Unit Softmax Scale (`softmax_scale=1.0`) & Parallel `Dense MLP + MoE` Branch**:
   - Because Q/K RMSNorm normalizes head vectors prior to attention, set `scale=1.0` in Flash Attention instead of `1 / sqrt(head_dim)`.
   - Every layer computes **both** a `Dense MLP` (`intermediate=2112`, `GELU`) and a `MoE` block (`128` experts, `intermediate=704`, `GEGLU`) in parallel and sums their outputs (`h = h + dense_mlp(x) + moe(x)`), accompanied by `7 x RMSNorm + layer_scalar` per block.
   - With `TP8` + Static Dual-Pool KV + `GMM` + `aclgraph`, Atlas A3 8-card decode reaches `~10 ms/step` and prefill `~76 ms`.

## 2. Tencent Hy3 (`295B-A21B`) Optimization on Atlas A3 & Ascend 950 (`hy3/optimization_report.md`)

Hy3 (`80` layers: Layer 0 Dense FFN, Layers 1–79 MoE with `192` routed experts + `1` shared expert, `GQA 64Q/8KV`, `head_dim=128`, sigmoid routing `npu_moe_gating_top_k(norm_type=1)`) runs on `8 cards / 16 ranks` (`Atlas A3`):
1. **Hybrid Parallelism**:
   - Attention `TP=4` (`DP=4`), MoE `EP=16` (`12` experts per rank) with `GMM` (`split_item=2`).
2. **Eager vs. TorchAir (`ge_graph`) MC2 Dispatch**:
   - Eager mode uses manual `all_to_all`; `ge_graph` mode switches to fused `MC2` (`npu_moe_distribute_dispatch_v2` / `combine_v2`) and `torchair` FlashAttention, eliminating host-side dynamic metadata sync.
3. **Ascend 950PR Further Tuning (`docs/models/hy3/hy3_optimization.md`)**:
   - Enables quantized `GMM_Swiglu_Quant` (`W8A8` / `MXFP8`) and fuses `QKV + QK_RMSNorm + RoPE` to saturate 950PR Cube/Vector pipelines.

## Verification Checklist

- On Gemma-4, verify `softmax_scale=1.0` and `k_cache` reuse for `value` on all 5 Full Attention layers (`layers 5, 11, 17, 23, 29`).
- On Hy3, confirm `npu_moe_gating_top_k` uses `norm_type=1` (sigmoid gating) rather than softmax.
