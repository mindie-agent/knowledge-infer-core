---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - Step-3.7-Flash 197B multimodal MoE, GPT_OSS 120B/20B, and Pangu-7B inference
domain_path: infer-core/models/emerging-llm-and-rec/step3p7-gpt-oss-and-pangu
evidence_strength: reproduced_empirical
id: unit-step3p7-flash-gpt-oss-and-pangu-7b-npu-inference-recipe
inherits_from:
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/step3p7_flash/README.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/gpt-oss/gpt_oss_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/pangu_7b/README.md
  source_digest: bc55fe39c0a6615159756544307ab2b22bd8380711ca9a515819736e0c27995f
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:models/step3p7_flash/README.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: NPU optimization guide for Step-3.7-Flash (197B hybrid full/sliding attention
  + EP16 288-expert MoE + SwiGLU clamp), GPT-OSS (120B/20B MoE TP + static KV cache),
  and Pangu-7B.
tier: L2
title: Step-3.7-Flash (197B Multimodal MoE), GPT-OSS (120B/20B), and Pangu-7B NPU
  Inference Recipe
triggers:
  error_signatures:
  - step 3.7 flash npu inference
  - gpt oss 120b 20b ascend optimization
  keywords:
  - step 3.7 flash npu inference
  - gpt oss 120b 20b ascend optimization
  - pangu 7b npu inference
  - swiglu clamp limit 7 16
  - hybrid full sliding window rope kvcache
  - step-3.7-flash
  - gpt-oss
  - pangu-7b
  - hybrid-sliding-attention
  - swiglu-clamp
  - ep16
  - npugraph-ex
---
## 1. Step-3.7-Flash (`197B` Multimodal MoE) on Atlas A3 (`16` Ranks)

`Step-3.7-Flash` (`45` decoder layers: `0..2` Dense, `3..44` MoE with `288` experts `top-8`, `BF16 ~394 GB`) combines multimodal vision injection with a hybrid text backbone:
1. **Dual-Pool Hybrid Attention (`12` Full Layers + `33` Sliding Layers)**:
   - **Full Layers** (1 in every 4 layers): `GQA Q=64/KV=8`, `head_dim=128`, `rope_theta=5e6`, `partial_rotary_factor=0.5` with Llama-3 YaRN scaling.
   - **Sliding Layers** (33 layers): `GQA Q=96/KV=8`, `head_dim=128`, `window=512`, `rope_theta=1e4`, `partial_rotary_factor=1.0`.
   - Both layer types apply per-head Q/K RMSNorm (`head_dim=128`) and head-wise sigmoid output gating (`attn_out * sigmoid(gate)`).
   - **Optimization**: Maintain two independent RoPE cos/sin tables and two separate KV Cache pools (`full_kv_pool` vs. circular/windowed `sliding_kv_pool`).
2. **Module-Specific Parallel Partitioning (`16` Ranks across `8` Atlas A3 Cards)**:
   - `Attention`: `attn_tp=1` (`DP16`) to maximize throughput without KV duplication.
   - `MoE (Layers 3..44)`: `moe_tp=1`, `EP=16` (`288 / 16 = 18` experts/rank, satisfying the `<= 24` experts/rank kernel constraint), `FP32` sigmoid + `router_bias` renormalization with `scaling=3.0`.
   - `Dense FFN (Layers 0..2)`: `dense_tp=8`; `Embed / LMHead`: `tp=16`.
3. **Layer-Specific SwiGLU Activation Clamping**:
   - Layers `43` and `44` require numerical clamping inside SwiGLU (`MoE limit=7.0`, `shared expert limit=16.0`) to prevent `BF16` overflow in final decoder layers.
4. **Decode Graph Capture (`npugraph_ex`)**:
   - Capturing the decode step into `aclgraph` reduces single-step decode latency from `~85 ms` (eager) to `~28 ms` (`3.0x` speedup).

## 2. `GPT_OSS` (`120B` 8-Card / `20B` Single-Card) & `Pangu-7B` Optimization

1. **`GPT_OSS` Attention & MoE TP (`gpt_oss_optimization.md`)**:
   - Merges `q_proj`, `k_proj`, `v_proj` into `qkv_proj` with `num_kv_heads_per_rank = max(num_kv_heads // tp_size, 1)`.
   - Merges MoE `gate_proj` and `up_proj` into `gate_up_proj`, applies `npu_swiglu` (with custom clamp/alpha where specified by `GPT_OSS` gated activation), and pre-allocates static KV cache updated via `torch_npu.scatter_update`.
2. **`Pangu-7B` Dense Pipeline (`models/pangu_7b/README.md`)**:
   - Standardizes `npu_add_rms_norm`, `npu_rotary_mul`, `npu_fused_infer_attention_score`, and `npugraph_ex` static graph replay on single-card or multi-card Atlas A2/A3.

## Verification Checklist

- On `Step-3.7-Flash`, verify SwiGLU clamping (`limit=7` for routed experts, `limit=16` for shared experts) is active on layers `43` and `44`.
- Confirm `partial_rotary_factor=0.5` is applied only to the 12 Full Attention layers while the 33 Sliding Attention layers use full rotary (`1.0`).
