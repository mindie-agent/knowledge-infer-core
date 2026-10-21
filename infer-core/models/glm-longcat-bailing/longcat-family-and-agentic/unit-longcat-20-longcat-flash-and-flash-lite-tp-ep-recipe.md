---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - LongCat-2.0, LongCat-Flash, and LongCat-Flash-Lite (68.5B-A3B) inference
domain_path: infer-core/models/glm-longcat-bailing/longcat-family-and-agentic
evidence_strength: reproduced_empirical
id: unit-longcat-20-longcat-flash-and-flash-lite-tp-ep-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/longcat_flash/longcat_flash_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/longcat-2.0/longcat_2.0_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/longcat_flash_lite/agentic/optimization_report_ep.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/longcat_flash_lite/agentic/optimization_report_tp.md
  source_digest: 92fcd4af300d941db9f90da22805b46a10e8628b1f4a2404bce8613d945b65ba
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/longcat_flash/longcat_flash_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Optimization recipe for the LongCat MoE family (LongCat-2.0, LongCat-Flash,
  LongCat-Flash-Lite) covering Shortcut-connected MoE, zero-computation identity experts,
  and empirical TP8 vs. EP8 crossover analysis.
tier: L2
title: LongCat-2.0, LongCat-Flash, and LongCat-Flash-Lite Agentic TP vs. EP Optimization
  Recipe
triggers:
  error_signatures:
  - longcat 2.0 optimization ascend
  - longcat flash shortcut moe
  keywords:
  - longcat 2.0 optimization ascend
  - longcat flash shortcut moe
  - longcat flash lite tp vs ep
  - zero computation identity expert moe
  - mla kv lora rank 512 longcat
  - longcat-2.0
  - longcat-flash
  - longcat-flash-lite
  - zero-computation-expert
  - shortcut-moe
  - tp-vs-ep
  - aclgraph
---
## 1. LongCat Architectural Characteristics (`LongCat-Flash` & `LongCat-2.0`)

1. **Shortcut-Connected MoE (`ScMoE`) & Dual-Sublayer Structure**:
   - Each transformer block contains two sequential Attention + MoE sublayers (`Sublayer 0` and `Sublayer 1`), where `Dense MLP` / Shared projections overlap directly with cross-rank MoE dispatch/combine via a shortcut connection.
2. **Zero-Computation (Identity) Experts**:
   - In addition to standard routed FFN experts (`N_real` experts, e.g., `256` or `512`), the router includes `N_zero` identity experts (`zero experts`, e.g., `128` in `LongCat-Flash-Lite` out of `384` total routing targets, `top-12`).
   - Tokens routed to indices `< N_real` execute `GMM + SwiGLU`, while tokens routed to indices `>= N_real` bypass FFN computation (`identity(x) * routing_weight`), dynamically adapting per-token FLOPs.
3. **MLA Attention with Scaled Q/K Projections**:
   - Uses `kv_lora_rank=512`, `q_lora_rank=1536`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`.

## 2. Zero-Expert & Routing Kernel Fusion

- **Anti-Pattern**: Materializing zero-weight dummy matrices for the `N_zero` identity experts wastes HBM bandwidth and `GroupedMatmul` FLOPs; conversely, using dynamic boolean indexing (`nonzero()`) breaks `aclgraph` static capture.
- **Optimal NPU Implementation**:
  1. Compute top-$k$ routing over all `N_real + N_zero` experts via `npu_moe_gating_top_k_softmax`.
  2. Partition routed indices into **Real Expert Tokens** (`idx < N_real`) and **Identity Shortcut Tokens** (`idx >= N_real`).
  3. Sum the routing weights of all selected identity experts per token into a single scalar multiplier $w_{	ext{zero}}(t) = \sum_{j \in 	ext{zero}} g_{t,j}$ and fuse $w_{	ext{zero}}(t) \cdot x_t$ into the residual accumulation without dispatching those tokens through `GMM` or `AlltoAll`.
  4. Pass only real expert counts `expert_tokens[:N_real]` to `npu_grouped_matmul` (`split_item=2`).

## 3. Empirical `TP8` vs. `EP8` Crossover (`LongCat-Flash-Lite` 68.5B-A3B on Atlas A3)

The agentic optimization reports (`optimization_report_tp.md` vs. `optimization_report_ep.md`) establish a quantitative decision boundary on 8 cards (`B=4, Seq=128`):
- **Eager Baseline (`TP8`)**: Prefill `671.8 ms`, Decode `161.2 ms/step`.
- **With Static KV Cache + Fused Ops (`FIA` + `GMM` + `npu_rms_norm`)**:
  - `TP8` Eager: Prefill `77.2 ms`, Decode `52.1 ms`.
  - `EP8` Eager (manual `all_to_all_single` twice per sublayer = `4x` per layer across `28` layers): Prefill `95.3 ms`, Decode `104.3 ms`.
- **With Graph Mode (`npugraph_ex` + `static_kernel`)**:
  - **`TP8` Graph**: Prefill `77.9 ms`, **Decode `17.5 ms`** (`9.2x` total decode speedup; `1.13x` faster than `EP8`).
  - **`EP8` Graph**: Prefill `87.3 ms`, **Decode `19.8 ms`** (`8.1x` total decode speedup).
- **Engineering Rule**: For models that fit in 8-card memory (`~137 GB` BF16 across `8 x 64 GB`) at small batch sizes (`B <= 8`), `TP8` (`2` `AllReduce` per sublayer) outperforms `EP8` when `MC2` fused dispatch/combine is not enabled because `all_to_all_single` incurs `4` collectives plus host/device split synchronization per layer. Use `EP` only when enabling `MC2` (`npu_moe_distribute_dispatch_v2`) or scaling to large batch / multi-node deployments.

## Verification Checklist

- Confirm `expert_tokens` passed to `npu_grouped_matmul` excludes the `N_zero` identity expert slots while preserving their weighted identity sum in the output residual.
