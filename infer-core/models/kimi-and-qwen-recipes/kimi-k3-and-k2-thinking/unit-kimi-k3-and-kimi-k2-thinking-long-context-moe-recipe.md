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
  - Kimi-K3 ultra-large MoE inference
  - Kimi-K2-Thinking long chain-of-thought reasoning
domain_path: infer-core/models/kimi-and-qwen-recipes/kimi-k3-and-k2-thinking
evidence_strength: reproduced_empirical
id: unit-kimi-k3-and-kimi-k2-thinking-long-context-moe-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/kimi_k3/kimi_k3_inference_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/kimi_k2_thinking/kimi_k2_thinking_inference_guide.md
  source_digest: 807e9eaa7c67b358c6b671652d072e5b1ca599241e43f4cc75e06bbe08ddecdf
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/kimi_k3/kimi_k3_inference_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Inference optimization recipe for Kimi-K3 and Kimi-K2-Thinking covering MLA
  long-context KV compression, W8A8/W4A8 mixed quantization, large-scale EP MC2 dispatch,
  and long reasoning output scheduling.
tier: L2
title: Kimi-K3 and Kimi-K2-Thinking Long-Context MoE Inference Optimization Recipe
triggers:
  error_signatures:
  - kimi k3 inference optimization
  - kimi k2 thinking ascend inference
  keywords:
  - kimi k3 inference optimization
  - kimi k2 thinking ascend inference
  - kimi long reasoning chain decode
  - kimi moe ep32 mc2 w4a8
  - kimi-k3
  - kimi-k2-thinking
  - moe
  - mla
  - long-context
  - w4a8
  - w8a8
  - ep
  - mc2
---
## 1. Architectural Profile: Kimi-K3 & Kimi-K2-Thinking

`Kimi-K2-Thinking` (`1T` total parameters, `32B` active parameters per token, `384` routed experts, `top-8` routing, MLA attention with `64` heads and `128K`–`256K` context window) and `Kimi-K3` stress both HBM capacity (due to ultra-high total expert parameter count) and decode KV bandwidth (due to `8K`–`32K` token thinking chains per request).

## 2. Parallelism & Memory Budget Strategy

1. **Decoupled Attention `DP+TP` and MoE `EP`**:
   - **Attention (`DP + small TP`)**: Because MLA compresses KV cache into a `512`-dim latent vector + `64`-dim RoPE key, duplicating KV cache across `TP=8` wastes HBM needed for long thinking chains. Configure `attn_tp=1` or `2` (or `4` for low-latency single stream) with high `DP` to partition request batches without KV duplication.
   - **MoE (`EP16` / `EP32` / `EP64`)**: Distributes the `384` routed experts across all cluster ranks (`24` or `12` experts per rank) with `moe_tp=1`, replacing tensor-parallel `AllReduce` with `MC2` `npu_moe_distribute_dispatch_v2` / `combine_v2`.
2. **Mixed-Precision Quantization (`W8A8` Attention + `W4A8` / `W8A8` MoE)**:
   - **Routed Experts**: Quantized to `W4A8` per-channel/per-group or `W8A8` dynamic per-token so `1T` weights fit within dual-node Atlas A2 (`16 x 64GB`) or single-node Atlas A3 (`16 dies x 64GB`) alongside KV cache.
   - **MLA Projections & Router Gate**: Kept in `BF16` (router gate in `FP32` accumulation) or high-precision `W8A8` to preserve long-horizon reasoning accuracy across `16K+` thinking steps.
   - **KV Cache Quantization (`C8`)**: Quantizes latent KV cache to `INT8` with per-token dynamic scale, doubling maximum concurrent thinking sequences.

## 3. Long Reasoning Chain (`Thinking`) Decode Optimization

1. **Chunked Prefill & Continuous Decode Budget Balancing**:
   - Long reasoning workloads exhibit a low Prefill-to-Decode ratio (`1K` input vs. `16K` output). Enabling chunked prefill (`chunk_size=2048` or `4096`) prevents newly arriving prompts from stalling active decode iterations (`TPOT` spikes).
2. **Fused Operator Chain**:
   - `npu_add_rms_norm` + `npu_quantize` before MLA and MoE blocks.
   - `npu_moe_gating_top_k_softmax` (or renormalization gate) directly feeding `MC2` dispatch.
   - `npu_grouped_matmul` (`split_item=2` or `3`) + `npu_swiglu` + dynamic requantization (`aclnnGroupedMatmulSwigluQuantV2`).
3. **Multi-Stream Shared Expert Overlap & `aclgraph` Capture**:
   - Runs the dense shared expert branch on a secondary stream concurrently with `MC2` dispatch/combine of routed experts.
   - Captures decode batch buckets (`B = 1, 2, 4, 8, 16, 32, 64`) in `aclgraph` / `npugraph_ex` to eliminate host dispatch overhead across thousands of autoregressive reasoning tokens.

## Verification Checklist

- Confirm `experts_per_rank = 384 / ep_size` divides evenly and satisfies `MC2` rank limits on the target Atlas A2/A3 topology.
- Validate reasoning accuracy on math/code benchmarks (AIME, LiveCodeBench) after enabling `W4A8` / `C8` quantization to ensure no premature EOS or repetition loops.
