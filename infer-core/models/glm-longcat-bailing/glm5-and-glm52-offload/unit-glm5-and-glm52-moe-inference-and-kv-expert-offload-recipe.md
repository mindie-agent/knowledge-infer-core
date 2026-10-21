---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - GLM-5 and GLM-5.2 MoE inference under constrained HBM or ultra-long context
domain_path: infer-core/models/glm-longcat-bailing/glm5-and-glm52-offload
evidence_strength: reproduced_empirical
id: unit-glm5-and-glm52-moe-inference-and-kv-expert-offload-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/glm_5_2/glm_5_2_offload_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/glm_5/README.md
  source_digest: 99a365fddb6cbf55bed19a4d8f4d4c8fbd59245cd2c4645ad197eeb3725cd859
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/glm_5_2/glm_5_2_offload_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Inference and memory-offload playbook for GLM-5 and GLM-5.2 covering TP/EP
  hybrid execution, fused MoE/Attention operators, and asynchronous layer-pipelined
  weight/KV offloading between Host DRAM and NPU HBM.
tier: L2
title: GLM-5 and GLM-5.2 NPU Inference Optimization and Hierarchical Offload Recipe
triggers:
  error_signatures:
  - glm 5 ascend inference optimization
  - glm 5.2 offload guide
  keywords:
  - glm 5 ascend inference optimization
  - glm 5.2 offload guide
  - glm moe expert and kv cache offload
  - async h2d d2h stream overlap glm
  - glm-5
  - glm-5.2
  - offload
  - moe
  - kv-offload
  - expert-offload
  - aclgraph
---
## 1. GLM-5 Core NPU Inference Optimization (`models/glm_5/README.md`)

1. **Parallelism & Weight Layout**:
   - Partitions Attention with `TP` (merging `q_proj`, `k_proj`, `v_proj`) and MoE layers with `EP` + `GMM` (`npu_grouped_matmul` + `npu_swiglu`).
   - Uses `npu_add_rms_norm` for pre/post-norm residual fusion and partial RoPE (`npu_rotary_mul`) matching GLM rotary dimensions.
2. **Static KV Cache & Graph Mode (`npugraph_ex` / `aclgraph`)**:
   - Pre-allocates static KV cache blocks and captures the decode step into an `aclgraph` with `static_kernel` compilation to eliminate host launch gaps.

## 2. GLM-5.2 Hierarchical Offload Architecture (`glm_5_2_offload_guide.md`)

When deploying `GLM-5.2` on memory-constrained NPU clusters or serving ultra-long context batches where `Model Weights + KV Cache > HBM Capacity`, `glm_5_2_offload_guide.md` defines a dual-path offload engine:

### Path A: Layer-Pipelined Expert Weight Offload
1. **Sliding Window HBM Double Buffering**:
   - Keeps Attention weights, Shared Experts, and `K` layers of Routed Experts resident in HBM while remaining MoE layers reside in NUMA-bound pinned host memory (`pin_memory=True`).
   - Allocates two ping-pong HBM staging buffers (`StageBuf[0]`, `StageBuf[1]`).
2. **Compute-Copy Stream Overlap**:
   - While `ComputeStream` executes Layer $l$ (Attention + MoE using `StageBuf[l % 2]`), `PrefetchStream` asynchronously copies Layer $l+1$ offloaded weights from Host DRAM into `StageBuf[(l+1) % 2]` via `tensor.copy_(host_tensor, non_blocking=True)`.
   - Synchronizes via per-layer `torch.npu.Event` pairs (`prefetch_done_event[l+1]` and `compute_done_event[l]`) to prevent buffer overwrite before kernel completion.

### Path B: Long-Context KV Cache Offload & Prefetch
1. **Block-Granular KV Tiering**:
   - Splits Paged KV Cache into an **HBM Active Pool** (current chunk + recent sliding window + active decode pages) and a **Host DRAM Cold Pool**.
2. **Prefill & Multi-Turn Prefix Reuse**:
   - During long prefill or multi-turn prefix hits, historical KV blocks are prefetched layer-by-layer on `KVH2DStream` during layer $l-1$ computation and evicted asynchronously on `KVD2HStream` after attention completes.
3. **Bandwidth Sizing Rule**:
   - Ensure per-layer compute time $T_{	ext{compute}}^{(l)} \ge rac{	ext{OffloadedBytes}^{(l)}}{B_{	ext{PCIe}}}$ by tuning the offload ratio $lpha \in (0, 1]$ (partial layer offload) and quantizing offloaded weights/KV blocks to `INT8` / `FP8`.

## Verification Checklist

- Verify all host offload tensors are allocated with `pin_memory=True` and contiguous memory layout; unpinned host tensors trigger synchronous staging copies in `torch_npu` that stall `ComputeStream`.
- Confirm `compute_done_event[l].synchronize()` or stream `wait_event` precedes overwriting `StageBuf[l % 2]`.
