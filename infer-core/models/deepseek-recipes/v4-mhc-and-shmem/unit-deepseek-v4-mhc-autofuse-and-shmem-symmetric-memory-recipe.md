---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - DeepSeek-V4 multi-stream residual, AutoFuse graph compilation, and SHMEM low-latency
    collective inference
domain_path: infer-core/models/deepseek-recipes/v4-mhc-and-shmem
evidence_strength: reproduced_empirical
id: unit-deepseek-v4-mhc-autofuse-and-shmem-symmetric-memory-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-gated-deltanet-mamba2-mhc-and-hybrid-moe-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_inference_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_mHC_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_autofuse_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_shmem_guide.md
  source_digest: 04fb54a26c253d61d10fa694955eabfd30a3668f13804dd38f9d3653bc8517fb
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_inference_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: DeepSeek-V4 inference optimization guide covering Manifold-Constrained Hyper-Connections
  (mHC), Compressor + SparseAttnSharedKV attention pipeline, AutoFuse compiler pass,
  and SHMEM symmetric memory one-sided collectives.
tier: L2
title: 'DeepSeek-V4 Inference Guide: mHC Hyper-Connections, AutoFuse, and SHMEM Symmetric
  Memory'
triggers:
  error_signatures:
  - deepseek v4 inference guide
  - manifold constrained hyper connections mhc
  keywords:
  - deepseek v4 inference guide
  - manifold constrained hyper connections mhc
  - sinkhorn knopp doubly stochastic residual
  - deepseek v4 autofuse torch compile
  - deepseek v4 shmem symmetric memory communication
  - deepseek-v4
  - mhc
  - autofuse
  - shmem
  - sinkhorn-knopp
  - sparse-attn-sharedkv
  - compressor
---
## 1. DeepSeek-V4 Attention & Architectural Innovations

DeepSeek-V4 extends MLA and sparse attention with three tightly coupled architectural components:
1. **`Compressor` + `LightningIndexer` + `SparseAttnSharedKV` (`SAS`)**:
   - Compresses historical KV blocks via `Compressor` (strided pooling/projection with offset write-back), evaluates token importance via `LightningIndexer`, and executes unified attention over sliding-window, compressed, and top-k sparse KV blocks in a single `SparseAttnSharedKV` kernel.
   - Separates value-dependent tile scheduling into an `AICPU Scheduler` (`npu_sparse_attn_sharedkv_metadata`) so `AICore` kernels run with deterministic static-graph capture.
2. **Manifold-Constrained Hyper-Connections (`mHC`)**:
   - Replaces single-stream residual connections with an $n$-stream residual highway ($n=4$ expansion rate) governed by pre-mapping $H^{\text{pre}}$, post-mapping $H^{\text{post}}$, and residual mixing matrix $H^{\text{res}} \in \mathbb{R}^{n \times n}$:
     $$x_{l+1} = H_l^{\text{res}} x_l + (H_l^{\text{post}})^\top \mathcal{F}(H_l^{\text{pre}} x_l, W_l)$$
   - Projects $H_l^{\text{res}}$ onto the Birkhoff polytope (doubly stochastic matrices) via exponentiation followed by iterative Sinkhorn-Knopp row/column normalization ($t_{\max}=20$ iterations) so multi-layer products preserve unit spectral norm and prevent signal explosion.

## 2. `mHC` Operator Fusion Strategy (`Pre-Op` and `Post-Op`)

Unfused `mHC` launches dozens of small elementwise, reduction, and Sinkhorn-Knopp kernels per layer, inflating HBM traffic by $n\times$:
- **`mhc_pre_op` Fusion**: Fuses multi-stream linear projection, RMSNorm across $n \times C$, sigmoid scaling for $H^{\text{pre}}$ / $H^{\text{post}}$, exponential activation, 20-step Sinkhorn-Knopp normalization in UB SRAM, and stream contraction $x_{\text{in}} = H^{\text{pre}} x_l$ into a single kernel.
- **`mhc_post_op` Fusion**: Fuses layer output expansion $(H^{\text{post}})^\top y_l$ with multi-stream residual mixing $H^{\text{res}} x_l$ in UB without intermediate GM materialization.

## 3. AutoFuse Compiler Optimization (`deepseek_v4_autofuse_guide.md`)

- Integrates an automatic FX/Inductor pattern-matching pass (`AutoFuse`) that identifies fragmented elementwise + reduction + norm subgraphs around Attention/MoE boundaries and lowers them into fused Ascend kernels.
- Eliminates Python dispatch overhead and intermediate tensor allocations across dynamic sequence lengths while remaining compatible with `aclgraph` capture.

## 4. SHMEM Symmetric Memory Collectives (`deepseek_v4_shmem_guide.md`)

- **Bottleneck**: Small-batch decode collectives (`AllReduce`, `AllGather`, MoE small-token dispatch) are dominated by HCCL kernel launch and two-sided handshake overhead.
- **SHMEM Solution**: Maps peer NPU HBM windows into a symmetric virtual address space over HCCS/RoCE.
- **Direct Put/Get & Device-Side Barrier**: Compute kernels write output tiles directly into remote symmetric buffers using one-sided DMA (`shmem_putmem_nbi` / MTE store) and synchronize via lightweight atomic flags in symmetric memory, cutting small-message collective latency by 40%–65% during decode.

## Verification Checklist

- Verify Sinkhorn-Knopp iteration count ($t_{\max}=20$) and epsilon guard match reference `TileKernels` numerical behavior (`rtol=1e-3`, `atol=1e-3` in `BF16`).
- Ensure `SHMEM` symmetric heap size is pre-allocated before `aclgraph` capture so symmetric pointers remain invariant across graph replays.
