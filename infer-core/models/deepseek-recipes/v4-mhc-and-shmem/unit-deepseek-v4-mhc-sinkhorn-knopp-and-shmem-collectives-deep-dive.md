---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - DeepSeek-V4 mHC residual stream fusion and SHMEM symmetric memory collective acceleration
domain_path: infer-core/models/deepseek-recipes/v4-mhc-and-shmem
evidence_strength: reproduced_empirical
id: unit-deepseek-v4-mhc-sinkhorn-knopp-and-shmem-collectives-deep-dive
inherits_from:
- unit-infer-core-gated-deltanet-mamba2-mhc-and-hybrid-moe-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_mHC_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_shmem_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_autofuse_guide.md
  source_digest: fcd727092e4a88df2e7c5b7a53aba5690394d053bde6745dda0ec9f93e0f570c
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_mHC_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Deep-dive implementation guide for DeepSeek-V4 Manifold-Constrained Hyper-Connections
  (`deepseek_v4_mHC_guide.md`) and SHMEM symmetric memory collectives (`deepseek_v4_shmem_guide.md`).
tier: L2
title: DeepSeek-V4 mHC Sinkhorn-Knopp Fusion and SHMEM Device-Side Collectives Deep
  Dive
triggers:
  error_signatures:
  - deepseek v4 mhc guide
  - deepseek v4 shmem guide
  keywords:
  - deepseek v4 mhc guide
  - deepseek v4 shmem guide
  - mhc_pre_op mhc_post_op sinkhorn 20 iterations
  - shmem symmetric heap putmem barrier npu
  - deepseek-v4
  - mhc-guide
  - shmem-guide
  - birkhoff-polytope
  - sinkhorn-knopp
  - one-sided-dma
---
## 1. DeepSeek-V4 `mHC` Mathematical & Kernel Specification (`deepseek_v4_mHC_guide.md`)

In standard Hyper-Connections (`HC`), unconstrained learnable residual matrices $H_l^{\text{res}} \in \mathbb{R}^{n \times n}$ cause activation norms to drift exponentially across $L$ layers ($\prod_{l=1}^L H_l^{\text{res}}$). **Manifold-Constrained Hyper-Connections (`mHC`)** enforce doubly stochastic residual mixing ($H_l^{\text{res}} \mathbf{1} = \mathbf{1}$, $\mathbf{1}^\top H_l^{\text{res}} = \mathbf{1}^\top$, $H_{i,j} \ge 0$):
1. **Dynamic Coefficient Generation**:
   - Flattens the $n$-stream hidden state $x_l \in \mathbb{R}^{B \times S \times n \times C}$ ($n=4$) to $\tilde{x}_l \in \mathbb{R}^{B \times S \times (nC)}$ and computes $\hat{x}_l = \text{RMSNorm}(\tilde{x}_l)$.
   - Projects $\hat{x}_l$ via a small fused linear layer $\Phi_l \in \mathbb{R}^{(nC) \times (2n + n^2)}$ (`32` outputs for $n=4$) with learned gating scalars $\alpha_l^{\text{pre}}, \alpha_l^{\text{post}}, \alpha_l^{\text{res}}$ and bias $b_l$:
     $$H_l^{\text{pre}} = 2 \cdot \sigma(\tilde{H}_l^{\text{pre}}), \quad H_l^{\text{post}} = 2 \cdot \sigma(\tilde{H}_l^{\text{post}}), \quad M^{(0)} = \exp(\tilde{H}_l^{\text{res}})$$
2. **In-UB Sinkhorn-Knopp Projection ($t = 1 \dots 20$)**:
   - Iteratively normalizes rows and columns in `FP32` registers/UB without touching HBM:
     $$M^{(t)} = \mathcal{T}_{\text{col}}\big(\mathcal{T}_{\text{row}}(M^{(t-1)})\big), \quad H_l^{\text{res}} = M^{(20)}$$
3. **Stream Contraction & Expansion Fusion**:
   - `mhc_pre_op` outputs both the contracted layer input $x_{\text{in}} = \sum_{i=1}^n H_{l,i}^{\text{pre}} x_{l,i} \in \mathbb{R}^{B \times S \times C}$ and the doubly stochastic matrix $H_l^{\text{res}}$.
   - `mhc_post_op` computes $x_{l+1,i} = \sum_{j=1}^n H_{l,i,j}^{\text{res}} x_{l,j} + H_{l,i}^{\text{post}} y_l$ in a single UB pass.

## 2. DeepSeek-V4 `SHMEM` Symmetric Memory Collectives (`deepseek_v4_shmem_guide.md`)

1. **Symmetric Heap Initialization**:
   - Allocates cross-rank symmetric HBM buffers via `shmem_malloc` during model initialization, mapping peer NPU HBM over HCCS (intra-node Atlas A3) or RoCE (inter-node).
2. **Fused Compute + One-Sided Put (`MatMul / MoE Combine + SHMEM Put`)**:
   - Instead of waiting for an entire tensor to finish before launching HCCL `AllReduce` or `AlltoAll`, AICore tiles write completed output blocks directly into target rank symmetric memory addresses via MTE/RDMA put instructions and increment remote arrival counters (`shmem_int_atomic_inc`), eliminating collective launch latency in decode.

## Verification Checklist

- Always keep the `20`-iteration Sinkhorn-Knopp loop in `FP32` inside UB even when inputs/outputs are `BF16` to prevent row/column sum drift.
