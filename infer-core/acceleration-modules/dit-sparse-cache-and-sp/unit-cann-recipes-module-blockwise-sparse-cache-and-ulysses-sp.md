---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - Plug-and-play DiT and multimodal acceleration modules in cann-recipes-infer/module
domain_path: infer-core/acceleration-modules/dit-sparse-cache-and-sp
evidence_strength: reproduced_empirical
id: unit-cann-recipes-module-blockwise-sparse-cache-and-ulysses-sp
inherits_from:
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/module/blockwise_sparse/sparse_method.py
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/mm_inference_design.md
  source_digest: a078e35ca3fe60347c8fc9f1072ac86026a318615b92edb4f5cc2bb6efe7e5b3
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:module/blockwise_sparse/sparse_method.py
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: 'Reference guide to cann-recipes-infer reusable acceleration modules (`module/blockwise_sparse`,
  `module/cache`, `module/parallel`): Ada-BSA blockwise sparse attention, TeaCache/FBCache/TaylorSeer,
  and CFG/Ulysses/VAE-Patch parallelism.'
tier: L2
title: 'Reusable Acceleration Modules: Blockwise Sparse Attention, TeaCache/TaylorSeer,
  and Ulysses SP'
triggers:
  error_signatures:
  - module blockwise sparse sparse_method
  - ada bsa blockwise sparse attention
  keywords:
  - module blockwise sparse sparse_method
  - ada bsa blockwise sparse attention
  - taylorseer dit feature forecasting
  - cann recipes infer module cache teacache
  - vae patch parallel module ascend
  - blockwise-sparse
  - ada-bsa
  - teacache
  - taylorseer
  - ulysses-sp
  - cfg-parallel
  - vae-patch-parallel
---
## 1. Adaptive Blockwise Sparse Attention (`module/blockwise_sparse/sparse_method.py`)

For long-sequence video/image DiTs (`S = 32K..128K` spatiotemporal tokens), full $O(S^2)$ attention dominates latency. `BlockwiseSparseAttention` (`Ada-BSA`) reduces attention compute via a 3-step block selection pipeline:
1. **Block Pooling & Representative Token Compression**:
   - Partitions Q and K sequences of length $S$ into coarse blocks of size $B_q, B_k$ (e.g., `block_size=64` or `128`) and computes mean/max pooled block representations $\bar{Q}, \bar{K} \in \mathbb{R}^{B \times H \times (S/B_b) \times D}$.
2. **Coarse Block Importance Scoring & CDF / Top-K Selection**:
   - Evaluates coarse block-to-block attention scores $S_{\text{coarse}} = \text{Softmax}(\bar{Q} \bar{K}^\top / \sqrt{D})$.
   - Selects either a fixed `top_k` blocks per query block or the minimal set of key blocks whose cumulative probability exceeds a dynamic threshold `cdf_threshold` (e.g., `0.90`–`0.95`), always retaining diagonal (local spatial/temporal) blocks.
3. **Sparse Block Gather & NPU FlashAttention Execution**:
   - Executes sparse block-level attention over only the selected key/value blocks on NPU, achieving `1.8x–3.2x` attention speedup on high-resolution video generation.

## 2. DiT Timestep Feature Caching (`module/cache`: `TeaCache`, `FBCache`, `TaylorSeer`)

1. **`TeaCache` & `FBCache`**:
   - Reuses block-stack residuals $\Delta x_{t-1}$ when modulated input or first-block output L1 drift is below threshold $\tau$.
2. **`TaylorSeer` (Taylor Series Feature Forecasting)**:
   - Instead of zero-order copy ($x_t \approx x_{t-1}$), `TaylorSeer` caches finite differences ($\Delta^1 x, \Delta^2 x$) across activated calibration timesteps and predicts intermediate timestep features via $m$-th order Taylor expansion:
     $$\hat{F}(t + k) = F(t) + \sum_{i=1}^m \frac{\Delta^i F(t)}{i!} \prod_{j=0}^{i-1} (k - j)$$
   - Permits higher skip intervals ($N_{\text{skip}} = 3..5$) without blurring fast-moving video subjects.

## 3. Composable Multimodal Parallelism (`module/parallel`)

- **`CFG Parallel`**: Splits positive and negative classifier-free guidance prompts across 2 parallel groups (`cfg_size=2`) and exchanges noise predictions via a single `AllGather` per denoising step.
- **`Ulysses SP` & `VAE Patch Parallel`**: Composes with `CFG Parallel` (`world_size = cfg_size * ulysses_size`) for DiT blocks and switches to spatial halo-exchange patch parallelism during 3D VAE decoding to prevent single-card OOM on `720p/1080p` videos.

## Verification Checklist

- Disable `BlockwiseSparseAttention` during the first `20%` of denoising steps (`warmup_steps`) where global spatial layout is formed, enabling sparsity only on later refinement steps.
