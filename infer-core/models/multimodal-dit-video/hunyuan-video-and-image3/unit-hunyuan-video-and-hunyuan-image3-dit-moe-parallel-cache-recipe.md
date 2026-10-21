---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - HunyuanVideo text/image-to-video DiT and HunyuanImage-3.0 MoE image generation
domain_path: infer-core/models/multimodal-dit-video/hunyuan-video-and-image3
evidence_strength: reproduced_empirical
id: unit-hunyuan-video-and-hunyuan-image3-dit-moe-parallel-cache-recipe
inherits_from:
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/hunyuan-video/hunyuan_video_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/hunyuan-image-3.0/hunyuan_image_3_optimization.md
  source_digest: 33eadd260cc66e3a9bb4c0479bfc7e8956096a521d80bd1456a9dbaa9033e9a4
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/hunyuan-video/hunyuan_video_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Optimization playbook for HunyuanVideo and HunyuanImage-3.0 covering Ulysses/Ring/Anything
  sequence parallelism, FBCache & TeaCache timestep skipping, VAE conv3d tiling, and
  DoubleRouting MoE EP.
tier: L2
title: HunyuanVideo and HunyuanImage-3.0 DiT/MoE Parallelism, FBCache/TeaCache, and
  Operator Fusion
triggers:
  error_signatures:
  - hunyuan video optimization ascend
  - hunyuan image 3.0 optimization
  keywords:
  - hunyuan video optimization ascend
  - hunyuan image 3.0 optimization
  - fbcache teacache dit residual reuse
  - ulysses anything attention arbitrary heads
  - hunyuan image 3.0 moe ep double routing
  - hunyuan-video
  - hunyuan-image-3.0
  - dit
  - ulysses
  - ring-attention
  - fbcache
  - teacache
  - double-routing
---
## 1. HunyuanVideo DiT Optimization Pipeline (`hunyuan_video_optimization.md`)

1. **Sequence Parallelism (`Ulysses` + `Ring Attention` + `Ulysses Anything Attention`)**:
   - **Standard Ulysses**: Splits sequence length $S$ across $P$ cards (`S / P` per card) outside attention and switches to head partitioning (`H / P`) inside `npu_fused_infer_attention_score` via `AlltoAll`.
   - **Ulysses Anything Attention**: Overcomes the strict divisibility requirement ($H mod P = 0$) by dynamically padding/aligning head dimensions across `AlltoAll`, supporting arbitrary head counts on 8-card or 16-card NPU clusters.
   - **Hybrid Ulysses + Ring**: Combines intra-node Ulysses `AlltoAll` with inter-node Ring Attention P2P KV passing for ultra-long video token sequences (`S > 100K`).
2. **Timestep Residual Caching (`FBCache` vs. `TeaCache`)**:
   - Diffusion transformers execute $T_{	ext{steps}} = 30..50$ denoising steps where adjacent timesteps produce highly correlated hidden state residuals $\Delta x_t = x_t^{	ext{out}} - x_t^{	ext{in}}$.
   - **`FBCache` (First-Block Cache)**: Evaluates the relative L1 difference of the **first DiT Block output** between step $t$ and step $t-1$. When below threshold $	au$, skips all remaining DiT blocks for step $t$ and reuses the cached block-stack residual $\Delta x_{t-1}$.
   - **`TeaCache` (Timestep Embedding Aware Cache)**: Uses a polynomial-rescaled relative L1 distance of the **first `img_modulate` output** to decide whether to reuse the cached residual, saving 1.5x–2.2x DiT compute FLOPs with negligible visual degradation.
3. **Fused Operators (`FIA`, `npu_rms_norm`, `npu_rotary_mul`, `W8A8` Dynamic Quant)**:
   - Replaces unfused LayerNorm/RMSNorm, 3D RoPE, and FlashAttention with `torch_npu` fused operators.

## 2. HunyuanImage-3.0 MoE Parallelism (`hunyuan_image_3_optimization.md`)

1. **Attention `TP` + Merged `QKV` and MoE `w13` + `npu_swiglu`**:
   - Concatenates `q_proj`, `k_proj`, `v_proj` into `merged_qkv_proj` and MoE `gate_proj` + `up_proj` into `w13_weight` followed by `torch_npu.npu_swiglu`.
2. **MoE `EP` with Sequence-Sliced `DoubleRouting`**:
   - Under pure `MoE TP`, every rank executes full-sequence `gating`, `routing`, and `finalizing` over all tokens $S$, making routing overhead scale poorly as rank count grows.
   - **Optimization (`DoubleRouting` EP)**: Slices input sequence $S$ into $S / 	ext{ep\_size}$ per rank before gating so routing and finalizing compute drop by $1 / 	ext{ep\_size}$, and uses a two-stage permutation (`local top-k gating -> AlltoAll dispatch -> expert GMM -> AlltoAll combine -> local finalize`) to balance multi-rank MoE execution.

## Verification Checklist

- Keep warmup denoising steps (`first 3..5 steps` and `final 2 steps`) uncached when enabling `FBCache` or `TeaCache` to preserve structural composition and high-frequency detail.
