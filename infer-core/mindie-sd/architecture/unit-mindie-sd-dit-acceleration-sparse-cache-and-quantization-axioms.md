---
schema: mindie-graded-unit/1
id: unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
title: MindIE-SD DiT/Diffusion acceleration architecture, sparse attention, CacheAgent, and quantization axioms
tier: L1
category: axiom
evidence_strength: software_contract
domain_path: infer-core/mindie-sd/architecture
inherits_from: []
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "aclnnBlockSparseAttention"
    - "inferShape function does not exist"
    - "W4A4_MXFP4_FOUROVERSIX_DYNAMIC"
  keywords:
    - MindIE-SD
    - mindiesd
    - RainFusion
    - rf_v2
    - rf_v3
    - ada_bsa
    - EagleQBSA
    - CacheAgent
    - MindieSDBackend
    - attention_forward
    - sparse_attention
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/architecture.md
  source_digest: 6d9df1e389bee0b17a8cb483685e3e920c0290bffd777733c999c0a38c8a1599
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/architecture.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/sparse.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/cache.md
  supersedes: []
revision: 1
---
## Core Invariants

1. **Three-Layer Diffusion Acceleration Architecture (`mindiesd`)**:
   - **Interface & Model Layer**: Provides unified Python entrypoints (`attention_forward`, `sparse_attention`, `rotary_position_embedding`, `rms_norm`, `fast_layernorm`, `layernorm_scale_shift`, `activate`, `quantize`, `MindieSDBackend`, `CacheAgent`, `enable_offload`, `usp_attention`, `eplb`) for both in-tree examples (`MiniMax-H3`, `Qwen-Image-2.1`, `Wan2.1/2.2`, `FLUX.1/2`, `HunYuanVideo`, `StepVideo`, `CogVideoX`) and third-party frameworks (`vLLM-Omni`, `LightX2V`, `DiffSynth-Engine`, `Cache-DiT`).
   - **Custom Operator Environment Registration (`ASCEND_CUSTOM_OPP_PATH`)**: Custom CANN operators (`RainFusionAttention`, `AdaBlockSparseAttention`, `EagleQuantBlockSparseAttention`, `LaPreprocess`, `SparseBlockEstimate`, `NormRopeConcat`, `AdaLayerNorm`) reside under `mindiesd/ops/vendors/*` and are registered at `import mindiesd` time via `mindiesd/env.py`. `import mindiesd` **must** execute before initializing the NPU context or allocating device tensors, otherwise `aclnnXxx ... inferShape function does not exist` is raised at runtime.

2. **Dense, Block-Sparse, and Quantized Attention Contracts**:
   - **Dense Attention (`attention_forward`)**: Supports `fused_attn_score` (`npu_fused_infer_attention_score`), `prompt_flash_attn` (`npu_prompt_flash_attention`), and `ascend_laser_attention` (`LaPreprocess` padding to 256/128 alignment + high-throughput Laser FA on Atlas A2).
   - **Sparse Attention (`sparse_attention`)**:
     - `rf_v2` (**RainFusion 2.0**): Predicts dynamic block-sparse masks via block-representative token pooling (`SparseBlockEstimate`) combined with 3D `[t, h, w]` spatio-temporal window token rearrangement (`text_first: bool`) and a first-frame attention sink (`first_frame_weight`).
     - `rf_v3` (**RainFusion 3.0**): Dispatches `aclnnBlockSparseAttention` with automatic V3 $\to$ V2 $\to$ V1 kernel fallback and four precision modes: `bf16`, `mix` (`EagleQBSA`: Q/K per-block INT8 + V per-channel FP8), `fp8` (`RotateAttention`: Hadamard transform + Q/K/V FP8), and `mxfp4` (`MXAttention`: Hadamard transform + Q/K FP4 E2M1 with E8M0 microscaling + V FP8 on Ascend 950).
     - `ada_bsa` (**Adaptive Block Sparse Attention**): Selects variable block sparsity per head via a cumulative distribution function (CDF) energy threshold (`cdf_threshold`).

3. **Step & Block Caching Contract (`CacheAgent`)**:
   - `CacheConfig` + `CacheAgent` accelerate denoising by reusing intermediate representations across adjacent diffusion timesteps within `[step_start, step_end)` at intervals of `step_interval` and across blocks `[block_start, block_end)`.
   - `dit_block_cache` (**DiTCache**) skips entire Transformer block evaluations (`Fn_compute_blocks` / `Bn_compute_blocks`), whereas `attention_cache` (**AttentionCache**) skips only Self-Attention / Cross-Attention / MLP submodules inside a block. The two cache granularities are **mutually exclusive** on the same block chain.

4. **Linear & Attention Quantization Contract (`quantize` & Online Schemes)**:
   - Supports descriptor-driven (`msmodelslim` `quant_desc_*.json`) and online quantization across `W8A16`, `W4A16`, `W8A8_STATIC`, `W8A8_DYNAMIC` (per-token dynamic INT8), `W8A8_MXFP8` (microscaling E4M3 with E8M0 shared exponent per 32-element block), `W8A8_TIMESTEP` (timestep-aware activation calibration), and `W4A4_MXFP4_FOUROVERSIX_DYNAMIC` (`quant_four_over_six_a5` adaptive 4/6 block scaling on Ascend 950).
   - Hardware capability and algorithm mapping for a named quantization tier vary by Ascend generation (Atlas A2 910B vs Atlas A3 910C vs Ascend 950 A5) and must be verified against actual device kernels rather than assumed from tier names alone.
