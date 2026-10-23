---
schema: mindie-graded-unit/1
id: unit-mindie-sd-dit-compute-sparse-cache-and-quant-selection-sop
title: MindIE-SD DiT compute bottleneck optimization, sparse/quant/cache tier selection, and combination search SOP
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: infer-core/mindie-sd/dit-compute
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "staying dense"
    - "Can't find Parallelism/Quantization Config for"
    - "|Δwall| >= 2 * |Δdev|"
  keywords:
    - dit-perf-opt
    - rf_v2
    - rf_v3
    - ada_bsa
    - EagleQBSA
    - DiTCache
    - AttentionCache
    - W8A8_MXFP8
    - W8A8_DYNAMIC
    - combination-search
    - lora-adapter-cost
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-perf-opt/SKILL.md
  source_digest: 76dd44bd584d0e0b20c537f7df547ed5f3d8bc17c5ee5a7919e9726f3497a2e6
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-perf-opt/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/sparse.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/cache.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Map Operator Breakdown to DiT Optimization Dimensions**:
   - From the 5-layer profiling diagnosis (`FA` / `MatMul` / `Vector` / `Comm` shares inside `<Pipeline>.diffuse`), prioritize lossless optimizations before lossy ones:
     `MindIE-SD Pattern/Operator Fusion > Linear/FA Quantization > Block-Sparse Attention > Step/Block Cache`.
   - **Intra-Step Host Dispatch vs Device Compute Attribution**: When `|Δwall| >= 2 * |Δdev|` inside a DiT step (e.g., runtime rank-$r$ LoRA adapter chains launching dozens of tiny `down`/`up` GEMMs per block), the bottleneck is **host dispatch / op count** rather than AI Core compute. For static serving-time LoRA adapters with constant request scale, merge the low-rank delta into base weights offline before quantization (`lora-merge-and-precision`) to eliminate the entire runtime adapter chain.

2. **Select and Verify Block-Sparse Attention (`rf_v2` / `rf_v3` / `ada_bsa`)**:
   - **3D Video vs 2D Image Eligibility**: `rf_v2` (`RainFusion2.0`) requires 3D video token geometry (`latent_shape = [t, h, w]`, `qkv_layout = "BSND"`). On 2D image pipelines, `rf_v2` falls back to `staying dense` and produces output byte-identical to lossless dense FA—classify this fail-closed as **not engaged**.
   - **Distinguish Eager External-Mask vs Fused Op Paths**: Eager external-mask sparse paths (`rf_v3` public Python wrapper) incur per-layer full-tensor rearrange/pooling data movement that scales linearly with layer count and can make single-op latency slower than dense FA at moderate sparsity. Always benchmark and prefer fused in-op mask+BSA kernels (`rf_v2` `RainFusionAttention` or `EagleQBSA` `precision="mix"`).
   - **Warmup & Tail Dense Steps**: Set `start_step` (initial dense warmup steps) and `end_step` (number of **final steps kept dense**; never set `end_step == num_inference_steps`). Sweep sparsity vs quality non-linearly rather than extrapolating linearly from density.

3. **Select and Verify Linear & Attention Quantization**:
   - **Linear Tiers**: Choose `W8A8_MXFP8` (microscaling E4M3 + E8M0, near-lossless on SwiGLU/MoE DiTs and eligible for Catlass `mm_swiglu_mxquant` fusion) or `W8A8_DYNAMIC` (per-token dynamic INT8 via `DynamicQuantV2 -> QuantBatchMatmulV3`). On Ascend 950 (A5), evaluate `W4A4_MXFP4_FOUROVERSIX_DYNAMIC` (`quant_four_over_six_a5`).
   - **FA Quantization**: Pair sparse attention with `EagleQBSA` (`precision="mix"`, Q/K INT8 + V per-channel FP8) or dense FA with `FP8RotateQuantFA` (`RotateAttention`) / `MXAttention` (`mxfp4`) where supported by the target SoC.

4. **Select and Verify `CacheAgent` / `Cache-DiT` Step & Block Reuse**:
   - Choose **one** cache granularity (`dit_block_cache` or `attention_cache`; never enable both simultaneously).
   - Because caching skips full block/submodule forward passes across steps, single-step kernel composition does not change; validate engagement via **total kernel invocation count drop** (or step reuse counters) plus end-to-end wall clock and quality gates.
   - On few-step distilled models (e.g., 4-step FlashGen), verify whether `max_warmup_steps` consumes all steps; if cache ON and OFF produce identical output `md5`, record cache as **structurally inactive** on the few-step tier.

5. **Execute Multi-Feature Combination Search (`[MUST]` Coverage & Synergy Ratio)**:
   - **Scale-Dependent Ranking Reversal**: Single-feature speedup rankings reverse with sequence length $S$. On short sequences, Cache > Quantization > Sparse; on long video sequences ($O(S^2)$ attention dominance), **Sparse Attention > Cache > Linear Quantization**. Furthermore, the marginal speedup of Linear Quantization **increases monotonically** once Sparse Attention and Cache shrink the attention denominator (`Synergy Ratio = Measured_Combo / Product_of_Single_Speedups > 1.0`).
   - **Mandatory Combination Matrix**: Test all pairwise combinations plus the mandatory triad `Cache + Quant(w8a8/f8) + Sparse`. When quality falls below the `accuracy-gate` threshold on sensitive early/late layers, apply **layer rollback** (keep sensitive blocks in BF16/dense) and re-measure full-step E2EL.
