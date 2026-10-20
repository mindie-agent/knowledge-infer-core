---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-memory-and-kv/prefetch-and-offload
evidence_strength: software_contract
id: unit-infer-core-weight-prefetch-nz-exclusion-and-kv-offload-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/decode_weight_prefetch_ordering.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/weight_nz_mc2_mm_all_reduce_exclusion.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/weight_nz.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/weight_prefetch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/taylorseer_predictive_dit_cache.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/video_dit_taylorseer_cache_offload.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-prefetch/SKILL.md
  source_digest: 7d35bdbbc906c282478d711815c6098b6a751c98669b240dddba90c4a3aec6f3
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/decode_weight_prefetch_ordering.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 2B (Weight Prefetch, NZ Format, & Cache
  Offload): Bandwidth-ordered npu_prefetch, MC2 RowParallel NZ exclusion, and TaylorSeer
  offload'
triggers:
  error_signatures:
  - npu_prefetch evicts active L2 cache lines or provides zero speedup when anchored
    on a short pointwise dependency kernel
  - Enabling FRACTAL_NZ globally breaks or slows down RowParallelLinear layers using
    MC2 npu_mm_all_reduce_base
  - Video DiT TaylorSeer multi-order derivative feature cache OOMs NPU HBM on high-resolution
    video generation
  keywords:
  - decode_weight_prefetch_ordering
  - npu_prefetch
  - weight_nz_mc2_mm_all_reduce_exclusion
  - FRACTAL_NZ
  - npu_mm_all_reduce_base
  - taylorseer_predictive_dit_cache
---
# Inference optimization rules Part 2B (Weight Prefetch, NZ Format, & Cache Offload): Bandwidth-ordered npu_prefetch, MC2 RowParallel NZ exclusion, and TaylorSeer offload

## Applicability and symptom scope

Use this SOP when scheduling HBM-to-L2 weight prefetching (`torch_npu.npu_prefetch`), applying `FRACTAL_NZ` weight layout conversions, or managing high-footprint predictive feature caches (`TaylorSeer`) on Ascend NPU.

## Procedure

1. **Schedule `torch_npu.npu_prefetch` in descending memory-bound weight size order anchored on sufficiently long dependency kernels (`decode_weight_prefetch_ordering.md`, `weight_prefetch.md`, `model-infer-prefetch/SKILL.md`).**
   - `torch_npu.npu_prefetch(target_weight, dependency_tensor, max_size)` issues an asynchronous MTE2 preload of `target_weight` from HBM into the NPU L2 cache concurrently with the kernel that produces or consumes `dependency_tensor`.
   - **Ordering & Anchor Rules:**
     1. Prioritize the largest memory-bound decode projections (`q_b_proj` / `kv_b_proj` / `o_proj` / `shared_expert.gate_up_proj` / `w13`) capped at `max_size <= 16 MiB` (or a tuned fraction of physical L2 capacity so prefetched weights do not thrash active L2 working sets).
     2. Anchor each prefetch on a preceding compute/communication operator whose execution duration is long enough to hide the HBM-to-L2 transfer (e.g., prefetch `o_proj.weight` during `FIA` attention; prefetch `shared_expert` / `routed_expert` weights during `mla_prolog` or `dispatch`). Never anchor a multi-megabyte prefetch on a microsecond `RMSNorm` kernel.

2. **Exclude MC2 `npu_mm_all_reduce_base` RowParallel layers from `FRACTAL_NZ` weight conversion (`weight_nz_mc2_mm_all_reduce_exclusion.md`, `weight_nz.md`).**
   - Casting linear weights from `ACL_FORMAT_ND` to `ACL_FORMAT_FRACTAL_NZ` (`npu_format_cast(..., 29)`) eliminates on-the-fly MTE transpose overhead in standard `MatMul` and `GroupedMatmul` kernels.
   - **Per-Layer Exclusion Rule:** When `RowParallelLinear` (`o_proj`, `down_proj`) is configured to use the MC2 fused `npu_mm_all_reduce_base` operator on certain CANN/SoC revisions where the MC2 kernel requires `ND` weight layout (or incurs an internal `NZ -> ND` transdata penalty), keep those specific `RowParallelLinear` weights in `ACL_FORMAT_ND` while casting `ColumnParallelLinear` and MoE `GMM` weights to `FRACTAL_NZ`.

3. **Offload multi-order `TaylorSeer` DiT derivative caches to pinned Host DRAM with pipelined H2D prefetch (`taylorseer_predictive_dit_cache.md`, `video_dit_taylorseer_cache_offload.md`).**
   - `TaylorSeer` extrapolates DiT block features across skipped denoising steps using $m$-th order finite-difference Taylor series derivatives, requiring $(m+1)\times$ feature tensors per DiT block (`10..30 GiB` on long video sequences).
   - Offload the derivative state tensors to pinned Host DRAM (`pin_memory=True`) on a dedicated async copy stream: prefetch block $l+1$'s Taylor derivatives from Host to NPU while block $l$ executes on AICore, and monitor host memory bandwidth when running multi-process workers on the same host.
