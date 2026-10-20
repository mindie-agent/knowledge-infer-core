---
category: workflow
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/interventions-and-cases/e2e-case-studies
evidence_strength: reproduced_empirical
id: unit-infer-core-deepseek-glm-moe-afd-and-ascendc-tiling-cases-workflow
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/afd_attn_ffn_disaggregation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/ascendc_attention_op_tiling_budget.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/dsa_decode_memory_budget_indexer_cache.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/mla_prefill_flash_attention_ntd_layout.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/cases/moe_multi_stream_limit_core_num.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/models/qwen3_next_atlas_a3_deployment.md
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/model-infer-optimize/workflows/optimize-workflow.md
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/model-infer-optimize/workflows/sota-approach-workflow.md
  source_digest: aabf0334488819efdfb926a1cadeafb9fe678531a5d39244eb2b6481e81d7a4a
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/cases/afd_attn_ffn_disaggregation.md
  source_type: cannbot_case_study_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'End-to-end optimization case studies & SOTA workflow: AFD disaggregation,
  AscendC L1/L0/UB tiling budgets, DSA Indexer memory budgeting, and MLA NTD prefill'
triggers:
  error_signatures:
  - AscendC fused attention or indexer kernel overflows UB or L1 SRAM when double-buffering
    is omitted from tiling byte budget
  - DSA long-context decode OOMs because Lightning Indexer k_idx cache was excluded
    from KVCacheManager block sizing
  - MLA prefill FlashAttention incurs expensive NPU transdata copies when intermediate
    tensors mismatch NTD/TND layout
  keywords:
  - afd_attn_ffn_disaggregation
  - ascendc_attention_op_tiling_budget
  - dsa_decode_memory_budget_indexer_cache
  - mla_prefill_flash_attention_ntd_layout
  - qwen3_next_atlas_a3_deployment
  - sota-approach-workflow
---
# End-to-end optimization case studies & SOTA workflow: AFD disaggregation, AscendC L1/L0/UB tiling budgets, DSA Indexer memory budgeting, and MLA NTD prefill

## Objective

Apply end-to-end production case studies and the multi-agent `model-infer-optimize` SOTA convergence workflow across DeepSeek-V3/V3.2/R1, GLM-5, Qwen3-Next, and LongCat-Flash on Ascend NPU.

## Workflow Steps

1. **Execute the Closed-Loop SOTA Optimization Workflow (`optimize-workflow.md`, `sota-approach-workflow.md`).**
   - **Phase 0 (Scenario & Baseline Profiling):** Freeze workload `(batch_size, input_len, output_len, tp/dp/ep/cp)`, capture baseline accuracy + operator-level `torch_npu.profiler` traces, and decompose layer latency against theoretical roofline (`Cube FLOPs / peak_TFLOPS` vs. `HBM bytes / peak_GBps`).
   - **Phase 1 (Candidate Generation & Plan Isolation):** Generate ranked optimization plans across Parallelism -> KVCache -> Operator Fusion -> Multi-Stream -> Quantization -> Graph/SuperKernel.
   - **Phase 2 (Iterative Implement & Review Gate):** Apply one plan at a time, enforce logit/token accuracy parity and profiler timeline verification (confirming absence of fallback `aclnn` ops or `TransData` insertions), and archive verified gains into `plan-dashboard.md`.

2. **Budget DSA Decode Memory Including the Lightning Indexer Cache (`dsa_decode_memory_budget_indexer_cache.md`).**
   - In DSA models (`DeepSeek-V3.2-Exp`, `GLM-5`), each token writes both the MLA latent cache (`576` bytes in C8 or `1152` bytes in BF16 per layer) **and** the Lightning Indexer key cache $k^{\text{idx}}$ (`indexer_head_dim * num_indexer_heads` per layer).
   - Deduct both MLA page bytes and Indexer page bytes in `KVCacheManager.calculate_block_num()` so long-context decode batches never OOM mid-generation.

3. **Eliminate `TransData` in MLA Prefill via `NTD` / `TND` Layout Alignment (`mla_prefill_flash_attention_ntd_layout.md`) and Deploy `Qwen3-Next` on Atlas A3 (`qwen3_next_atlas_a3_deployment.md`).**
   - Align the output head/token axes of `q_b_proj` and `kv_b_proj` directly with the layout (`TND` / `NTD`) consumed by `npu_prompt_flash_attention` / `FIA` so GE/eager execution inserts zero hidden `Transpose` / `TransData` kernels.
   - For `Qwen3-Next` on Atlas A3 (`16` dies), combine Gated DeltaNet recurrent state management (`MambaManager`), `ZeCO` / Ulysses context parallelism for long prefill, dual-stream shared-expert overlap, and `TorchAir` decode graph capture.

4. **Size AscendC Attention/Indexer Tiling Strictly Within L1, L0A/B/C, and Double-Buffered UB Byte Ceilings (`ascendc_attention_op_tiling_budget.md`).**
   - When computing block tiling sizes $(B_m, B_n, B_k)$ in `op_host`:
     - Partition the outer $M$ (query token/head) axis evenly across active AIC/AIV cores first;
     - Account for **2x (`BUFFER_NUM = 2`) double-buffering** on every `TQue` in Unified Buffer (`UB` $\le 192\text{ KiB}$ on 910B/910_93), `L1` ($\le 512	ext{ KiB}$), and `L0A`/`L0B` ($\le 64	ext{ KiB}$ each) / `L0C` ($\le 128	ext{ KiB}$), plus `32-byte` UB alignment rounding on every intermediate vector register buffer.
