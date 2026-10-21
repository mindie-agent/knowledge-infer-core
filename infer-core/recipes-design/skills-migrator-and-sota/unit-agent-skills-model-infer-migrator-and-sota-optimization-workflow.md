---
category: workflow
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - End-to-end migration and SOTA optimization of PyTorch/HuggingFace inference models
    on Ascend NPU
domain_path: infer-core/recipes-design/skills-migrator-and-sota
evidence_strength: software_contract
id: unit-agent-skills-model-infer-migrator-and-sota-optimization-workflow
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-sota-approach/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-migrator/SKILL.md
  source_digest: cd15743a95dde10b1ec73f3b914744f1ac79d4c0460e7761e9487a0fc2485ac7
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:.agents/skills/model-infer-sota-approach/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Standardized 5-stage agentic workflow (`model-infer-migrator` + `model-infer-sota-approach`)
  for migrating HuggingFace models to Ascend NPU and systematically applying Parallelism,
  Static KV, Fusion, Quantization, and Graph Mode.
tier: L3
title: Agentic Model Inference Migrator and SOTA Optimization Closed-Loop Workflow
triggers:
  error_signatures:
  - model infer migrator skill
  - model infer sota approach workflow
  keywords:
  - model infer migrator skill
  - model infer sota approach workflow
  - five stage npu model optimization
  - gpu to ascend npu inference migration
  - model-infer-migrator
  - model-infer-sota-approach
  - agent-workflow
  - npu-migration
  - five-stage-optimization
---
## 1. Stage 0: NPU Migration & Golden Baseline (`model-infer-migrator`)

1. **Device & Backend Substitution**:
   - Import `torch_npu` and `from torch_npu.contrib import transfer_to_npu` (or explicit `cuda -> npu` / `nccl -> hccl` device mapping).
   - Disable dynamic JIT compilation (`torch_npu.npu.set_compile_mode(jit_compile=False)`) and internal format mutation (`torch.npu.config.allow_internal_format = False`) unless explicitly verified.
2. **CUDA Custom Op De-Coupling**:
   - Replace `flash_attn_func` / `xformers` / custom Triton CUDA kernels with pure PyTorch or `torch_npu` equivalents first to establish a working Eager NPU baseline.
3. **Golden Accuracy Gate**:
   - Record deterministic greedy decoding (`temperature=0`, `top_k=1`) output token IDs and top-5 logits on 3–5 standard prompts as the immutable Golden Reference before any performance transformation.

## 2. Five-Stage SOTA Optimization Closed Loop (`model-infer-sota-approach`)

Every stage must be applied **incrementally**, verified against the Golden Reference (`100%` greedy token match or `< 1%` dataset delta), and benchmarked (`Prefill TTFT` + `Decode TPOT` after warmup) before proceeding to the next stage:

1. **Stage 1 — Parallelism Partitioning (`model-infer-parallel-analysis`)**:
   - Size HBM capacity (`Weights + KV Cache + Workspace <= 0.85 * HBM`).
   - Partition Attention (`TP` with merged `QKV` + row-parallel `o_proj`, or `DP` for MLA), MoE (`EP` with `GMM` / `MC2` or `TP` with merged `gate_up`), and Embedding/LM Head (`VocabParallel`).
2. **Stage 2 — Static KV Cache & In-Place Update (`model-infer-kvcache`)**:
   - Eliminate dynamic `torch.cat([past_kv, new_kv], dim=-2)` allocation inside `forward()`.
   - Pre-allocate static contiguous or paged KV pools and update via `torch_npu.scatter_update_` or `npu_reshape_and_cache`.
3. **Stage 3 — Fused Operator Replacement (`model-infer-fusion`)**:
   - Replace decomposed norms, RoPE, attention, and MoE loops with `npu_add_rms_norm`, `npu_rotary_mul` / `npu_apply_rotary_pos_emb`, `npu_fused_infer_attention_score` (`FIA`), and `npu_moe_gating_top_k_softmax` + `npu_grouped_matmul` + `npu_swiglu`.
4. **Stage 4 — Graph Mode Capture (`model-infer-graph`) & Multi-Stream Overlap (`model-infer-stream-overlap`)**:
   - Eliminate host-device syncs (`item()`, dynamic indexing, Python conditionals on tensor values) in the decode path.
   - Enable `npugraph_ex` (`aclgraph` + `static_kernel`) or `torchair` (`ge_graph`) for Decode, and overlap shared-expert / dual-microbatch compute with collective communication.
5. **Stage 5 — Low-Bit Quantization (`model-infer-quantization`)**:
   - Apply `W8A8` dynamic per-token, `W4A8`, `MXFP8`, or `C8` KV cache quantization when memory bandwidth or capacity remains the bottleneck.

## Verification Checklist

- Never stack two unverified stages in a single edit; if greedy token match diverges or latency regresses, immediately bisect using the per-stage baseline table.
