---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - Bailing-2.5 and Ling-3.0-Flash (Bailing-V3) MoE inference
domain_path: infer-core/models/glm-longcat-bailing/bailing-and-ling-flash
evidence_strength: reproduced_empirical
id: unit-bailing-25-and-ling-30-flash-pypto-megakernel-inference-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/integration/ling-3.0-flash/bailing_v3_pypto_operator_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/bailing_2_5/README.md
  source_digest: a5fc7f335e0b52c69a7c5dc9a08f0e1b0bd3c19a655b600eaadab1d0a461ed6a
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/integration/ling-3.0-flash/bailing_v3_pypto_operator_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Inference optimization recipe for Bailing-2.5 and Ling-3.0-Flash (Bailing-V3)
  combining TP/EP hybrid parallelism, static KV cache, and PyPTO MPMD tile-level mega-kernels
  for attention/norm/quant subgraphs.
tier: L2
title: Bailing-2.5 and Ling-3.0-Flash (Bailing-V3) NPU Inference and PyPTO Operator
  Recipe
triggers:
  error_signatures:
  - bailing 2.5 ascend inference
  - ling 3.0 flash bailing v3 pypto
  keywords:
  - bailing 2.5 ascend inference
  - ling 3.0 flash bailing v3 pypto
  - bailing v3 pypto operator guide
  - pypto mpmd attention moe fusion
  - bailing-2.5
  - ling-3.0-flash
  - bailing-v3
  - pypto
  - mpmd
  - megakernel
  - moe
---
## 1. Bailing-2.5 & Ling-3.0-Flash (`Bailing-V3`) Model Optimization Pipeline

1. **Architecture & Parallelism (`models/bailing_2_5/README.md`)**:
   - Sparse MoE architecture with shared + routed experts and MLA / GQA hybrid attention.
   - Deployed across 8 or 16 NPU ranks using `Attention TP/DP` and `MoE EP/TP` with merged `qkv_proj` and `gate_up_proj`.
2. **Standard Fused Operator Stack**:
   - `npu_add_rms_norm` for residual + RMSNorm fusion.
   - `npu_rotary_mul` for rotary embeddings and `npu_fused_infer_attention_score` (`FIA`) with static paged/contiguous KV cache.
   - `npu_moe_gating_top_k_softmax` + `npu_moe_init_routing_v2` + `npu_grouped_matmul` + `npu_swiglu` + `npu_moe_finalize_routing` for MoE layers.

## 2. Ling-3.0-Flash (`Bailing-V3`) PyPTO Mega-Kernel Optimization (`bailing_v3_pypto_operator_guide.md`)

In `Ling-3.0-Flash`, fine-grained pre-attention projections (Q/K/V LoRA down/up projections, per-head RMSNorm, RoPE split/concat, and KV cache scatter writes) create a memory-bound chain of small kernels during decode and chunked prefill.

### PyPTO MPMD (`Multiple Program Multiple Data`) Design
1. **Tile-Graph Producer-Consumer Scheduling**:
   - Expresses the entire pre-attention and attention subgraph as a single PyPTO Tensor/Tile graph.
   - Instead of SPMD global barriers between kernels, the PyPTO runtime compiles tile operations into an MPMD task dependency graph where AIC (Cube) cores produce projected Q/K/V tiles and signal AIV (Vector) cores via hardware event flags (`ffts` / cross-core sync) to immediately apply RMSNorm, RoPE, and KV cache updates in SRAM.
2. **SRAM Reuse & Zero Intermediate GM Writes**:
   - Keeps intermediate tiles (`q_nope`, `q_rope`, `k_nope`, `k_pe`, `kv_compressed`) resident in L1 / UB buffers across consecutive operations, eliminating redundant Global Memory (`GM`) round-trips.
3. **Cross-Platform PTO Instruction Lowering**:
   - Lowers the same PyPTO tile definition to both Atlas A2/A3 (`Ascend 910B/910C`) and `Ascend 950` hardware by adjusting only the tile shape config (`TileConfig`) without rewriting low-level AscendC synchronization barriers.

## Verification Checklist

- Confirm PyPTO compiled `.so` libraries are loaded before `aclgraph` capture so custom operator symbols bind into the static execution stream.
- Validate numerical consistency of the fused PyPTO pre-attention + attention subgraph against the eager `torch_npu` reference (`cosine similarity > 0.999`).
