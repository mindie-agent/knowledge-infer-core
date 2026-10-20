---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-communication-and-parallel/multistream-and-overlap
evidence_strength: software_contract
id: unit-infer-core-dual-stream-core-balance-vector-cube-and-microbatch-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/decode_multi_stream_overlap_core_balance.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/gmm_swigluquant_vector_core_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/hccl_aiv_expansion_vs_prefill_micro_batch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/hide_moe_attention_tp_ep_comm.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/multi_stream.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/prefill_micro_batch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/ring_attention_overlap.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/SKILL.md
  source_digest: d2ebeef1969801c2e1feb6bdd9a4beb71f79c9d6caba21a15efe558f8816b406
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/decode_multi_stream_overlap_core_balance.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 3A (Multi-Stream & Overlap): Cube-vs-Vector
  core complementarity, dual-stream core budgeting, and AIV collective vs micro-batch
  rules'
triggers:
  error_signatures:
  - Placing two Cube-heavy MatMul kernels on concurrent NPU streams yields zero wall-clock
    overlap without limit_core_num
  - Enabling HCCL AIV mode (HCCL_OP_EXPANSION_MODE=AIV) during prefill dual-stream
    micro-batching causes severe AIV vector core contention
  - Cross-stream tensor consumed without wait_tensor or record_stream produces intermittent
    numerical corruption
  keywords:
  - decode_multi_stream_overlap_core_balance
  - gmm_swigluquant_vector_core_parallel
  - hccl_aiv_expansion_vs_prefill_micro_batch
  - HCCL_OP_EXPANSION_MODE
  - limit_core_num
  - npu_stream_switch
  - wait_tensor
---
# Inference optimization rules Part 3A (Multi-Stream & Overlap): Cube-vs-Vector core complementarity, dual-stream core budgeting, and AIV collective vs micro-batch rules

## Applicability and symptom scope

Use this SOP when designing multi-stream compute-compute or compute-communication overlap in Eager, `ACLGraph`, or `TorchAir` GE Graph modes on Ascend NPU.

## Procedure

1. **Exploit AIC (Cube) vs. AIV (Vector) hardware unit complementarity before splitting cores (`gmm_swigluquant_vector_core_parallel.md`, `decode_multi_stream_overlap_core_balance.md`, `model-infer-multi-stream/SKILL.md`).**
   - On Ascend 910B/910_93, Cube (AIC) and Vector (AIV) pipelines are physically separate execution engines:
     - **Naturally Complementary Pair (No Core Split Needed):** A Cube-dominated operator (`MatMul` / `npu_grouped_matmul` GMM, which leaves AIV cores mostly idle) placed on Stream A naturally overlaps in hardware with a pure Vector operator (`npu_dequant_swiglu_quant`, `npu_rms_norm`, `npu_rotary_mul`, or `indexer_prolog` vector sub-chain) on Stream B.
     - **Same-Engine Pair (Explicit `limit_core_num` Mandatory):** When both Stream A and Stream B launch Cube-heavy or Vector-heavy operators concurrently (e.g., Shared-Expert MLP vs. Routed-Expert GMM), hardware waves serialize unless each stream is bounded via `torchair.scope.limit_core_num(aic_num, aiv_num)` so $\text{aic}_A + \text{aic}_B \le \text{total\_aic}$ and $\text{aiv}_A + \text{aiv}_B \le \text{total\_aiv}$.

2. **Toggle `HCCL_OP_EXPANSION_MODE` (AIV vs. AI_CPU/SDMA) based on Prefill Micro-Batch overlap (`hccl_aiv_expansion_vs_prefill_micro_batch.md`).**
   - On Atlas A3 single-stream execution, setting `HCCL_OP_EXPANSION_MODE=AIV` accelerates collectives by using AIV vector cores for collective reduction and data movement.
   - However, in **Prefill Micro-Batch Dual-Stream Overlap** (`prefill_micro_batch.md`, where Micro-batch 0's MoE `AllToAll`/`AllReduce` communication stream overlaps with Micro-batch 1's Attention/MLA compute stream), `AIV`-expanded collectives compete directly with Micro-batch 1's Vector kernels (FA softmax, RMSNorm, SwiGLU) for AIV cores and HBM bandwidth.
   - **Rule:** Disable AIV collective expansion (or cap AIV core allocation) when running dual-stream prefill micro-batch overlap so communication uses SDMA/RDMA engines while compute retains full AIV throughput.

3. **Hide TP/EP and Ring-Attention communication behind independent sub-chains (`hide_moe_attention_tp_ep_comm.md`, `ring_attention_overlap.md`).**
   - Launch `shared_expert` `AllReduce` asynchronously (`async_op=True`) while the router/EP combine finishes, or fuse it into the final combine reduction.
   - In Ring Attention CP, overlap `P2P Send/Recv` of KV shard $(i+1)$ on a dedicated communication stream while computing `FIA` on KV shard $i$ on the default compute stream.
