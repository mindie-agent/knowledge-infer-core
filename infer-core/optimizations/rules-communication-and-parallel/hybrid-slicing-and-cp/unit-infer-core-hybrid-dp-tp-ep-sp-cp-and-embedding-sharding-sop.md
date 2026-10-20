---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-communication-and-parallel/hybrid-slicing-and-cp
evidence_strength: software_contract
id: unit-infer-core-hybrid-dp-tp-ep-sp-cp-and-embedding-sharding-sop
inherits_from:
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/prefill_mla_dp_vs_sp_tp_sp_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dp_attention_tp_oproj_mlp_hybrid.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/local_tp_memory_heavy_projections.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/long_seq_prefill_context_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/generative_recommender_vocab_parallel_embedding.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/ascend_deploy_parallel_variant_and_nondefault_platform.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/prefill_decode_cluster_scale_mismatch.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-impl/SKILL.md
  source_digest: 61a9d68953d234b21da5e6e9b66ff8ef0c4dfe38ba600358cce4ab6adbf68686
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/prefill_mla_dp_vs_sp_tp_sp_parallel.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 3C (Hybrid Parallel Slicing): MLA DP vs
  SP-TP-SP, selective local TP on O_proj/MLP/LM_Head, and VocabParallel Embedding'
triggers:
  error_signatures:
  - Pure DP Attention replicates heavy O_proj, dense MLP, and LM_Head weights on every
    card and OOMs or bottlenecks decode bandwidth
  - Low-concurrency long-sequence MLA prefill leaves DP ranks idle when SP-TP-SP or
    CP is not enabled
  - Prefill and Decode default YAML configs assume different cluster card counts and
    fail on smaller validation nodes
  keywords:
  - prefill_mla_dp_vs_sp_tp_sp
  - dp_attention_tp_oproj_mlp_hybrid
  - local_tp_memory_heavy_projections
  - long_seq_prefill_context_parallel
  - vocab_parallel_embedding
  - attn_dp_size
  - mlp_tp_size
---
# Inference optimization rules Part 3C (Hybrid Parallel Slicing): MLA DP vs SP-TP-SP, selective local TP on O_proj/MLP/LM_Head, and VocabParallel Embedding

## Applicability and symptom scope

Use this SOP when designing module-granular hybrid parallelism (`Attention-DP` + `Local-TP` + `MoE-EP` + `Prefill-CP/SP`) across Ascend NPU clusters.

## Procedure

1. **Select MLA Prefill Parallelism by request concurrency vs. sequence length (`prefill_mla_dp_vs_sp_tp_sp_parallel.md`, `long_seq_prefill_context_parallel.md`).**
   - **High request concurrency, moderate sequence length:** Use **Pure `Attention-DP`** (no intra-attention TP collectives; each DP rank processes independent requests while MoE layers join across all DP ranks via `EP`).
   - **Low request concurrency (`batch_size < dp_size`) or ultra-long prompts ($S \ge 32\text{K}$):** Pure DP leaves cards idle or OOMs on single-card activation memory. Switch prefill to **`SP-TP-SP`** (`ReduceScatter` / `AllGather` keeping sequence sharded across norms and latent projections) combined with **Context Parallelism (`ZigZag-CP` / `PCP` / `DCP`)** on attention, while MoE reuses the CP*TP group as its `EP` domain.

2. **Apply selective Local-TP (`mlp_tp_size` / `oproj_tp_size`) to memory-heavy projections under `Attention-DP` (`dp_attention_tp_oproj_mlp_hybrid.md`, `local_tp_memory_heavy_projections.md`, `generative_recommender_vocab_parallel_embedding.md`).**
   - In `Attention-DP + MoE-EP` deployments, replicating dense first-K layers, `o_proj`, `shared_expert`, `EmbedTokens`, and `LM_Head` on every DP rank wastes HBM capacity and decode HBM bandwidth.
   - Partition adjacent DP ranks into small **local TP subgroups** (e.g., `local_tp_size = 2` or `4` within the same HCCS node) specifically for `o_proj`, dense MLP layers, `shared_expert`, and `VocabParallelEmbedding` / `LM_Head`:
     - At the `DP -> Local-TP` boundary: `AllGather` token activations across the `local_tp` subgroup;
     - Execute TP-sharded weight matmul (loading only $1/\text{local\_tp\_size}$ of the weight per rank);
     - At the `Local-TP -> DP` boundary: `ReduceScatter` back to the per-DP-rank token slice.
   - For generative recommenders (`HSTU`) or massive-vocab LLMs, shard embedding tables above a size threshold via `VocabParallelEmbedding` with out-of-range mask zeroing and `AllReduce`.

3. **Reconcile Prefill/Decode cluster scale defaults (`prefill_decode_cluster_scale_mismatch.md`, `ascend_deploy_parallel_variant_and_nondefault_platform.md`).**
   - Always check `tp_size * dp_size * cp_size == world_size` and `moe_ep_size * moe_tp_size == world_size` when adapting separate Prefill (e.g., 16-die A3) and Decode (e.g., 32/64-die A3) default configurations to a smaller test cluster.
