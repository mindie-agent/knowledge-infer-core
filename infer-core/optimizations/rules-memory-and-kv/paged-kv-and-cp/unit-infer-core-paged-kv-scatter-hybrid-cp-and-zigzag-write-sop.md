---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-memory-and-kv/paged-kv-and-cp
evidence_strength: software_contract
id: unit-infer-core-paged-kv-scatter-hybrid-cp-and-zigzag-write-sop
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/hybrid_kv_cache_context_parallel_block_size_alignment.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/ascend_paged_attention_kv_cache.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/paged_kv_cache_write_scatter_update_vs_scatter_nd.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/graph_mode_kvcache_inplace_mla_prolog_adaptation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mla_prolog_zigzag_cp_kv_write.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/causal_cp_prefill_zigzag_actual_seqlens.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-kvcache/SKILL.md
  source_digest: 07904b865eb3f639ee0ed60bff07df34f9cced9eb00db0004d7c5940670499c2
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/hybrid_kv_cache_context_parallel_block_size_alignment.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 2A (Paged KV & CP Memory): Stage-specific
  KV scatter ops, hybrid block-size LCM alignment, and ZigZag-CP two-segment KV write'
triggers:
  error_signatures:
  - Hybrid KV cache model with multiple block_size groups fails assertion when Context
    Parallelism (DCP/PCP) is enabled
  - MLA-prolog under ZigZag-CP prefill corrupts paged KV cache when writing both non-contiguous
    zigzag segments via a single slot_mapping
  - Decode single-token KV update incurs high launch/indexing overhead when using
    contiguous-range scatter_update_ instead of scatter_nd_update_
  keywords:
  - hybrid_kv_cache_context_parallel
  - paged_kv_cache_write_scatter
  - npu_scatter_nd_update_
  - scatter_update_
  - mla_prolog_zigzag_cp_kv_write
  - causal_cp_prefill_zigzag
---
# Inference optimization rules Part 2A (Paged KV & CP Memory): Stage-specific KV scatter ops, hybrid block-size LCM alignment, and ZigZag-CP two-segment KV write

## Applicability and symptom scope

Use this SOP when implementing or optimizing Paged Attention KV cache writes, hybrid multi-spec KV pools under Context Parallelism (`PCP`/`DCP`), and ZigZag-CP prefill on Ascend NPU.

## Procedure

1. **Select stage-optimal KV scatter operators and guard in-place graph updates (`ascend_paged_attention_kv_cache.md`, `paged_kv_cache_write_scatter_update_vs_scatter_nd.md`, `graph_mode_kvcache_inplace_mla_prolog_adaptation.md`).**
   - **Prefill (contiguous slot ranges):** Use `torch_npu.scatter_update_` (or fused `reshape_and_cache` / `npu_mla_prolog` contiguous block write) to burst-write contiguous token segments into paged blocks.
   - **Decode (single-token scattered slots across batch $B$):** Use `torch_npu.npu_scatter_nd_update_(kv_cache_view, slot_mapping.view(-1, 1), new_kv)` (or in-place `npu_mla_prolog` with `slot_mapping`), which is optimized for 1-token-per-request random page updates.
   - **Graph-Mode In-Place Guard:** In `torchair` GE graph mode, any custom operator that mutates `kv_cache` in place without returning a new tensor handle can be dead-code-eliminated or reordered before downstream attention; anchor the data dependency via `kv_cache = torch.add(kv_cache, 0)` (or `auto_functionalized` mutation registration).

2. **Align Hybrid KV Cache block sizes with Context Parallelism via LCM arithmetic (`hybrid_kv_cache_context_parallel_block_size_alignment.md`).**
   - Upstream vLLM blocks hybrid KV cache models (e.g., FullAttention + SlidingWindow or Mamba with distinct `block_size` groups $B_1, B_2$) from enabling Context Parallelism (`dcp_size > 1` or `pcp_size > 1`) due to a CUDA-specific assumption.
   - On Ascend NPU, lift the blanket prohibition and enforce the exact divisibility invariant on prompt chunking and page allocation granularity:
     $$\text{cp\_align\_granularity} = \operatorname{lcm}(B_1, B_2, \dots) \times \text{dcp\_size} \times \text{pcp\_size}$$
     ensuring every CP shard boundary lands on an integer block boundary for all KV cache groups simultaneously.

3. **Execute two-segment KV writes and asymmetric `actual_seq_lengths` in ZigZag-CP Prefill (`mla_prolog_zigzag_cp_kv_write.md`, `causal_cp_prefill_zigzag_actual_seqlens.md`).**
   - In ZigZag-CP prefill, each rank holds two non-adjacent sequence segments: a head segment $S_{\text{head}}$ (length $L_{\text{seg}}$, causal prefix length $L_{\text{prefix}, 0}$) and a tail segment $S_{\text{tail}}$ (length $L_{\text{seg}}$, causal prefix length $L_{\text{prefix}, 1} > L_{\text{prefix}, 0}$).
   - **KV Write Rule:** Do not pass a single combined `slot_mapping` to `mla_prolog` when the operator assumes monotonic contiguous slots; split the KV cache write into two explicit segment updates (`head` and `tail`) or scatter via explicit per-token `slot_mapping` indices after `AllGather` across the CP group.
   - **Causal Attention Rule:** Invoke causal attention separately for the head chunk (`q_len = L_seg`, `kv_len = L_prefix,0`) and the tail chunk (`q_len = L_seg`, `kv_len = L_prefix,1`) so neither chunk attends to future tokens or wastes FLOPs on masked suffix blocks.
