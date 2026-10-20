---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/models-and-modules/mla-dsa-indexer
evidence_strength: software_contract
id: unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
inherits_from: []
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/mla.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/dsa.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/lightning_indexer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/sparse_flash_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/mla_prolog.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/lightning_indexer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/indexer_prolog.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/sparse_attn_sharedkv.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/compressor.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/deepseek_v3_2_exp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/deepseek_v4.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/glm_5.md
  source_digest: 824e310c0fe4543d190175300df450edc5067286897c3125f58a6d3c73e2671c
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/modules/mla.md
  source_type: cannbot_concept_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Architectural invariants of MLA, DeepSeek Sparse Attention (DSA), Lightning
  Indexer, and Compressor KV kernels on Ascend NPU
triggers:
  error_signatures:
  - MLA decode without weight absorption expands latent kv_lora_rank=512 into full
    per-head K/V and exhausts HBM bandwidth
  - DSA sparse attention top-k gather reads out-of-bounds KV slots when Lightning
    Indexer block stride mismatches SFA block_table
  - DeepSeek-V4 SparseAttnSharedKV or Compressor C4A/C128A fails on mismatched latent
    nope/rope dimension split
  keywords:
  - Multi-head Latent Attention
  - DeepSeek Sparse Attention
  - Lightning Indexer
  - npu_mla_prolog
  - indexer_prolog
  - SparseAttnSharedKV
  - GatherSelectionKvCache
  - Compressor
  - q_lora_rank
  - kv_lora_rank
  - qk_rope_head_dim
---
# Architectural invariants of MLA, DeepSeek Sparse Attention (DSA), Lightning Indexer, and Compressor KV kernels on Ascend NPU

## Condition

Applies to inference and kernel orchestration for Multi-head Latent Attention (MLA) and DeepSeek Sparse Attention (DSA) model families (`DeepSeek-R1`, `DeepSeek-V3.2-Exp`, `DeepSeek-V4`, `GLM-5`, `Kimi-K2-Thinking`) on Ascend NPU.

## Constraint

1. **MLA Latent Decomposition & Weight Absorption Dual-Form (`mla.md`, `mla_prolog.md`, `fused_infer_attention_score.md`):**
   - MLA splits per-token key/value state into a compressed latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ (`kv_lora_rank = 512`) and a decoupled rotary key $k_t^R \in \mathbb{R}^{d_r}$ (`qk_rope_head_dim = 64`), reducing KV cache footprint to `576` elements per token across all heads.
   - **Prefill (Unabsorbed MHA Form):** Projects $c_t^{KV}$ through $W^{UK}$ and $W^{UV}$ into explicit per-head $K_{\text{nope}} \in \mathbb{R}^{H \times 128}$ and $V \in \mathbb{R}^{H \times 128}$ (compute-bound Cube GEMMs, $d_k = 192, d_v = 128$).
   - **Decode (Weight-Absorbed MQA Form):** Absorbs $W^{UK}$ into $W^{UQ}$ ($q_{\text{absorb}} = q_{\text{nope}} W^{UK} \in \mathbb{R}^{H \times 512}$) so attention scores are computed directly against the paged latent cache $c_t^{KV}$ (`num_kv_heads = 1`, $d_k = 512 + 64, d_v = 512$), followed by multiplying the attention output by $W^{UV}$.
   - `npu_mla_prolog` (`v2`/`v3`) fuses RMSNorm, $q_a / q_b / kv_a$ projections, RoPE, and paged latent cache scatter into a single prologue kernel.

2. **DSA Two-Stage Sparse Selection: `indexer_prolog` -> `lightning_indexer` -> `sparse_flash_attention` (`dsa.md`, `lightning_indexer.md`, `indexer_prolog.md`, `sparse_flash_attention.md`):**
   - In `DeepSeek-V3.2-Exp` and `GLM-5`, full-context attention complexity $O(L^2)$ is reduced to $O(L \cdot k_{\text{sel}})$ via a lightweight **Lightning Indexer (LI)**:
     1. `indexer_prolog` projects hidden states into low-dimensional indexer queries $q^{\text{idx}}$ and keys $k^{\text{idx}}$, applies RoPE (and optional Hadamard rotation for INT8 quantization), and scatters $k^{\text{idx}}$ into a dedicated indexer cache.
     2. `lightning_indexer` (`npu_lightning_indexer` / `npu_quant_lightning_indexer` / `npu_lightning_indexer_pto`) computes weighted ReLU/dot-product relevance scores across candidate KV blocks and selects the top-$k$ block/token indices (`topk_indices`).
     3. `sparse_flash_attention` (`SFA`) or `GatherSelectionKvCache` + `FIA` attends strictly over the selected `topk_indices` from the MLA latent cache.

3. **DeepSeek-V4 `Compressor` (`C4A`/`C128A`) and `SparseAttnSharedKV` (`SAS`) Invariants (`deepseek_v4.md`, `compressor.md`, `sparse_attn_sharedkv.md`):**
   - `Compressor` (`C4A` / `C128A`) hierarchical KV compression aggregates fine-grained token KV states into coarse summary blocks before indexing, while `SparseAttnSharedKV` (`SAS`) shares compressed KV representations across adjacent layer groups to bound HBM capacity at ultra-long context lengths.
