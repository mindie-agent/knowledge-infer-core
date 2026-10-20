---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-compute-fusion/attention-and-mla
evidence_strength: software_contract
id: unit-infer-core-tnd-layout-fia-mla-absorb-and-prolog-fusion-sop
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/tnd_packed_sequence_standard_and_bsnd_shim.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/attention_compressor_tnd_layout_migration.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/attention_to_fia_sparse_mode_sink_window.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/bidirectional_attention_fia_non_causal_full.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mla_weight_absorption_latent_kv.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mla_prolog_v3_prefill_seqlen_doc.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mlapo_fused_preprocess_w8a8_non_cp_gate.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dsa_sparse_prefill_mla_scheme_by_seqlen.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/sfaa_nonpa_qkv_layout_constraint.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-fusion/SKILL.md
  source_digest: f8d90c7a7b67c01e34de99d1fb82efc2a7862a72890a9af88f05d7debbc559fa
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/tnd_packed_sequence_standard_and_bsnd_shim.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 1A (Attention & MLA Fusion): TND packed
  layout, FIA sparse_mode mapping, MLA weight absorption, and MLAPO gating'
triggers:
  error_signatures:
  - BSND attention layout pads variable-length batch to max_seq_len and wastes AICore
    compute or crashes on ragged TND inputs
  - MLAPO fused decode preprocess silently falls back per layer when quantization
    is not W8A8 or SFA-CP is enabled
  - Non-PagedAttention SparseFlashAttention fails with host tiling GRAPH_FAILED when
    Q and KV layouts differ
  keywords:
  - tnd_packed_sequence_standard
  - attention_to_fia_sparse_mode
  - mla_weight_absorption
  - mlapo_fused_preprocess
  - dsa_sparse_prefill_mla_scheme
  - sfaa_nonpa_qkv_layout
  - actual_seq_qlen
---
# Inference optimization rules Part 1A (Attention & MLA Fusion): TND packed layout, FIA sparse_mode mapping, MLA weight absorption, and MLAPO gating

## Applicability and symptom scope

Use this SOP when migrating model attention layers to Ascend fused attention operators (`npu_fused_infer_attention_score` FIA, `npu_mla_prolog_v2/v3`, `MLAPO`, `SparseFlashAttention`) across Prefill and Decode stages.

## Procedure

1. **Standardize on `TND` (`[Total_tokens, Num_heads, Head_dim]`) packed-sequence layout (`tnd_packed_sequence_standard_and_bsnd_shim.md`, `attention_compressor_tnd_layout_migration.md`, `sfaa_nonpa_qkv_layout_constraint.md`).**
   - Unpadded `TND` layout concatenates variable-length sequences along axis `0` (`T = sum(seq_lens)`) without right-padding waste and is required by continuous batching, chunked prefill, and paged KV caches.
   - When migrating `attention` or `compressor` operators from `BSND` to `TND`, flatten `[B, S, N, D] -> [T, N, D]`, pass cumulative sequence lengths `actual_seq_lengths = torch.cumsum(seq_lens, dim=0)`, and ensure `dequant_scale` tensors share the merged token axis `T`.
   - On non-PagedAttention sparse/dequantized attention paths (`sfaa_nonpa_qkv_layout_constraint.md`), enforce identical `input_layout` (`TND` or `BNSD`) for both `Q` and `KV`; mixing `TND` queries with `BSND` keys triggers host-tiling `GRAPH_FAILED`.

2. **Map causal, sliding-window, attention-sink, and bidirectional masks to FIA `sparse_mode` (`attention_to_fia_sparse_mode_sink_window.md`, `bidirectional_attention_fia_non_causal_full.md`).**
   - Select `npu_fused_infer_attention_score` (FIA) for inference instead of training-oriented `npu_fusion_attention`.
   - Set `sparse_mode` and compressed boolean `atten_mask` (`[2048, 2048]` lower-triangular inversion on NPU where `True / 1` means **masked out**):
     - Standard causal prefill/decode: `sparse_mode = 2` (leftUpCausal) or `sparse_mode = 3` (rightDownCausal for chunked prefill) with preallocated `[2048, 2048]` `torch.bool` mask.
     - Sliding window (`pre_tokens`, `next_tokens = 0`): `sparse_mode = 4` (band mode).
     - Bidirectional block diffusion / DLLM decode (`bidirectional_attention_fia_non_causal_full.md`): `sparse_mode = 0` (`atten_mask = None`, `pre_tokens = 2147483647`, `next_tokens = 2147483647`); never route bidirectional blocks through causal FA fast paths.

3. **Switch MLA prefill scheme by sequence length and gate `MLAPO` decode fusion (`mla_weight_absorption_latent_kv.md`, `dsa_sparse_prefill_mla_scheme_by_seqlen.md`, `mla_prolog_v3_prefill_seqlen_doc.md`, `mlapo_fused_preprocess_w8a8_non_cp_gate.md`).**
   - `npu_mla_prolog_v3` supports full-length prefill sequences (`T > 16`) despite legacy header docstrings stating `0..16`.
   - **Short-Sequence DSA/MLA Prefill:** Use unabsorbed **MHA + Causal Mask** (`d_k = 192, d_v = 128`), where expanding $c_t^{KV}$ is cheaper than the $512$-dim absorbed attention FLOPs.
   - **Long-Sequence DSA/MLA Prefill & All Decode Steps:** Switch to **Weight-Absorbed + Sparse (`Absorb + SFA`)** so attention operates only on the top-$k$ selected latent blocks without materializing full-sequence `[T, H, 128]` K/V tensors in HBM.
   - **MLAPO Decode Gate:** The mega-fused `MLAPO` operator (fusing MLA prolog + dynamic quant + cache update) requires **both** `W8A8` quantization **and** `cp_size == 1` (non-SFA-CP); verify both gates at model init rather than hitting per-layer runtime fallback checks.
