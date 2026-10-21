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
  - Paged KV cache management, MTP speculative decoding, and variable-length TND packed
    execution
domain_path: infer-core/recipes-design/kvcache-mtp-and-packed-tnd
evidence_strength: software_contract
id: unit-kv-cache-manager-mtp-speculative-and-packed-sequence-tnd-design
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/kv_cache_design.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/mtp_design.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/common/mtp_model_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/packed_sequence_design.md
  source_digest: 00ee747822a09e180fbb94429d68470c084cf5dc4990e2645c1c2a587f8b3520
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/design/kv_cache_design.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Core design specification for KV Cache layouts (BSH/BNSD/PA_BSND/MLA compressed),
  Multi-Token Prediction (MTP) draft-verify state alignment, and Packed Sequence (TND)
  cumulative sequence length execution.
tier: L2
title: KV Cache Manager, Multi-Token Prediction (MTP), and Packed Sequence TND Execution
  Design
triggers:
  error_signatures:
  - cann recipes kv cache design
  - mtp multi token prediction design guide
  keywords:
  - cann recipes kv cache design
  - mtp multi token prediction design guide
  - packed sequence tnd layout design
  - actual seq lengths cumulative tnd
  - mla absorbed kv cache layout
  - kv-cache-design
  - mtp-design
  - packed-sequence
  - tnd-layout
  - paged-attention
  - speculative-decoding
---
## 1. KV Cache Architecture & Layout Contracts (`docs/design/kv_cache_design.md`)

1. **Contiguous vs. Paged KV Cache**:
   - **Static Contiguous (`BSH` / `BNSD`)**: Pre-allocates `[max_batch, max_seq_len, kv_heads * head_dim]` for offline benchmarking, updated in-place via `torch_npu.scatter_update_(kv_cache, slot_indices, new_kv, axis=-2)`.
   - **Paged Block Pool (`[num_blocks, block_size, kv_heads, head_dim]`)**: Eliminates external fragmentation for online serving; `BlockManager` maps logical sequence blocks to physical `block_tables` (`[B, max_blocks_per_seq]`) and `slot_mapping` (`[T_total]`).
2. **Heterogeneous & MLA Compressed Pools**:
   - **MLA Split/Combined Cache**: Stores `kv_c` (`latent dim=512`) and `k_pe` (`rope dim=64`) either in a unified `576`-dim block or separate aligned `512` + `64` tensors so `FIA` / `AMLA` reads contiguous 32-byte aligned tiles.
   - **Hybrid Attention Models** (`Gemma-4`, `Step-3.7-Flash`, `Qwen3-Next`): Allocates independent block pools per attention group (`Full Attention`, `Sliding Window Attention`, `Linear Attention Recurrent State`).

## 2. Multi-Token Prediction (`MTP`) Design & Adaptation (`mtp_design.md` & `mtp_model_guide.md`)

1. **MTP Execution Loop (`Draft -> Target Verify -> Rollback/Advance`)**:
   - Given target hidden state $h_t^{(L)}$ and token $y_t$, $K$ sequential MTP modules predict draft tokens $\hat{y}_{t+1}, \dots, \hat{y}_{t+K}$ using lightweight 1-layer transformer blocks (`eh_proj(concat(embed(y), norm(h)))` + shared/dedicated LM Head).
   - **Target Verification Step**: Packs $[y_t, \hat{y}_{t+1}, \dots, \hat{y}_{t+K}]$ (`q_len = K + 1` per request) into a single target model forward pass using `FIA` with causal tree/chain mask (`sparse_mode=2` or `3`).
2. **KV Cache & Position Alignment on Rejection**:
   - Speculative verification produces `1 <= n_accepted <= K + 1` accepted tokens.
   - For the next step, `positions` advances by `n_accepted`, and unaccepted draft KV slots at the tail of the sequence are overwritten on the next `scatter_update_` / `reshape_and_cache` without copying or reallocating KV blocks.

## 3. Packed Sequence (`TND`) Design (`docs/design/packed_sequence_design.md`)

1. **`TND` Flattening Invariant**:
   - Flattens variable-length sequences $S_1, S_2, \dots, S_B$ into $T_{	ext{total}} = \sum_{i=1}^B S_i$ tokens with tensor layout `[T_total, N_heads, Head_Dim]` (`TND`) for attention and `[T_total, Hidden_Size]` (`TH`) for linear/MoE layers.
2. **Cumulative Sequence Length Contract (`actual_seq_lengths` / `actual_seq_lengths_kv`)**:
   - In `TND` mode, `npu_fused_infer_attention_score` requires **cumulative** sequence lengths (`prefix sum`: `[S_1, S_1+S_2, ..., T_total]`), whereas `BSH`/`BSND` mode uses **per-request** lengths (`[S_1, S_2, ..., S_B]`).
   - Mixing up per-request vs. cumulative `actual_seq_lengths` in `TND` layout causes silent attention corruption or AICore out-of-bounds faults.

## Verification Checklist

- Confirm `actual_seq_lengths` and `actual_seq_lengths_kv` are passed as cumulative prefix sums (`torch.cumsum(seq_lens, dim=0)`) whenever `input_layout="TND"` is selected.
