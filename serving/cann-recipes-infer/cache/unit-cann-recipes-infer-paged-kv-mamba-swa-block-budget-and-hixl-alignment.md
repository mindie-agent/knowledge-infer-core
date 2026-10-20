---
category: sop
conditions:
  repository: cann/cann-recipes-infer
  revision: 3c9266727c2a25bd17d503ad37d189a134ff364b
domain_path: serving/cann-recipes-infer/cache
evidence_strength: single_observation
id: unit-cann-recipes-infer-paged-kv-mamba-swa-block-budget-and-hixl-alignment
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/cann-recipes-infer-reviewed-source.json
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/kv_cache_manager.py#L45-L225
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/single_type_kv_cache_manager.py#L29-L410
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/cache_utils.py#L276-L706
  source_digest: e22a56b2d9fef942da72244840eb3a4b85d40663270485d013a9f79f10868abc
  source_locator: docs/evidence/cann-recipes-infer-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Hybrid FullAttention, SlidingWindow, and Mamba block budgeting with 2 MiB HIXL-aligned HBM allocation
triggers:
  error_signatures:
  - Not enough NPU memory to allocate the minimum required blocks
  - Requested to allocate more blocks than available
  - Block size is less than Mamba state token requirement
  keywords:
  - cann-recipes-infer
  - kv_cache_manager
  - FullAttentionManager
  - SlidingWindowManager
  - MambaManager
  - calculate_fixed_block_memory_bytes
  - _alloc_aligned_hbm_tensor
  - HIXL
---
At the pinned `cann/cann-recipes-infer` revision, `KVCacheManager` (`executor/core/kv_cache/kv_cache_manager.py:45-225`) coordinates heterogeneous per-layer KV and recurrent state managers through a shared `BlockPool` (`FreeKVCacheBlockQueue` with sentinel-linked doubly linked nodes, `block_id=0` reserved as `null_block`), while splitting HBM budgeting between fixed-block state (`MambaSpec`) and dynamic paged KV blocks (`FullAttentionSpec` and `SlidingWindowSpec`).

## 1. Per-Spec Manager Dispatch and Shared Block Pool Invariants

1. **Three-Manager Architecture (`single_type_kv_cache_manager.py:29-410`)**:
   - `FullAttentionManager` (`lines 198-219`): Standard Paged Attention block allocation where every sequence position remains addressable across the full context length; skips already allocated blocks on subsequent chunked steps (`new_blocks = req_blocks[len(allocated_blocks):]`).
   - `SlidingWindowManager` (`lines 222-331`): Nullifies blocks that fall outside `sliding_window` tokens before allocating new blocks (`req_blocks[i] = self.null_block` for indices `< start_block_idx`), returning evicted blocks to `BlockPool.free_blocks()` immediately before requesting new blocks (`free_then_allocate_blocks`) and appending a speculative draft block (`num_draft_tokens = self.num_speculative_tokens`).
   - `MambaManager` (`lines 334-410`): Reserves fixed-count recurrent state blocks per request (`num_state_blocks = 1` normally, or `2` when `num_speculative_tokens > 0` to hold draft verification state). Raises `RuntimeError` if `block_size < num_state_tokens`.
2. **Rollback on Partial Allocation Failure (`kv_cache_manager.py:130-169`)**:
   - `allocate_slots()` iterates across `self.managers` (`full_attention`, `sliding_window`, `mamba`). If any manager returns `None`, `KVCacheManager` rolls back all prior managers that succeeded in the current step (restoring `req_to_blocks[request_id]` to its pre-step prefix or calling `manager.free(request_id)` if previously empty) so no orphan blocks leak across hybrid layers.

## 2. Two-Stage HBM Block Budgeting and TP-Minimal Synchronization

1. **Stage 1 — Deduct Fixed Mamba State Memory (`cache_utils.py:276-306`)**:
   - `calculate_fixed_block_memory_bytes(kv_cache_spec, max_batch_size, num_speculative_tokens)` sums `mamba_page_size_bytes * mamba_block_num` across all `MambaSpec` groups, where `mamba_block_num = max_batch_size * (2 if num_speculative_tokens > 0 else 1) + 1` (adding `+1` for `null_block`).
   - Why: Mamba recurrent state scales with `max_batch_size` rather than token sequence length, so deducting it before dividing remaining HBM prevents Mamba state buffers from growing with sequence capacity.
2. **Stage 2 — Dynamic Paged Block Count and `ReduceOp.MIN` Sync (`cache_utils.py:309-396`)**:
   - `calculate_block_num()` queries `torch.npu.mem_get_info()`, computes `usable_memory = int(total_memory * memory_fraction) - allocated_memory - fixed_block_memory_bytes`, and divides by the summed `page_size_bytes` of all non-Mamba specs (`FullAttentionSpec`, `SlidingWindowSpec`).
   - Synchronizes `num_blocks` across `comm_group.attn_tp_group` via `dist.all_reduce(num_blocks_tensor, op=dist.ReduceOp.MIN, group=comm_group.attn_tp_group)` so all TP ranks allocate an identical block table size even when per-rank HBM fragmentation differs.
   - Enforces a minimum block floor: `min_blocks_needed = cdiv(max_seq_len, block_size) * cdiv(max_batch_size, attn_dp_size) + 1` for FullAttention and `(cdiv(sliding_window - 1, block_size) + 1 + cdiv(num_speculative_tokens, block_size)) * cdiv(max_batch_size, attn_dp_size) + 1` for SlidingWindow.

## 3. 2 MiB HIXL Address Alignment for Disaggregated KV Transfer (`cache_utils.py:442-468`)

- When disaggregated PD KV transfer (`enable_kv_transfer=True`) is active on Ascend NPU, `_alloc_aligned_hbm_tensor(shape, dtype, device, align=2 * 1024 * 1024)` allocates a flat `uint8` buffer padded by `align` bytes, computes `offset = (-base_ptr) % align`, and slices `raw_bytes[offset : offset + elem_bytes].view(dtype).reshape(shape)`.
- This guarantees that every KV cache tensor (`k_cache`, `v_cache`, `rope_cache`, `quant_k_cache`) starts on a strict 2 MiB physical HBM boundary required by HIXL RDMA/RoCE zero-copy registration.

This unit is a source-derived inspection procedure at the exact repository revision in `conditions`.

Pinned source ranges:

- [executor/core/kv_cache/kv_cache_manager.py:45-225](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/kv_cache_manager.py#L45-L225)
- [executor/core/kv_cache/single_type_kv_cache_manager.py:29-410](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/single_type_kv_cache_manager.py#L29-L410)
- [executor/core/kv_cache/cache_utils.py:276-706](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/kv_cache/cache_utils.py#L276-L706)
