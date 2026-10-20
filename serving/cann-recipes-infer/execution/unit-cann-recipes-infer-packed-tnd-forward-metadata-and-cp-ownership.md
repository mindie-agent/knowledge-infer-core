---
category: sop
conditions:
  repository: cann/cann-recipes-infer
  revision: 3c9266727c2a25bd17d503ad37d189a134ff364b
domain_path: serving/cann-recipes-infer/execution
evidence_strength: single_observation
id: unit-cann-recipes-infer-packed-tnd-forward-metadata-and-cp-ownership
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/cann-recipes-infer-reviewed-source.json
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/forward_metadata.py#L18-L135
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/forward_data_info/forward_data_info.py#L38-L412
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/engine/execution_engine.py#L50-L286
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L290-L329
  source_digest: 899e46172cab0f4bab6ee5551d9b7d093ae9fc75941114dae299d6b0cc2dfddb
  source_locator: docs/evidence/cann-recipes-infer-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Packed TND ForwardMetaData construction, Context Parallel slot ownership, and IQR step-latency filtering
triggers:
  error_signatures:
  - TypeError in npu_fused_infer_attention_score when actual_seq_lengths is Tensor instead of list
  - Context Parallel non-owner rank fails Mamba recurrent state lookup after prefill cleanup
  - Prefill CP slot_mapping overwrites persistent KV cache blocks with temporary compute slots
  keywords:
  - cann-recipes-infer
  - ForwardMetaData
  - PrefillCPMetaData
  - ForwardDataInfo
  - get_request_owner_manager_keys
  - actual_seq_lengths_kv
  - process_infer_time
---
At the pinned `cann/cann-recipes-infer` revision, `ForwardDataInfo` (`executor/core/forward_data_info/forward_data_info.py:38-412`), `ForwardMetaData` (`executor/utils/forward_metadata.py:18-135`), and `ExecutionEngine` (`executor/core/engine/execution_engine.py:50-286`) enforce strict contracts between host-list vs device-tensor sequence lengths for packed `TND` execution, separate temporary vs persistent KV slots under Prefill Context Parallelism (`CP`), and retain Mamba state on non-owner CP ranks.

## 1. Packed `TND` `ForwardMetaData` Host-List vs Device-Tensor Contract (`forward_metadata.py:18-135`)

1. **Dual Representation of Sequence Lengths (`ForwardMetaData`, `lines 51-118`)**:
   - `actual_seq_lengths_q: Union[torch.Tensor, List[int]]` and `actual_seq_lengths_kv: Union[torch.Tensor, List[int]]` are maintained alongside explicit host lists `seq_lens_list: List[int]` and `actual_seq_lengths_q_list: List[int]`.
   - Why: Ascend NPU operators differ in whether they consume host-side `List[int]` tiling descriptors (e.g., `torch_npu.npu_fused_infer_attention_score` `actual_seq_lengths` / `actual_seq_lengths_kv`) or device-resident `int32` tensors (e.g., custom Triton/AscendC kernels and graph-captured inputs). Pre-building both in `ForwardDataInfo.prepare_forward_inputs()` avoids per-layer `.tolist()` D2H synchronizations inside the model forward pass.
2. **Prefill Context Parallelism (`PrefillCPMetaData`, `lines 18-48`)**:
   - Separates `slot_mapping_cp` (temporary compute KV cache slot mapping for local CP chunk attention) from `slot_mapping_ persist` (persistent KV cache slot mapping owned by the target rank after CP all-gather/redistribution), alongside `gather_index` and `actual_seq_lengths_q_Next` / `actual_seq_lengths_kv_prev`.

## 2. Context Parallel Request Ownership vs Mamba State Retention (`forward_data_info.py:165-238`)

1. **Selective Manager Key Ownership (`get_request_owner_manager_keys`, `lines 165-190`)**:
   - Under Context Parallelism (`attn_cp_size > 1`), each request is assigned an owner rank (`owner_cp_rank = request_idx % attn_cp_size`) for paged KV blocks (`full_attention` and `sliding_window`).
   - However, `mamba` state is **exempted** from CP non-owner eviction: `get_request_owner_manager_keys()` returns all manager keys on the owner rank, and still includes `"mamba"` on non-owner CP ranks (`[k for k in all_keys if k == "mamba"]`) so every rank participating in Mamba layers retains its recurrent state blocks across steps.
2. **Temporary CP Block Reclamation (`free_temp_cp_blocks`, `lines 215-238`)**:
   - Temporary KV blocks allocated on non-owner CP ranks during prefill are freed immediately after prefill via `kv_cache_manager.free_by_keys(request_id, non_owner_keys)`, reclaiming paged KV blocks while leaving `"mamba"` state intact.

## 3. Interquartile Range (`IQR`) Step-Latency Outlier Filtering (`common_utils.py:290-329`, `execution_engine.py:240-286`)

- `ExecutionEngine` records per-step prefill and decode latencies across warmup and benchmark iterations and passes them to `process_infer_time(infer_time_list)`.
- When `len(infer_time_list) >= 4`, `process_infer_time()` computes `Q1` (25th percentile) and `Q3` (75th percentile), sets `IQR = Q3 - Q1`, filters out samples outside `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]`, and returns the mean of the inlier latencies to prevent single-step compilation/HCCL warmup spikes from skewing steady-state TPOT/TTFT reporting.

This unit is a source-derived inspection procedure at the exact repository revision in `conditions`.

Pinned source ranges:

- [executor/utils/forward_metadata.py:18-135](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/forward_metadata.py#L18-L135)
- [executor/core/forward_data_info/forward_data_info.py:38-412](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/forward_data_info/forward_data_info.py#L38-L412)
- [executor/core/engine/execution_engine.py:50-286](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/core/engine/execution_engine.py#L50-L286)
- [executor/utils/common_utils.py:290-329](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L290-L329)
