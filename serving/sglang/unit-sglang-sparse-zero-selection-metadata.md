---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:e8fc69becebd1c5e74359f33df3caba844177ac9414369dcfdafa11a529e1c1a
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-sparse-zero-selection-metadata
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-sparse-zero-selection-metadata-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py#L270-L276
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py#L293-L355
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py#L110-L145
  source_digest: 9e362dace528a1317dc2d2fc1aa866b854ae5e7913cfc7fe85dc6acadf4fc12c
  source_locator: docs/evidence/unit-sglang-sparse-zero-selection-metadata-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Sparse selection can leave zero pages before metadata rewriting
triggers:
  error_signatures: []
  keywords:
  - sglang
  - sparse
  - selection
  - can
  - leave
  - zero
  - pages
  - before
  - metadata
  - rewriting
---
The sparse coordinator gates requests by stored prompt length against min_sparse_prompt_len. retrieve_topk returns an empty selection for a non-sparse row and also for a sparse row whose page count is no greater than num_recent_pages. Otherwise it scores historical pages, excludes recent pages from that selection, chooses at least one historical page within the available history, and appends recent pages. Consequently the total retained-page count need not equal the history ratio alone.

FlashAttentionAdaptor returns the current metadata unchanged when its saved original metadata is absent (self._original_metadata is None) or no row is sparse. Otherwise it copies the saved original page table and sequence lengths into the current metadata, writes page-table entries only where sparse_mask and valid_mask both hold, and replaces sparse sequence lengths using valid_lengths times page_size minus the partial-page adjustment. Under the source-level condition of a sparse row with zero valid lengths, the formula yields zero for an aligned final page or a negative adjustment otherwise, while no selected page-table entry is updated. This is an arithmetic consequence of the shown branches, not an observed backend failure; callers and workload reachability require separate validation.

Reviewed limits and corrections:

- The short-history branch returns an empty selection; it does not implement a dense fallback.
- Keep the zero-length metadata consequence explicitly conditional and source-derived.
- The early return tests saved original metadata, not the current metadata object; the copy restores the saved arrays.

Omitted scope:

- Offload orchestration, alternate sparse algorithms and hardware failure or speed claims are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py:270-276](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py#L270-L276)
- [python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py:293-355](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py#L293-L355)
- [python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py:110-145](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py#L110-L145)
