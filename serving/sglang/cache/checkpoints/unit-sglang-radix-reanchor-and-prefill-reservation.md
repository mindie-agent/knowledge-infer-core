---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:ad1d15d17e923fc21b0092cf8f57644836a89fd3681098d3cad537cfbd9778cf
domain_path: serving/sglang/cache/checkpoints
evidence_strength: single_observation
id: unit-sglang-radix-reanchor-and-prefill-reservation
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-radix-reanchor-and-prefill-reservation-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/radix_cache.py#L510-L543
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/radix_cache.py#L582-L616
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/prefill_budget.py#L91-L140
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/prefill_budget.py#L156-L168
  source_digest: f1ac24a9709a95b027cf2c242d99335eb7df1279bf9e72bb799f0cebcc23fd61
  source_locator: docs/evidence/unit-sglang-radix-reanchor-and-prefill-reservation-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Radix checkpoint reanchors ownership and prefill reserves future capacity separately
triggers:
  error_signatures: []
  keywords:
  - sglang
  - radix
  - checkpoint
  - reanchors
  - ownership
  - and
  - prefill
  - reserves
  - future
  - capacity
  - separately
---
RadixCache.checkpoint excludes an already-evicted SWA prefix, inserts the eligible key, rematches it and rewrites the request row to the tree-owned indices. The protected length is only the cached aligned prefix; a partial page remains request-owned. It releases the old lock and locks the new last node. Lock-reference transitions move token counts between evictable and protected categories only at zero-to-one and one-to-zero boundaries. PrefillBudget admits its total-token check only when requested total is strictly less than remaining total, while immediate paged allocation uses a less-than-or-equal comparison against both current and total capacity. reserve page-aligns the extension, adds an extra page and optional extra tokens to both offsets, and charges max_new_tokens only to total_offset. These inequalities and headroom charges serve distinct admission decisions.

Reviewed limits and corrections:

- Do not treat partial-page tokens outside cache_protected_len as tree-owned.
- Do not replace the two admission inequalities with one generic available-capacity rule.
- The selected subset does not verify page-interleaved communication or backend integration.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/radix_cache.py:510-543](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/radix_cache.py#L510-L543)
- [python/sglang/srt/mem_cache/radix_cache.py:582-616](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/radix_cache.py#L582-L616)
- [python/sglang/srt/mem_cache/prefill_budget.py:91-140](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/prefill_budget.py#L91-L140)
- [python/sglang/srt/mem_cache/prefill_budget.py:156-168](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/prefill_budget.py#L156-L168)
