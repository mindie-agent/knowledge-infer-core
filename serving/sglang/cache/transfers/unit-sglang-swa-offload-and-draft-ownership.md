---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:9b3154d40b6e01dcd72f4131814e91f53b4f823ba0abd319cd94d629f2b4136e
domain_path: serving/sglang/cache/transfers
evidence_strength: single_observation
id: unit-sglang-swa-offload-and-draft-ownership
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-swa-offload-and-draft-ownership-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L159-L167
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L240-L251
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L368-L424
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L221-L231
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L256-L266
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L520-L543
  source_digest: ca8d189a00b60d1e1c579203f3353d54bb3010ea1f5bbcb986f78f4ceff3c76c
  source_locator: docs/evidence/unit-sglang-swa-offload-and-draft-ownership-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: SWA offload masks and fused draft wrappers retain asymmetric ownership
triggers:
  error_signatures: []
  keywords:
  - sglang
  - swa
  - offload
  - masks
  - and
  - fused
  - draft
  - wrappers
  - retain
  - asymmetric
  - ownership
---
SWAKVPool owns layer-transfer waiting and clears the inner pools' counters to avoid duplicate waits. Its CPU-copy path saves only SWA indices greater than zero, preserving a mask because zero is a dummy slot and tail-only allocations leave some full indices unmapped. Restore intersects that saved selection with the new mapping and skips an empty resulting SWA set. UnifiedDraftSWAKVPool routes full-to-SWA translation through the host allocator; its overridden key/value/KV getters do not wait, but the inherited scale-buffer getter still calls _wait_for_layer. Therefore storing a transfer counter can matter on the scale path. bind_fused_draft deduplicates families in placement.hosts order and applies binders in that order, despite the docstring saying registry order.

Scope and caveats:

- The draft wrapper's counter is not universally ineffective: inherited scale access still waits.
- CPU restore uses old and new mapping membership, rather than blindly replaying saved SWA rows.
- Use actual host-placement order rather than the binder-order docstring; no runtime offload or compaction test was performed.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/swa_memory_pool.py:159-167](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L159-L167)
- [python/sglang/srt/mem_cache/swa_memory_pool.py:240-251](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L240-L251)
- [python/sglang/srt/mem_cache/swa_memory_pool.py:368-424](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/swa_memory_pool.py#L368-L424)
- [python/sglang/srt/mem_cache/unified_draft_pool.py:221-231](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L221-L231)
- [python/sglang/srt/mem_cache/unified_draft_pool.py:256-266](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L256-L266)
- [python/sglang/srt/mem_cache/unified_draft_pool.py:520-543](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_draft_pool.py#L520-L543)
