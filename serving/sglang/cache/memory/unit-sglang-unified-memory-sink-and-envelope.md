---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:fa1e99c5676ae145d507f711a2baba4e2d9b6fc336358646f1dd849f164ee908
domain_path: serving/sglang/cache/memory
evidence_strength: single_observation
id: unit-sglang-unified-memory-sink-and-envelope
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-unified-memory-sink-and-envelope-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L350-L368
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L388-L409
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L533-L542
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L883-L903
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1013-L1025
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1041-L1068
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1240-L1257
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1545-L1567
  source_digest: 4f905a2b091ccbc14f71f88398fb5d6f9de20060dd0f59e065a4cb38d4adf159
  source_locator: docs/evidence/unit-sglang-unified-memory-sink-and-envelope-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Unified memory reserves a physical sink and exposes mode-specific envelope
  contracts
triggers:
  error_signatures: []
  keywords:
  - sglang
  - unified
  - memory
  - reserves
  - a
  - physical
  - sink
  - and
  - exposes
  - mode-specific
  - envelope
  - contracts
---
Unified memory reserves a slot-zero sink equal to the maximum of one entry for every sub-pool and one whole page for non-Mamba sub-pools. A sub-pool must have a real slot above that floor. Unfused Mamba entry_bytes returns the raw host-entry bytes, whereas fused draft state is aligned through a separate path. MHA physical page moves assert a zero sub-pool anchor before copying; the shown MLA move and zero helpers do not perform that anchor assertion, although MLA transfer registration does. This is a source-level check asymmetry, not a demonstrated corruption. MLA and Mamba transfer metadata use the raw buffer pointer and envelope stride, with Mamba per-tensor state metadata empty. The factory feasibility helper rejects a byte budget below the sum of its itemized single-request floor; its livelock rationale is a source invariant, not a reproduced runtime event.

Reviewed limits and corrections:

- Do not describe all Mamba entries as uniformly aligned.
- Zero-anchor checks are path-specific; their asymmetry alone does not establish an actual device fault.
- The accepted ranges do not prove MultiEndedAllocator compaction order or inter-stream synchronization.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/unified_memory_pool.py:350-368](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L350-L368)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:388-409](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L388-L409)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:533-542](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L533-L542)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:883-903](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L883-L903)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:1013-1025](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1013-L1025)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:1041-1068](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1041-L1068)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:1240-1257](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1240-L1257)
- [python/sglang/srt/mem_cache/unified_memory_pool.py:1545-1567](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_memory_pool.py#L1545-L1567)
