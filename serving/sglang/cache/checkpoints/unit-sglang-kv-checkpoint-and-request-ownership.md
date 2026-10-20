---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:1271aefffa92483c1049d177b4e0b937de3bb894a93549d6d81c608700197abf
domain_path: serving/sglang/cache/checkpoints
evidence_strength: single_observation
id: unit-sglang-kv-checkpoint-and-request-ownership
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-kv-checkpoint-and-request-ownership-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/common.py#L270-L279
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/common.py#L395-L439
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/allocation.py#L175-L201
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/allocation.py#L219-L246
  source_digest: a381be620b52f773432d4ca8f1e60fec68876671ec79d620e7e0c34c5c71fc4a
  source_locator: docs/evidence/unit-sglang-kv-checkpoint-and-request-ownership-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: KV checkpoint and release separate tree ownership from request allocation
triggers:
  error_signatures: []
  keywords:
  - sglang
  - kv
  - checkpoint
  - and
  - release
  - separate
  - tree
  - ownership
  - from
  - request
  - allocation
---
checkpoint_kv_cache requires an unfinished request, optionally publishes through tree_cache.checkpoint and advances prefix_len even when insertion is skipped. release_kv_cache first handles a Mamba-only allocation or a streaming session that claims the row. On the normal path it optionally checkpoints owned KV, requiring a finished request for a Mamba tree, then frees only the request-owned suffix after cache_protected_len, unlocks the tree, releases overallocation, calls on_release, frees the request row and marks KV released. Paged extension computes a conservative demand with per-request page and shard headroom, passes that estimate to get_extend_allocation_demand and evicts the returned demand, which need not equal the conservative estimate. The subsequent allocator result is checked and a missing allocation raises a visible prefill OOM error.

Scope and caveats:

- The conservative estimate is an input to the allocator demand function, not necessarily the exact eviction request.
- A checkpoint preserves a tree-owned prefix; releasing a request must not free that protected prefix.
- Whole-file citations were replaced with the specific ownership and allocation branches; native hashing and speculative allocation claims are not included in this subset.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/common.py:270-279](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/common.py#L270-L279)
- [python/sglang/srt/mem_cache/common.py:395-439](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/common.py#L395-L439)
- [python/sglang/srt/mem_cache/allocation.py:175-201](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/allocation.py#L175-L201)
- [python/sglang/srt/mem_cache/allocation.py:219-246](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/allocation.py#L219-L246)
