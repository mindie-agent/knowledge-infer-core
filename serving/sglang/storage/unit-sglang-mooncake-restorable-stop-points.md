---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:04700dedc5e090305ee2dec4308c9ef60a2a04f681bdcea8ca50799d13990054
domain_path: serving/sglang/storage
evidence_strength: single_observation
id: unit-sglang-mooncake-restorable-stop-points
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-mooncake-restorable-stop-points-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L868-L949
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L951-L1041
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1043-L1092
  source_digest: 5ea18048d724fbf7727cb7ad2afca7934994b20b373cbf401ea90e2a26d9b38d
  source_locator: docs/evidence/unit-sglang-mooncake-restorable-stop-points-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Mooncake v2 intersects restorable stop points across component pools
triggers:
  error_signatures: []
  keywords:
  - sglang
  - mooncake
  - v2
  - intersects
  - restorable
  - stop
  - points
  - across
  - component
  - pools
---
MooncakeStore.batch_exists_v2 treats an absent physical kv_buffer as a logical anchor: all requested KV prefix lengths initially remain candidates. Otherwise it uses the existing KV prefix. For each side pool, a page exists only when all required components return existence state 1. ALL_PAGES keeps prefix lengths before the first missing page. TRAILING_PAGES evaluates the complete trailing window for each potential stop point, so its accepted set can contain holes. The method intersects these sets across pools and returns both the largest retained stop point and the full retained set. The comment requires callers to preserve that set for cross-rank intersection; the largest number alone cannot represent all valid stop points.

The v2 IO path batches component keys and buffer metadata across transfers. Sets first query existence, treat state 1 as already successful, and put only missing components; gets perform the read directly. Metadata generation selects pages with missing components and leaves zero placeholders for fully present pages. Partially missing pages retain metadata for all components, while IO submits only missing components. These contracts describe local batching and result representation, without establishing atomicity or measured RPC savings.

Reviewed limits and corrections:

- A non-contiguous restorable set is intentional for trailing-window policies; do not mislabel the largest valid stop point as an unsafe prefix.
- The full set is needed to describe valid stop points and the comment-described cross-rank intersection.

Omitted scope:

- Group semantics internals, transfer atomicity and measured backend performance are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:868-949](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L868-L949)
- [python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:951-1041](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L951-L1041)
- [python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:1043-1092](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1043-L1092)
