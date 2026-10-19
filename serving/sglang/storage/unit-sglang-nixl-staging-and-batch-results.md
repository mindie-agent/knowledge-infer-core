---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:c1b701864dc2633ebc7f5c042cf77455aca9a3c2bfdd7b7b7f393be1b130b583
domain_path: serving/sglang/storage
evidence_strength: single_observation
id: unit-sglang-nixl-staging-and-batch-results
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-nixl-staging-and-batch-results-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L343-L408
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L415-L457
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L753-L781
  source_digest: 9f3975c6ba00c5b4aafef7101ab6e69be16e59e45cd6c347ff665991837d7e9a
  source_locator: docs/evidence/unit-sglang-nixl-staging-and-batch-results-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: NIXL selects staging per host pool and reports batch-level results
triggers:
  error_signatures: []
  keywords:
  - sglang
  - nixl
  - selects
  - staging
  - per
  - host
  - pool
  - and
  - reports
  - batch-level
  - results
---
The NIXL backend initially enables zero-copy for page_first or page_first_direct host layouts. An absent kv_buffer represents a logical anchor and forces copy mode with a small marker buffer, using 4096 bytes when page alignment is required. When alignment is required for a physical pool, failed base/stride alignment also disables zero-copy. Copy mode allocates separate registered set and get bounce buffers using a dummy page's shape and dtype. Hybrid non-KV pools make a separate registration choice and retain it in their own context; they either register all hybrid buffers or allocate their own pair of bounce buffers.

The V1 _batch_xfer path computes one success value for the underlying transfer and repeats it across the result list. Reads use one result per key string, while writes use one per logical page. Thus that list does not expose independent per-page outcomes. These source branches establish staging and reporting behavior only; they do not prove absence of synchronization races, durability, or zero-copy performance on a particular backend.

Reviewed limits and corrections:

- Call the allocations registered bounce buffers; their allocator implementation is not shown in this subset.
- Repeated batch success is not a collection of separately observed page successes.

Omitted scope:

- NIXL cleaner routing, other storage backends, v2 existence and hardware or filesystem acceptance are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:343-408](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L343-L408)
- [python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:415-457](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L415-L457)
- [python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:753-781](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L753-L781)
