---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:01a8389223385a9e6f15aa4c10a36bf8a0e330c5e5b9163ebdbf99c6cbb98dd3
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-host-logical-physical-slot-geometry
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-host-logical-physical-slot-geometry-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L111-L150
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L193-L225
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L492-L567
  source_digest: ea95502e0cef09a682b9291ac49c62152ebb25e2b59ae2c205bd772fdb4deae6
  source_locator: docs/evidence/unit-sglang-host-logical-physical-slot-geometry-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Host KV pages separate physical geometry from logical slots
triggers:
  error_signatures: []
  keywords:
  - sglang
  - host
  - kv
  - pages
  - separate
  - physical
  - geometry
  - from
  - logical
  - slots
---
HostKVCache requires the supplied widened page width to be divisible by dcp_size and stores the quotient as its physical page size. Logical size and logical page size multiply the physical values back by dcp_size. Explicit host_size uses decimal gigabytes divided by bytes per token and, when a multi-rank initialized PP context is available, synchronizes the fixed capacity with a CPU-group MIN reduction. Ratio sizing instead scales device token capacity. The final page count is size divided by physical page size, floored, plus one: a nonaligned capacity rounds up, while an already aligned capacity gains a full page.

clear initializes logical free-slot indices, usage flags and an empty deferred-release queue. alloc requires a multiple of logical_page_size, returns None if total availability is insufficient, merges deferred releases only when the immediate free list is too short, and marks selected slots used. free returns on empty input, asserts the supplied slots are in use, clears their flags and queues them for later reuse. This is slot bookkeeping, not a standalone concurrency or lease guarantee.

Reviewed limits and corrections:

- The final rounding formula deliberately adds a page even at exact alignment.
- Distinguish physical page width from widened logical allocation width.

Omitted scope:

- Host allocator backend choice, CUDA registration and measured capacity are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/pool_host/base.py:111-150](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L111-L150)
- [python/sglang/srt/mem_cache/pool_host/base.py:193-225](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L193-L225)
- [python/sglang/srt/mem_cache/pool_host/base.py:492-567](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/pool_host/base.py#L492-L567)
