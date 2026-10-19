---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:28b9e2f86c2ae58983bb69e690cf5da01f27cc5e0912fc5219da6d2a44335611
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-swa-span-attachment
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-swa-span-attachment-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2767-L2872
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py#L730-L756
  source_digest: d3df7e9e119e6e925f7cdb1664b1c124b3c08bc12fed054e6019590fe70e29c2
  source_locator: docs/evidence/unit-sglang-swa-span-attachment-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: SWA window attachment mutates matched spans incrementally
triggers:
  error_signatures: []
  keywords:
  - sglang
  - swa
  - window
  - attachment
  - mutates
  - matched
  - spans
  - incrementally
---
_walk_span follows matching children until the requested end, stopping at a missing child, a device-evicted child with no backup, or a zero-length match. It yields a matched segment before checking whether that segment covers only part of the child key; a partial match is therefore yielded once before the walk stops. swa_tombstone_ranges clips these spans to the requested interval, skips live SWA values and merges adjacent missing ranges.

attach_swa_window checks tensor length and materializes the structural walk, then processes each segment: it asserts the SWA value is still absent, splits at segment boundaries and installs a cloned value through set_component_device_value. It asserts final coverage only after those mutations. Materializing the walk does not prevalidate all tombstone states, and no rollback appears in this method. If a later assertion fails after an earlier segment was attached, earlier mutations can remain. That is a conditional control-flow consequence, not a reproduced failure; callers must satisfy the whole-span matching and tombstone preconditions before invoking this operation.

Reviewed limits and corrections:

- A partial child span is yielded before termination.
- Do not describe attachment as an all-or-nothing transaction merely because the structural walk is materialized first.

Omitted scope:

- Tree sanity checks, canary walks, Rust alternatives and observed concurrency failures are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2767-2872](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2767-L2872)
- [python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py:730-756](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py#L730-L756)
