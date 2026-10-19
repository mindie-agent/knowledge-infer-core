---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:1ed2253c8c69d3383d4db7e4b7aabd96285e4dcdc04bc4ea4a6761d80691b496
domain_path: serving/sglang/lifecycle
evidence_strength: single_observation
id: unit-sglang-decode-retraction-and-oom-abort
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-decode-retraction-and-oom-abort-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L2325-L2360
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L3389-L3476
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L3752-L3775
  source_digest: e1e48f05fea4e5ac452520b013099a8c4c114af837ccf135d001bc6d873c1117
  source_locator: docs/evidence/unit-sglang-decode-retraction-and-oom-abort-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Decode retraction preserves the final request only until its explicit OOM abort
  path
triggers:
  error_signatures: []
  keywords:
  - sglang
  - decode
  - retraction
  - preserves
  - the
  - final
  - request
  - only
  - until
  - its
  - explicit
  - oom
  - abort
  - path
---
ScheduleBatch.release_req conditionally backs up decode KV to host, then releases request ownership, evicts as required and resets request state. During decode retraction the loop avoids retracting the last remaining request. If memory still cannot satisfy that last request, a separate path marks it aborted for out-of-memory and removes it through batch filtering; an empty batch is therefore a valid resulting state. filter_batch explicitly clears the request list when the retained index set is empty. Avoiding the final normal retraction is not an unconditional guarantee that one request survives.

Reviewed limits and corrections:

- Replace the always-keep-one assertion with the separate last-request OOM-abort behavior.
- Host backup is conditional and should not be assumed for every request release.
- No allocator capacity, running OOM rate or hardware behavior was measured.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/schedule_batch.py:2325-2360](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L2325-L2360)
- [python/sglang/srt/managers/schedule_batch.py:3389-3476](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L3389-L3476)
- [python/sglang/srt/managers/schedule_batch.py:3752-3775](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L3752-L3775)
