---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:72b795d0b8072de71cb34cc3220c503a87e43a74ac11c5b10e4514aea085d964
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-internal-state-and-pool-invariants
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-internal-state-and-pool-invariants-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L147-L167
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L195-L215
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L255-L282
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/invariant_checker.py#L78-L194
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/invariant_checker.py#L428-L450
  source_digest: c7730f04973d4c395ee368700dd8a52a7815cf6ff25c746e66db3ae33599b818
  source_locator: docs/evidence/unit-sglang-internal-state-and-pool-invariants-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Internal-state updates validate before applying while pool invariants remain
  mode-specific
triggers:
  error_signatures: []
  keywords:
  - sglang
  - internal-state
  - updates
  - validate
  - before
  - applying
  - while
  - pool
  - invariants
  - remain
  - mode-specific
---
SchedulerInternalStateController validates every requested update before its apply loop and catches UpdateRejected only during validation. The apply loop can directly mutate worker state before a later handler fails; no rollback is shown. DFlash acceptance-threshold relaxation is rejected while an unfinished request carries a sampling mask, including running and PP batches, waiting and grammar queues, chunked requests, and decode preallocation, transfer, retracted and held-rebootstrap queues. The generic pool identity adds available, evictable, protected, session-held and uncached tokens. Hybrid-SSM total capacity comes from req_to_token_pool.schedulable_token_capacity. Page-interleaved sharding uses aggregate free pages and checks alignment; the separate legacy DCP branch tolerates a residual mismatch. SWA request-ring token conservation is skipped, and the request-pool check instead compares free slots plus session-held requests with size plus preallocation headroom.

Reviewed limits and corrections:

- Validate-then-apply is not an atomic transaction.
- The sampling-mask function actually begins at line 255 and its final queue branches and predicate end at 282; the model review range 252-279 was still incomplete.
- Conservation exceptions are explicit mode conditions, not evidence that every residual represents a leak.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/scheduler_components/internal_state.py:147-167](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L147-L167)
- [python/sglang/srt/managers/scheduler_components/internal_state.py:195-215](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L195-L215)
- [python/sglang/srt/managers/scheduler_components/internal_state.py:255-282](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/internal_state.py#L255-L282)
- [python/sglang/srt/managers/scheduler_components/invariant_checker.py:78-194](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/invariant_checker.py#L78-L194)
- [python/sglang/srt/managers/scheduler_components/invariant_checker.py:428-450](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/invariant_checker.py#L428-L450)
