---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:64be14ed54ddfeb3a17790cf7f22fcb9a483733e4b8e722f02a89ec7b7726d4b
domain_path: serving/sglang/lifecycle
evidence_strength: single_observation
id: unit-sglang-scheduler-abort-and-outstanding-work
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-scheduler-abort-and-outstanding-work-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3487-L3531
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3533-L3585
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3713-L3762
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L4398-L4421
  source_digest: 03a995951eddaf28a009424afbd0ddd48a898202f3766727ccce47adeb28439d
  source_locator: docs/evidence/unit-sglang-scheduler-abort-and-outstanding-work-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Scheduler abort admission and resource release must respect outstanding work
triggers:
  error_signatures: []
  keywords:
  - sglang
  - scheduler
  - abort
  - admission
  - and
  - resource
  - release
  - must
  - respect
  - outstanding
  - work
---
The scheduler rejects an incoming request when the waiting queue is full unless priority scheduling finds a strictly lower-priority queued victim; equal victim priorities use later queue-entry time first. Waiting-timeout polling scans the waiting queue, whereas running-timeout polling deduplicates request IDs and skips already finished or to_finish requests. It emits AbortReq objects rather than directly removing requests; the source states that ranks must drop the same requests in the same iteration. Chunked-prefill abort is deferred to process_pending_chunked_abort: if the request has moved, the marker is dropped only when finished or no longer holding KV, otherwise abort is retried at the new location. For the still-current chunked request, sender/bootstrap metadata and KV are released before clearing the chunked marker. During decode, finishing in-flight requests are filtered, and KV pressure with such pending completions skips a decode step instead of immediately retracting the remaining requests. Pending offload copies are drained before the later retraction path.

Scope and caveats:

- Finished/to_finish/deduplication skips belong only to running-timeout polling.
- The same-iteration rank requirement and overlap drain description are source invariants; no distributed execution was performed.
- This accepted subset does not claim complete coverage of embedding ingress, multimodal validation or storage-prefetch retries.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/scheduler.py:3487-3531](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3487-L3531)
- [python/sglang/srt/managers/scheduler.py:3533-3585](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3533-L3585)
- [python/sglang/srt/managers/scheduler.py:3713-3762](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3713-L3762)
- [python/sglang/srt/managers/scheduler.py:4398-4421](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L4398-L4421)
