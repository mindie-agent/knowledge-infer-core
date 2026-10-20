---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:9c6aef9758b7de34514000f36205c6e80023419413cd2aebdef19e9f8af48f6a
domain_path: serving/sglang/execution
evidence_strength: single_observation
id: unit-sglang-manager-completion-and-ownership
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-manager-completion-and-ownership-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/async_d2h_copy_worker.py#L11-L40
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/async_d2h_copy_worker.py#L79-L106
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/communicator.py#L41-L79
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/communicator.py#L90-L103
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/cache_controller.py#L1102-L1153
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/cache_controller.py#L1421-L1433
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L121-L154
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L233-L239
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L321-L324
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L343-L358
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L875-L899
  source_digest: df642e15ebdfb73ede7ea3a8c1f36cde1e13af1ab0c853d7ab52c0b14cb90b1d
  source_locator: docs/evidence/unit-sglang-manager-completion-and-ownership-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: SGLang manager completion signals retain distinct success and ownership conditions
triggers:
  error_signatures: []
  keywords:
  - sglang
  - manager
  - completion
  - signals
  - retain
  - distinct
  - success
  - and
  - ownership
  - conditions
---
At the pinned SGLang revision, HostCopyDone.query reports completion even when a copy failed; synchronize waits and rethrows a stored error. The copy worker signals completion in finally after waiting for the producer event and using its private stream, while shutdown only joins for the requested timeout without a post-join liveness check. FanOutCommunicator queueing calls share an untagged event and result list, clear them only after the wait returns, and count received responses against fan_out. Cancellation of a waiter does not itself correlate late responses or clear this shared state. HiCache _page_transfer continues producing one PrefetchAck per storage batch after cooperative termination or a failed batch; the consumer MIN-reduces completed token counts across completion groups. DPBudget speculatively increments requests and estimated input tokens; refresh is throttled to 20 ms and total_tokens_scheduler charges len(input_ids). This is a dispatch heuristic, not an observed completion cost. Follow-bootstrap routing takes bootstrap_room modulo the entire worker list without checking active status at that point; elastic activation asserts that its socket was already bound, so an inactive slot does not necessarily mean a None socket.

Scope and caveats:

- Completion is not copy success; destination validity requires the error-bearing synchronization path.
- Untagged shared communicator state does not prove that an actual race occurred in a running service.
- Preserve the per-batch acknowledgement cadence even when no further storage copy succeeds.
- Replace the candidate's inactive-slot-implies-None claim with the narrower missing active-status guard.
- No throughput, latency improvement, hardware behavior or complete scheduler cancellation protocol was measured.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/async_d2h_copy_worker.py:11-40](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/async_d2h_copy_worker.py#L11-L40)
- [python/sglang/srt/managers/async_d2h_copy_worker.py:79-106](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/async_d2h_copy_worker.py#L79-L106)
- [python/sglang/srt/managers/communicator.py:41-79](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/communicator.py#L41-L79)
- [python/sglang/srt/managers/communicator.py:90-103](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/communicator.py#L90-L103)
- [python/sglang/srt/managers/cache_controller.py:1102-1153](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/cache_controller.py#L1102-L1153)
- [python/sglang/srt/managers/cache_controller.py:1421-1433](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/cache_controller.py#L1421-L1433)
- [python/sglang/srt/managers/data_parallel_controller.py:121-154](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L121-L154)
- [python/sglang/srt/managers/data_parallel_controller.py:233-239](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L233-L239)
- [python/sglang/srt/managers/data_parallel_controller.py:321-324](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L321-L324)
- [python/sglang/srt/managers/data_parallel_controller.py:343-358](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L343-L358)
- [python/sglang/srt/managers/data_parallel_controller.py:875-899](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/data_parallel_controller.py#L875-L899)
