---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:6eb78ac9a351459ca5c30dc1cebf4e30662a80eb53a420b0b4726ec6ce1b1ad4
domain_path: serving/sglang/lifecycle
evidence_strength: single_observation
id: unit-sglang-cancellation-and-cache-ownership
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/sglang-reviewed-source.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L2253-L2277
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L5548-L5689
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L3713-L3763
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1126-L1297
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1379-L1386
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L354-L364
  source_digest: 216aeb7c82a645720df20329dd096ce2a51e8733856f5905a7d433840309511f
  source_locator: docs/evidence/sglang-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Trace SGLang abort state, cache ownership and local asynchronous drains
triggers:
  error_signatures: []
  keywords:
  - sglang
  - cancellation
  - reclamation
  - ownership
  - cache
---
Use this diagnostic for the scheduler and unified radix-cache paths at the
exact repository revision in conditions. Identify the request's current owner
and path before interpreting an abort as completed reclamation.

1. Trace the affected request through `abort_request`. Distinguish an in-flight
   chunked request recorded for later handling, a waiting request removed
   directly, and a running request marked through `to_finish`. In the
   host-staged decode branch, verify that the host destination is retained and
   abort is prepared; that branch does not call `receiver.abort`.
   [Abort branches and host destination lifetime](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler.py#L5548-L5689)
2. For the unified radix-cache path, inspect `checkpoint` rewriting the
   request-to-token mapping, moving the request lock and running component
   cleanup. Session reference registration requires
   `enable_session_radix_cache`, `is_finished`, and a finish reason other than
   `FINISH_ABORT`; an abort alone does not satisfy this registration condition.
   [Checkpoint ownership and session-reference conditions](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1126-L1297)
3. Inspect `_drain_device_frees` and the `FreeComponentDeviceSlot` actions it
   dispatches. Keep the downstream action implementation outside this accepted
   subset; dispatch alone is not evidence of physical device-memory release.
   [Device-free action dispatch](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1379-L1386)
4. Inspect `_drain_async_work`'s own `work_list`: it waits on each handle and
   clears that list. Treat this as completion of that inspected list, without
   extending it to every cache, scheduler, transfer or rank operation.
   [Local asynchronous-work drain](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L354-L364)

These are source observations and bounded diagnostic steps. They establish no
immediate interruption of already-launched device work, reproduced distributed
failure, physical-memory measurement, or global-quiescence guarantee. The
accepted source subset and exact file/range hashes are in `docs/evidence/sglang-reviewed-source.json`.
