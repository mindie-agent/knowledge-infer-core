---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:526d78059cd37349c0b31a1fcd9cd83c4cab3bf0c5545df8cdd8dc8d6da9135f
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-tokenizer-dispatch-and-abort-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-tokenizer-dispatch-and-abort-boundary-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1781-L1822
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1936-L1944
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1964-L1983
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L2253-L2272
  source_digest: abe3f9cc1d4329ee922a373acf9c5ca7852cd638243c43b4750bf57e905e26b5
  source_locator: docs/evidence/unit-sglang-tokenizer-dispatch-and-abort-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Tokenizer dispatch bookkeeping and abort requests have separate completion
  boundaries
triggers:
  error_signatures: []
  keywords:
  - sglang
  - tokenizer
  - dispatch
  - bookkeeping
  - and
  - abort
  - requests
  - have
  - separate
  - completion
  - boundaries
---
TokenizerManager prepares multimodal and positional-embedding transports, dispatches to the scheduler, marks the request state as dispatched, and only then sets its local dispatched flag. The finally block cancels prepared transports only while that flag remains false; this records local control flow, not a scheduler acknowledgement. An Engine request with no HTTP request object waits on its state event without the HTTP polling timeout, whereas an HTTP request periodically checks disconnect state. Single-request abort rejects an empty rid, deduplicates live states with abort_sent, and resets that flag if scheduler dispatch raises. abort_all bypasses the per-state lookup and deduplication. LoRA release has its own lora_released flag and schedules at most one release task when LoRA is enabled and the request carries a LoRA path.

Reviewed limits and corrections:

- Use actual narrow boundaries 1781-1822, 1936-1944 and 2253-2272 rather than the approximate review endpoints.
- A local dispatched flag does not prove remote acceptance or downstream KV release.
- No global at-most-once claim applies to abort_all, and no bounded completion time is inferred from scheduling a LoRA release task.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/tokenizer_manager.py:1781-1822](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1781-L1822)
- [python/sglang/srt/managers/tokenizer_manager.py:1936-1944](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1936-L1944)
- [python/sglang/srt/managers/tokenizer_manager.py:1964-1983](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L1964-L1983)
- [python/sglang/srt/managers/tokenizer_manager.py:2253-2272](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L2253-L2272)
