---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:64a0112e1b436338bbe00d8aed27f3879b53f8729d05b6c88a1df753ebf9539a
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-finish-precedence-and-kv-release
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-finish-precedence-and-kv-release-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L1562-L1579
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L1996-L2087
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L2096-L2134
  source_digest: f344608ee72038e7ef79bee42592537bfc9a8445c28cda4ddf4cf2616bae3c91
  source_locator: docs/evidence/unit-sglang-finish-precedence-and-kv-release-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Request finish precedence and KV ownership release are explicit state transitions
triggers:
  error_signatures: []
  keywords:
  - sglang
  - request
  - finish
  - precedence
  - and
  - kv
  - ownership
  - release
  - are
  - explicit
  - state
  - transitions
---
Req detaches its owned KV allocation from request bookkeeping before returning the owned range, including the speculative tail. Finish evaluation first honors a deferred to_finish reason, then invalid vocabulary output, stop strings, EOS, length and grammar completion in that order. Resetting for retraction asserts that KV has already been released and clears execution state before the request is reused. The shown offload helper backs up KV but does not itself free the request allocation; the load path clears its backup reference after restoring it. Consequently finish, offload, free and retraction reset must not be collapsed into one operation.

Reviewed limits and corrections:

- Preserve the actual finish-reason precedence instead of treating all terminal conditions as interchangeable.
- A successful offload backup is not evidence that the device allocation has been released.
- The selected ranges do not prove full multimodal request normalization or all sampler semantics.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/schedule_batch.py:1562-1579](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L1562-L1579)
- [python/sglang/srt/managers/schedule_batch.py:1996-2087](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L1996-L2087)
- [python/sglang/srt/managers/schedule_batch.py:2096-2134](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_batch.py#L2096-L2134)
