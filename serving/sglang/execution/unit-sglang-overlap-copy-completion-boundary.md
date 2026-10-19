---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:7d7c1ef50be6cf224723254bcf2bf21a7f64ea03728f91d1a720e15ac70f96bd
domain_path: serving/sglang/execution
evidence_strength: single_observation
id: unit-sglang-overlap-copy-completion-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-overlap-copy-completion-boundary-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/overlap_utils.py#L212-L253
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/overlap_utils.py#L544-L604
  source_digest: c62fb5dc162960d8f57dabd401d165167de81fc3b5477f637f85e51543059707
  source_locator: docs/evidence/unit-sglang-overlap-copy-completion-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Overlap CPU consumers wait on copy completion while stream ordering has a narrower
  role
triggers:
  error_signatures: []
  keywords:
  - sglang
  - overlap
  - cpu
  - consumers
  - wait
  - 'on'
  - copy
  - completion
  - while
  - stream
  - ordering
  - has
  - a
  - narrower
  - role
---
The overlap confidence-copy path schedules a device-to-host copy after its producer dependency and records a completion event; the CPU consumer queries completion before using that copied confidence. These excerpts contain no wait for a previous occupant when a ring slot is selected, so a complete slot-reuse lifetime proof cannot be inferred. resolve_seq_lens_cpu can skip the initial event wait when stream_ordered is set, but a private device-to-host copy still has its own completion wait before CPU use. When CPU sequence lengths are unnecessary it returns None; otherwise the selected path synchronizes the transfer before returning host values.

Reviewed limits and corrections:

- A producer stream dependency and a CPU-visible copy completion are separate boundaries.
- Do not infer safe ring-buffer slot reuse from the confidence consumer check alone.
- No real asynchronous device race or throughput effect was measured.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/overlap_utils.py:212-253](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/overlap_utils.py#L212-L253)
- [python/sglang/srt/managers/overlap_utils.py:544-604](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/overlap_utils.py#L544-L604)
