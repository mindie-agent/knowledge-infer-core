---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:d2579ec7e7d565decbdc18c257af5fe1b78de504bed188170c63b4db4c8ed558
domain_path: serving/sglang/admission
evidence_strength: single_observation
id: unit-sglang-queue-fallback-and-chunk-budgets
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-queue-fallback-and-chunk-budgets-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L214-L262
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L301-L312
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L834-L853
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L1544-L1558
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L1581-L1611
  source_digest: d2ccf7c0fe5cf8d8d037136f6ecf4c8a9e572431982e384ce463b762f88a3e99
  source_locator: docs/evidence/unit-sglang-queue-fallback-and-chunk-budgets-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Scheduler queue fallback and chunk budgets retain distinct selection behavior
triggers:
  error_signatures: []
  keywords:
  - sglang
  - scheduler
  - queue
  - fallback
  - and
  - chunk
  - budgets
  - retain
  - distinct
  - selection
  - behavior
---
The configured LPM or HRRN policy can return an active FCFS policy when the waiting queue exceeds its configured threshold. However, calc_priority checks the configured self.policy for the explicit FCFS sorting branch; the active-policy fallback reaches a pass branch and preserves existing queue order. Thus the threshold fallback is not evidence of a newly sorted FCFS queue. Prefill budget evaluation distinguishes NO_TOKEN for memory or Mamba exhaustion from OTHER for input, diffusion-model or chunk limits. Replay can reduce the effective chunk limit, and later alignment rounds a permitted chunk down to the required boundary; a successful budget check need not admit the full remaining request.

Scope and caveats:

- Do not equate the active fallback label with execution of the configured-FCFS sort.
- Keep memory/token exhaustion separate from other scheduling-budget limits.
- The accepted subset establishes source decisions only, without throughput or starvation measurements.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/schedule_policy.py:214-262](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L214-L262)
- [python/sglang/srt/managers/schedule_policy.py:301-312](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L301-L312)
- [python/sglang/srt/managers/schedule_policy.py:834-853](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L834-L853)
- [python/sglang/srt/managers/schedule_policy.py:1544-1558](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L1544-L1558)
- [python/sglang/srt/managers/schedule_policy.py:1581-1611](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/schedule_policy.py#L1581-L1611)
