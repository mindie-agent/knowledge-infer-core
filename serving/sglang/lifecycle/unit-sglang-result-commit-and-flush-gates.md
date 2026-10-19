---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:d6c4646166a7cf8dff87acfca0c3fa9dfcf5818e8d1f24e88d283095735899a8
domain_path: serving/sglang/lifecycle
evidence_strength: single_observation
id: unit-sglang-result-commit-and-flush-gates
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-result-commit-and-flush-gates-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L307-L328
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L379-L396
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L807-L825
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L1354-L1387
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22-L61
  source_digest: 20bdb4b12bf2a71fb7b7f498fbd3981d61814eada121e1cc196128ecab2d3f95
  source_locator: docs/evidence/unit-sglang-result-commit-and-flush-gates-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Scheduler result commit, KV release and flush completion use separate gates
triggers:
  error_signatures: []
  keywords:
  - sglang
  - scheduler
  - result
  - commit
  - kv
  - release
  - and
  - flush
  - completion
  - use
  - separate
  - gates
---
The result processor commits the shown prefill state only when a request is unfinished, not retracted and has no in-flight chunks; a separate condition controls the KV checkpoint. Chunk acceptance similarly excludes already finished requests. Abort handling chooses offload or direct release from its actual mask and configuration, while the later lookahead assertion is conditional on a Mamba mask being present. FlushWrapper rejects a second pending flush request, attempts immediate completion when no timeout was supplied or the scheduler is idle, and otherwise defers until idle or the deadline. A deferred return value is not by itself a successful cache flush.

Reviewed limits and corrections:

- Discard the original empty __init__.py line-1 citation because that file has no line 1.
- Do not generalize the conditional Mamba lookahead assertion to every request.
- Commit eligibility, release choice and user-visible flush completion are separate contracts; no live scheduler test was run.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/scheduler_components/batch_result_processor.py:307-328](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L307-L328)
- [python/sglang/srt/managers/scheduler_components/batch_result_processor.py:379-396](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L379-L396)
- [python/sglang/srt/managers/scheduler_components/batch_result_processor.py:807-825](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L807-L825)
- [python/sglang/srt/managers/scheduler_components/batch_result_processor.py:1354-1387](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L1354-L1387)
- [python/sglang/srt/managers/scheduler_components/flush_wrapper.py:22-61](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/flush_wrapper.py#L22-L61)
