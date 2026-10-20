---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:563bb856409fcc86e57fb5ba7a532e4c6030ef04fb43f1585c3d9f6e3a0bbfc5
domain_path: serving/sglang/execution
evidence_strength: single_observation
id: unit-sglang-metrics-and-output-rank-gates
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-metrics-and-output-rank-gates-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L362-L412
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L1174-L1206
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/output_streamer.py#L421-L461
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py#L235-L245
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py#L315-L327
  source_digest: 7530583780a5592599350cea4f5c0650f14062154134bad7091069b61c8d80de
  source_locator: docs/evidence/unit-sglang-metrics-and-output-rank-gates-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Forward metrics and finished output retain their rank, timing and beam gates
triggers:
  error_signatures: []
  keywords:
  - sglang
  - forward
  - metrics
  - and
  - finished
  - output
  - retain
  - their
  - rank
  - timing
  - beam
  - gates
---
Forward-pass metrics initialization is conditional on observability enablement, attention-TP rank zero and the final PP rank. This initialization always installs or reuses a DeviceTimer and sets _fpm_uses_device_timer true. The emit method contains a monotonic fallback, but the shown normal initialization does not make that fallback reachable; a zero accumulated device time suppresses emission. Output-stream acceptance applies its beam gate before its finished-request branch, so only a finished beam leader passes that gate. Nonterminal streaming uses unsent-token thresholds and may suppress a stop-string prefix. Ordinary full-pool usage subtracts both available and evictable capacity; the shown hybrid-SWA statistics path additionally clamps full and SWA usage when HiSparse is enabled.

Scope and caveats:

- Do not describe the monotonic fallback as an observed normal FPM execution path.
- Finished requests are still subject to the earlier beam gate.
- HiSparse clamping in the hybrid-SWA statistics path is not a universal clamp on all pool-reporting functions.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/scheduler_components/metrics_reporter.py:362-412](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L362-L412)
- [python/sglang/srt/managers/scheduler_components/metrics_reporter.py:1174-1206](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L1174-L1206)
- [python/sglang/srt/managers/scheduler_components/output_streamer.py:421-461](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/output_streamer.py#L421-L461)
- [python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:235-245](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py#L235-L245)
- [python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:315-327](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py#L315-L327)
