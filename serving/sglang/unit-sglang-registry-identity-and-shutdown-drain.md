---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:b45a32ec68dd8e03f633753a960262e05c2ce339100e2bc1d5b00d8440802792
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-registry-identity-and-shutdown-drain
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-registry-identity-and-shutdown-drain-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3520-L3583
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3589-L3655
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3810-L3886
  source_digest: d1e7bd9b4444cdfd7c87acf979547b354a3b49072009bf0d8bc139c32a1d5633
  source_locator: docs/evidence/unit-sglang-registry-identity-and-shutdown-drain-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Tokenizer registry identity guards protect newer requests during failure cleanup
triggers:
  error_signatures: []
  keywords:
  - sglang
  - tokenizer
  - registry
  - identity
  - guards
  - protect
  - newer
  - requests
  - during
  - failure
  - cleanup
---
TokenizerManager normalizes incoming request IDs, rejects duplicates against both the live registry and the new batch, creates states, then installs them together. Failure cleanup checks object identity: a different state under the same rid is left untouched, and that early continue also skips encoder_dispatch_ready removal and notification. A matching dispatched state is aborted and retained for the scheduler response; an undelivered state is deleted locally and its LoRA release helper is called. Abort echoes ignore health-check requests and missing rids, then mark a found state finished, build terminal output, delete it, release LoRA and notify the event. Graceful shutdown first waits for the registry to drain in five-second intervals without a request-drain deadline unless unhealthy or forced shutdown triggers the force-exit handler. Only after that loop does it send ShutdownReq and apply the scheduler-exit deadline; scheduler polling uses blocking time.sleep inside the async coroutine.

Reviewed limits and corrections:

- Registry presence does not by itself mean active model execution.
- The identity guard does not always drain encoder_dispatch_ready; the mismatch branch deliberately skips that later action.
- Replace whole admitted-unit citations with exact function ranges.
- The accepted subset makes no claim about scoring-layer cancellation or unseen base-generator cleanup.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/tokenizer_manager.py:3520-3583](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3520-L3583)
- [python/sglang/srt/managers/tokenizer_manager.py:3589-3655](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3589-L3655)
- [python/sglang/srt/managers/tokenizer_manager.py:3810-3886](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tokenizer_manager.py#L3810-L3886)
