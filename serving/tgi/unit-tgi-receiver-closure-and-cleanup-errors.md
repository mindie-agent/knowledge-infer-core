---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:b77dd0984217df703e9c0281352eebe1a670ad4f1608ddbaf90a40afafa30e01
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-receiver-closure-and-cleanup-errors
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-receiver-closure-and-cleanup-errors-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/backend.rs#L239-L356
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/backend.rs#L361-L394
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/queue.rs#L230-L274
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/queue.rs#L309-L346
  source_digest: af5d3427d097b0e1627a74400613c327a154771cd40a4a1d8e0b381b733f2a31
  source_locator: docs/evidence/unit-tgi-receiver-closure-and-cleanup-errors-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Receiver closure is observed at batch boundaries, and cleanup calls can still
  fail
triggers:
  error_signatures: []
  keywords:
  - tgi
  - receiver
  - closure
  - is
  - observed
  - at
  - batch
  - boundaries
  - and
  - cleanup
  - calls
  - can
  - still
  - fail
---
TGI v2 drops queued requests with closed response receivers and removes stopped entries after a prefill/decode result is dispatched. The current RPC is not explicitly preempted by those checks. filter_batch uses cardinality equality as a shortcut; otherwise it filters request IDs or clears the shard cache. In prefill/decode error handlers, cache cleanup is attempted and its Result is discarded before errors are sent to all affected entries.

Reviewed limits and corrections:

- Replace statements that error cleanup succeeds with 'attempts clear_cache'; backend.rs:271 and 315-318 discard cleanup errors.
- State that request-ID equality is an assumed invariant behind the size shortcut, not a demonstrated runtime bug.
- build_server(false) concerns generated bindings only; it does not establish absence of server code elsewhere.
- queue.rs contains unit-test source; no tests were executed in this review.
- Additional source-level caveat: queue.rs:240-274 mutates token totals before rejecting an over-budget entry; batch.max_tokens at 334-338 can include that rejected entry when earlier entries remain. No throughput impact was measured.

Evidence scope: Corrected runtime ranges read; test module presence checked, test suite not run.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [backends/v2/src/backend.rs:239-356](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/backend.rs#L239-L356)
- [backends/v2/src/backend.rs:361-394](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/backend.rs#L361-L394)
- [backends/v2/src/queue.rs:230-274](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/queue.rs#L230-L274)
- [backends/v2/src/queue.rs:309-346](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v2/src/queue.rs#L309-L346)
