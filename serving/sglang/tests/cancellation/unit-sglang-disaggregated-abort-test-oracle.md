---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:b3b07aa2f4bba466cf862161f0cb6db27c78eb3ab68376c9790004b2f85258ea
domain_path: serving/sglang/tests/cancellation
evidence_strength: single_observation
id: unit-sglang-disaggregated-abort-test-oracle
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-disaggregated-abort-test-oracle-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L17-L27
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L37-L50
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L53-L113
  source_digest: f20238e1a928e4436bfe40ab238b658454f63073b5c64c8cd6257a8c93812d8d
  source_locator: docs/evidence/unit-sglang-disaggregated-abort-test-oracle-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Disaggregated abort test checks response shape and endpoint liveness
triggers:
  error_signatures: []
  keywords:
  - sglang
  - disaggregated
  - abort
  - test
  - checks
  - response
  - shape
  - and
  - endpoint
  - liveness
---
The chunked-prefill abort test launches with max-running-requests four and chunked-prefill-size 64. A worker sends a long-prompt generate request with max_new_tokens 4096 and ignore_eos. After one second, the test repeatedly posts the same request ID with abort_all false to both prefill and decode abort endpoints while the worker remains alive. It loops with 0.2-second sleeps, checking an eight-second deadline between iterations; the synchronous abort POSTs each have a ten-second timeout and can extend elapsed time. It does not inspect each abort POST's response status. It then joins the worker for up to 60 seconds and requires termination, no captured RequestException and an accepted abort result.

The response oracle accepts HTTP 200 only when finish_reason.type is abort, or HTTP 500/503 when the body contains abort case-insensitively. Finally, health endpoints on the load balancer, prefill role and decode role must return 200. The test therefore expresses an abort-response and liveness contract. It does not check KV or Mamba reclamation, a subsequent successful generation, or scheduler evidence that the abort actually arrived mid-chunk. The test source was inspected here; it was not executed as an acceptance run.

Reviewed limits and corrections:

- The test name does not establish the scheduler phase at which cancellation arrived.
- Health responses and an abort-shaped result do not prove resource reclamation or readiness for a subsequent generation.

Omitted scope:

- The other six test modules in the source record and any hardware or throughput result are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py:17-27](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L17-L27)
- [test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py:37-50](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L37-L50)
- [test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py:53-113](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py#L53-L113)
