---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:31c7037a4fc9cd59ea9e32b7a63be98c549beb443ddfbb4cf1d7bf48c7899699
domain_path: serving/sglang/admission
evidence_strength: single_observation
id: unit-sglang-priority-admission-test-oracles
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-priority-admission-test-oracles-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L38-L44
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L57-L134
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L395-L416
  source_digest: 687ab56f92e4247d09f31936c01e09e78c538aedbe80a08b933f59cd91b71b07
  source_locator: docs/evidence/unit-sglang-priority-admission-test-oracles-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Priority tests distinguish queued eviction from incoming rejection
triggers:
  error_signatures: []
  keywords:
  - sglang
  - priority
  - tests
  - distinguish
  - queued
  - eviction
  - from
  - incoming
  - rejection
---
The priority-scheduling tests configure one running slot, three queued slots and priority scheduling. In the ordering case, a priority-zero holder starts first, followed by two priority-one arrivals and a priority-two arrival. The test checks successful responses and finish order: holder, priority two, then the priority-one requests in arrival order. The helper checks response status and, for failures, the error object and exact message.

Queue pressure has two separately asserted outcomes. Behind a priority-one holder, queued priorities two, three and four are later displaced by priorities five, six and seven: the displaced requests expect HTTP 503 with the higher-priority-abort message, while new requests succeed and finish seven, six, five. Behind a priority-seven holder with six, five and four queued, incoming three, two and one instead expect HTTP 503 with the queue-full message. These are explicit test oracles for selected workloads; they do not expose the scheduler's comparator, prove a general victim-selection rule, or establish that this test suite passed on the current environment.

Reviewed limits and corrections:

- Describe the asserted scenarios and finish order without promoting them to a full comparator or victim-selection implementation.
- The response helper uses zip; this subset does not claim exhaustive cardinality validation for other cases.

Omitted scope:

- Preemption thresholds, routing keys, multiple-running cardinality mismatch, overlap/retraction, async D2H, load snapshots and mixed-prefill tests are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [test/registered/scheduler/test_priority_scheduling.py:38-44](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L38-L44)
- [test/registered/scheduler/test_priority_scheduling.py:57-134](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L57-L134)
- [test/registered/scheduler/test_priority_scheduling.py:395-416](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/test/registered/scheduler/test_priority_scheduling.py#L395-L416)
