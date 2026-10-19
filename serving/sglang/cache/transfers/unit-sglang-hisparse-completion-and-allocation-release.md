---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:3fd06aea25000c1770eb168b594970d47cd3bfa746b4c02ed38fa9bfde38df82
domain_path: serving/sglang/cache/transfers
evidence_strength: single_observation
id: unit-sglang-hisparse-completion-and-allocation-release
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-hisparse-completion-and-allocation-release-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L157-L188
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L1063-L1090
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L1397-L1458
  source_digest: 0195f7917e8ae8d9701c42e16dbde4275eefba5bec3a1c1cc3860977d32fea4c
  source_locator: docs/evidence/unit-sglang-hisparse-completion-and-allocation-release-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: HiSparse completion releases the current allocation after waiting for its producers
triggers:
  error_signatures: []
  keywords:
  - sglang
  - hisparse
  - completion
  - releases
  - the
  - current
  - allocation
  - after
  - waiting
  - for
  - its
  - producers
---
At the pinned SGLang revision, speculative HiSparse swap-in is restricted by its explicit enablement, speculative algorithm and draft-window checks. Ready-prefix accounting waits for the request-specific backup event before publishing the backed-up prefix. request_finished first waits for pending swap-in and backup work, then frees the persistent and scratch device allocations currently owned by the request. It uses allocated length rather than sequence length for the owned device range, clears the full request-to-token mapping, frees host allocation and resets request bookkeeping. These operations establish an ordering and ownership contract in the source; they do not establish a measured device cleanup duration or prove that every sparse-cache path uses identical synchronization.

Reviewed limits and corrections:

- Retain waits for both swap-in and backup instead of treating a logical request finish as immediate safe reuse.
- Use the actual allocated range and current persistent/scratch union; do not substitute the live sequence length.
- Only the cited HiSparse subset is accepted; detokenizer, embedding and disaggregation helper claims remain outside this editorial decision.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/hisparse_coordinator.py:157-188](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L157-L188)
- [python/sglang/srt/managers/hisparse_coordinator.py:1063-1090](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L1063-L1090)
- [python/sglang/srt/managers/hisparse_coordinator.py:1397-1458](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/hisparse_coordinator.py#L1397-L1458)
