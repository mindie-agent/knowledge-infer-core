---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:7098cb5ce2a39cb218876137015b2dffaed0cb46229effe841a747013117cab6
domain_path: serving/sglang/admission
evidence_strength: single_observation
id: unit-sglang-generate-input-normalization
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-generate-input-normalization-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L444-L515
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L907-L947
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L959-L1064
  source_digest: 1bb3e7af58b6e53d09fa01384adf086819b4ee480af7bc15d58a27d905af5ac3
  source_locator: docs/evidence/unit-sglang-generate-input-normalization-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: GenerateReqInput normalization is not an exactly-one input-form contract
triggers:
  error_signatures: []
  keywords:
  - sglang
  - generatereqinput
  - normalization
  - is
  - not
  - an
  - exactly-one
  - input-form
  - contract
---
GenerateReqInput normalization at the pinned revision rejects an all-empty input and a simultaneous text/input_ids/input_embeds triple. The actual predicates still allow pairs. In the text branch it clears input_embeds but does not clear input_ids, so the excerpts do not support an exactly-one-form invariant. Bootstrap normalization expands a scalar bootstrap_room across a batch with per-item offsets in the shown branch. __getitem__ constructs and caches a per-request object by indexing selected normalized fields; this is not a fresh general validation pass and it does not prove every field has been independently shape-checked.

Scope and caveats:

- Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/io_struct.py:444-515](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L444-L515)
- [python/sglang/srt/managers/io_struct.py:907-947](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L907-L947)
- [python/sglang/srt/managers/io_struct.py:959-1064](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/io_struct.py#L959-L1064)
