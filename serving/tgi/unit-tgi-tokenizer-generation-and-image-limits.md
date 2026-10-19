---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:fe0943721f1a4e34c0fd842da05bfd0e074d4a3b4bd484bbe5387d2fa8e268a2
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-tokenizer-generation-and-image-limits
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-tokenizer-generation-and-image-limits-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L43-L88
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L145-L199
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L570-L637
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L812-L860
  source_digest: c2bd3242196f43b3434e4ab5756ea249d534f3a571026484c15be2795f91e751
  source_locator: docs/evidence/unit-tgi-tokenizer-generation-and-image-limits-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Tokenizer admission, generation reservation and image limits bound different
  resources
triggers:
  error_signatures: []
  keywords:
  - tgi
  - tokenizer
  - admission
  - generation
  - reservation
  - and
  - image
  - limits
  - bound
  - different
  - resources
---
TGI validation sends requests through unbounded channels to blocking tokenizer workers; the Python tokenizer uses one worker. prepare_input encodes the full rewritten input before the retained token suffix and accounting are truncated. The current generation pass reserves at most DEFAULT_GENERATION_LENGTH while retaining a separate total-new-token request. HTTP image loading bounds encoded response bytes; the data-URI branch has no max_image_fetch_size check in this function.

Reviewed limits and corrections:

- Do not claim the fetch byte cap is a decoded image-memory cap. Image-library default limits were not inspected.
- The zero-worker constructor path can spin, but v2 CLI main.rs:147-150 rejects zero validation_workers; do not report this as a demonstrated deployed configuration bug.
- Vertex join_all waits for all sibling futures before collecting the Result; a returned error does not make it cancel siblings.
- Vertex maps errors returned from generate_internal to 500, but this unit does not establish every upstream validation-error path.
- The caller's dropped reply does not cancel work already running in the blocking worker.

Evidence scope: All 22 corrected reference ranges read; library decoding defaults and production concurrency were not executed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [router/src/validation.rs:43-88](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L43-L88)
- [router/src/validation.rs:145-199](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L145-L199)
- [router/src/validation.rs:570-637](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L570-L637)
- [router/src/validation.rs:812-860](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/router/src/validation.rs#L812-L860)
