---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:827af4185fc9f47f819a0517d121262c94cfe2845cbd76b67da3aa7a7aed5982
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-paged-allocation-and-chunk-admission
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-paged-allocation-and-chunk-admission-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L6-L25
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L57-L84
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/queue.rs#L287-L374
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/queue.rs#L491-L503
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/backend.rs#L148-L205
  source_digest: 93b49ed87ca1d47a47fb882eb2b444a03cc595944f6f2ee1d5138ea4e2fd006b
  source_locator: docs/evidence/unit-tgi-paged-allocation-and-chunk-admission-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Paged allocation ownership and chunking have separate admission contracts
triggers:
  error_signatures: []
  keywords:
  - tgi
  - paged
  - allocation
  - ownership
  - and
  - chunking
  - have
  - separate
  - admission
  - contracts
---
TGI v3 uses a serialized allocator for non-padding batches and attaches BlockAllocation ownership to each admitted entry. Dropping an owned allocation queues a free operation; it does not synchronously prove reclamation. Prefix-cached requests leave at least one token to compute. Chunking allows partial postfix admission and removes the min/max size limit from later concatenations, whereas the initial batch still receives max_batch_size.

Reviewed limits and corrections:

- Zero-budget rejection belongs to Queue::next_batch:93-95; block rounding belongs to State::next_batch:264-265.
- Include backend.rs:160 when citing waiting_tokens initialization.
- Equal cardinality does not itself compare request IDs; the candidate already described the shortcut, so remove the reviewer claim that it was omitted.
- Clone/Drop alone does not establish a duplicate-free execution path; observed call sites clone blocks/slots and move the allocation.
- On RPC failure, clear_cache is attempted with its error discarded; do not describe successful cleanup.
- Use block_allocator.rs:165-168 for the HPU extra slot, separate from 171-214 window and slot logic.
- The no-allocator estimate is updated before rejecting a request; describe estimate overcount as source-derived, not a measured memory or scheduling impact.

Evidence scope: All non-client corrected ranges plus symbol corrections read; shared client aggregation was independently read in records 0 and 1.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [backends/v3/src/block_allocator.rs:6-25](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L6-L25)
- [backends/v3/src/block_allocator.rs:57-84](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L57-L84)
- [backends/v3/src/queue.rs:287-374](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/queue.rs#L287-L374)
- [backends/v3/src/queue.rs:491-503](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/queue.rs#L491-L503)
- [backends/v3/src/backend.rs:148-205](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/backend.rs#L148-L205)
