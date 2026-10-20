---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:1fedab58b21ac24e9826f56481c0077fdf9156100a4890347c48dd82dd12e726
domain_path: serving/tgi/runtime
evidence_strength: single_observation
id: unit-tgi-radix-release-and-eviction
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-radix-release-and-eviction-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L88-L121
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L157-L207
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L368-L405
  source_digest: 71cf4877b249ad77c31e83452853204244acb55e9dc5d11db44a51e51aba6a7a
  source_locator: docs/evidence/unit-tgi-radix-release-and-eviction-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Radix prefix release returns reusable blocks without implying full eviction
triggers:
  error_signatures: []
  keywords:
  - tgi
  - radix
  - prefix
  - release
  - returns
  - reusable
  - blocks
  - without
  - implying
  - full
  - eviction
---
TGI's radix allocator increments the matched prefix reference before reclaiming suffix blocks and undoes that reference if allocation fails. Freeing an allocation releases its reference, inserts aligned prefill content into the trie, returns duplicate-prefix blocks and releases non-prefill suffix blocks. Trie eviction selects unreferenced leaves and truncates a leaf when only part is needed; it can return fewer blocks than requested when reclaimable leaves are exhausted.

Scope and caveats:

- Remove the false whole-node eviction overshoot claim: a whole leaf is removed only when its size is at most the remaining need; otherwise suffix truncation is exact.
- The subtraction underflow condition is prefix_len > tokens, not any non-matching prefill.
- hash(slice) special-cases a single-token slice, not an arbitrary one-block slice.
- Hash-collision concerns remain source-level possibilities; no collision or runtime failure was reproduced.
- Use narrow method ranges instead of the entire 1041-line file.

Evidence scope: All corrected ranges plus allocate failure rollback and eviction loop read from pinned Git.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [backends/v3/src/radix.rs:88-121](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L88-L121)
- [backends/v3/src/radix.rs:157-207](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L157-L207)
- [backends/v3/src/radix.rs:368-405](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L368-L405)
