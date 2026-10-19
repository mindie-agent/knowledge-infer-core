---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:e88f74475567f3b74f4bce57103b769fce4147a62a6ad78cffec1d8ca6f84cdf
domain_path: serving/sglang/cache/transfers
evidence_strength: single_observation
id: unit-sglang-prefetch-completion-and-hybrid-gate
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-prefetch-completion-and-hybrid-gate-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L2360-L2473
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L2474-L2582
  source_digest: b7d64d26617db3fdc94bc73fa995bf87374f806385160450f309e25ca32058fb
  source_locator: docs/evidence/unit-sglang-prefetch-completion-and-hybrid-gate-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Storage-prefetch completion ownership and hybrid acceptance
triggers:
  error_signatures: []
  keywords:
  - sglang
  - storage-prefetch
  - completion
  - ownership
  - and
  - hybrid
  - acceptance
---
_handle_prefetch_result owns the completed host span and, when acknowledged as done, the accompanying component transfers. A rejected hybrid result is already retired by _check_hybrid_prefetch_result. The handler resolves the gap between allocated and completed tokens; buffer mode then stages a bounce result for later admission. The normal cache path inserts the completed span, applies actions, commits PREFETCH transfers, frees the already-resident prefix and records completed_tokens minus prefix_len as loaded from storage. It resolves prefix_len immediately; the usable tail remains pending until admission proves L2-to-L1 load-back. A dropped insertion releases the completed span and transfers and records zero loaded tokens.

The hybrid gate removes KV-derived transfers before checking completeness. It rejects an incomplete result only when non-KV pool transfers remain: completed_tokens must match the expected hash-page span, and each transfer needs keys and a matching hit count. With no such transfer, partial completion can pass this gate. The comment describes an upstream cross-rank minimum; the helper itself performs no reduction.

Reviewed limits and corrections:

- Hybrid completeness is conditional on non-KV transfers, not a blanket all-or-nothing policy for every layout.
- Only the newly loaded tail remains unresolved after the normal insertion; the existing prefix is resolved locally.
- Do not attribute the comment-described cross-rank reduction to this helper body.

Omitted scope:

- Abort routing, acknowledgement pump ordering and other accounting helpers are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/unified_radix_cache.py:2360-2473](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L2360-L2473)
- [python/sglang/srt/mem_cache/unified_radix_cache.py:2474-2582](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L2474-L2582)
