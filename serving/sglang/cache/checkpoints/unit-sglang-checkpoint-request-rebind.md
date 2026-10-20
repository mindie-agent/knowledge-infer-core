---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:b24dda2aff1e74ed89c8e5125894dfdd37677d4cf6ed3761fc9b60d26467b67b
domain_path: serving/sglang/cache/checkpoints
evidence_strength: single_observation
id: unit-sglang-checkpoint-request-rebind
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-checkpoint-request-rebind-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1126-L1163
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1164-L1299
  source_digest: 4902e9cd3b5c18dce3d2252ea4476eedac2f1a6f9a2d09c2e0ced5f711d5057e
  source_locator: docs/evidence/unit-sglang-checkpoint-request-rebind-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Checkpoint insertion and request ownership in UnifiedRadixCache
triggers:
  error_signatures: []
  keywords:
  - sglang
  - checkpoint
  - insertion
  - and
  - request
  - ownership
  - in
  - unifiedradixcache
---
checkpoint_into_tree first bounds the token prefix through component preparation and floor_cache_len, then page-aligns the key before inserting copied int64 KV indices. A disabled cache only performs component cleanup for a finished request; a nonpositive effective length also cleans up and returns. If insert reports rotation_tail_declined, this method cleans up and returns before rebinding the request or changing its tree lock. The comment explains that retaining the request's own pages avoids mixing rotation runs.

The prompt-boundary insert is an additional operation when the tree has exactly one BASE component and the prompt key is a strict nonempty prefix. It does not replace the subsequent ownership update. The normal path obtains indices from result.last_device_node, writes the unprotected portion of req_to_token, unlocks the old request lock, takes a new lock, and updates cache_protected_len and last_node. An opt-in flag can omit MAMBA from that new lock; its safety depends on the admission/COW ordering described in the comment and is not independently established by this method.

Scope and caveats:

- Treat the prompt-boundary insertion as additive; the usual request rebind and lock migration still follow.
- The rotation-tail branch and MAMBA lock comment establish local behavior and a stated precondition, not a concurrency or device-level proof.

Omitted scope:

- Constructor and registry setup, eviction shortfalls, host backup, load-back and storage-prefetch protocols are outside this accepted subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/unified_radix_cache.py:1126-1163](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1126-L1163)
- [python/sglang/srt/mem_cache/unified_radix_cache.py:1164-1299](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_radix_cache.py#L1164-L1299)
