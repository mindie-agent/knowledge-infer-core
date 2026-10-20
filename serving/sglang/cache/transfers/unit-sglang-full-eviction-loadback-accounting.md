---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:73c59e868398e988b9b5b6d0fc6aa8102fb0ff7690b874920a2d933199a96ab7
domain_path: serving/sglang/cache/transfers
evidence_strength: single_observation
id: unit-sglang-full-eviction-loadback-accounting
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-full-eviction-loadback-accounting-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/components/full.py#L169-L196
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/components/full.py#L372-L463
  source_digest: b2e734645bc65823be91b5a74aa87e76070688d3db4e2bcb01853e7c15531409
  source_locator: docs/evidence/unit-sglang-full-eviction-loadback-accounting-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Full cache eviction defers device clearing before load-back accounting
triggers:
  error_signatures: []
  keywords:
  - sglang
  - full
  - cache
  - eviction
  - defers
  - device
  - clearing
  - before
  - load-back
  - accounting
---
FullComponent.evict_component queues a device tensor for freeing and subtracts reclaimable_tokens from the evictable ledger, but deliberately leaves its value reference in place. The comment assigns clearing to _cascade_evict because SWA's free operation still needs Full.value. Host eviction differs: it queues host_value and immediately clears that field. This is a component-ordering dependency, not evidence that every tree field is cleared at the same time.

For LOAD_BACK, build_hicache_transfers walks from the matching node toward its parents while nodes are device-evicted, requires each host value, and reverses the collected values and node IDs into root-first order. Its comment relies on Full device eviction being leaf-only. commit_hicache_transfer then slices the returned device indices by each node's host length, installs copied values, and credits either protected or evictable size according to lock_ref. The source defines ownership and ledger transitions; asynchronous completion and actual reclamation still depend on the surrounding controller.

Scope and caveats:

- Distinguish deferred device clearing from immediate host clearing.
- The parentward load-back walk carries a stated leaf-eviction precondition; do not generalize it to arbitrary component eviction.

Omitted scope:

- Internal cascade-priority values, base-class design comments, controller drain ordering and measured reclamation are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/unified_cache/components/full.py:169-196](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/components/full.py#L169-L196)
- [python/sglang/srt/mem_cache/unified_cache/components/full.py:372-463](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/unified_cache/components/full.py#L372-L463)
