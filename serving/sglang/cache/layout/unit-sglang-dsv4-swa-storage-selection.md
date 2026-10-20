---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:740d63f50a203ca5d263a77d2aef529f34f3c53f8bc6c20042fb1d81cbec6bae
domain_path: serving/sglang/cache/layout
evidence_strength: single_observation
id: unit-sglang-dsv4-swa-storage-selection
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-dsv4-swa-storage-selection-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1131-L1245
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1246-L1261
  source_digest: ac1dd73c13851da6e19e72e205ca8b2cc2921b118f129c01bbb2fb0c1d49d98d
  source_locator: docs/evidence/unit-sglang-dsv4-swa-storage-selection-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: DeepSeek V4 SWA storage branch selection
triggers:
  error_signatures: []
  keywords:
  - sglang
  - deepseek
  - v4
  - swa
  - storage
  - branch
  - selection
---
The DeepSeekV4TokenToKVPool SWA assignment is an ordered if/elif chain. Encoder replay on a non-draft worker creates a RequestWindow, with capacity based on the sliding window and draft-token allowance and workspace sized against chunk, graph and per-request requirements. Otherwise uses_draft_swa_ring selects a per-request ring. Otherwise _unified_kv creates DeepSeekV4UnifiedKVPool after asserting KVLayout.V4. The final branch creates the paged pool and switches its class to DeepSeekV4UniformFP8KVPool only when uniform_fp8 is set, asserting float8_e4m3fn in that case.

The request-window and draft-ring branches do not carry those layout assertions. After the chain, a non-HIP DSpark draft with a non-null swa_kv_pool separately allocates swa_valid_from. That row state can accompany ring or paged SWA; it is not a fifth mutually exclusive storage selection. The code also calculates needs_paged_swa_allocator separately, because a speculative draft without its own ring can still depend on the target's paged mapping.

Scope and caveats:

- Keep the four storage assignments separate from the subsequent draft row-state allocation.
- The V4 and fp8 assertions belong to specific branches; do not generalize them to RequestWindow or ring storage.

Omitted scope:

- Per-ratio compression geometry, PP layer mapping, indexer pools and measured capacity effects are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:1131-1245](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1131-L1245)
- [python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:1246-1261](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1246-L1261)
