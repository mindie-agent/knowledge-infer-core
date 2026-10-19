---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:5321a18c7b670e526af9f5ed3bc882d2520a34c5dfd029c5dc417a98c300d84b
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-dsa-index-and-tail-offload
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-dsa-index-and-tail-offload-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5331-L5379
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5398-L5464
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5530-L5550
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5627-L5659
  source_digest: d1cb403182f2fe27366bc2efaa2223436aec59d4673480ad24bd74c9582cb3de
  source_locator: docs/evidence/unit-sglang-dsa-index-and-tail-offload-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: DSA index and compression state must accompany KV offload
triggers:
  error_signatures: []
  keywords:
  - sglang
  - dsa
  - index
  - and
  - compression
  - state
  - must
  - accompany
  - kv
  - offload
---
The DSA pool derives index_page_size from attention page_size divided by index_kpool, with an additional DCP division in the replicated case. It requires index_head_dim 128 and checks platform-specific physical index-page sizes. A supplied skip_topk_layers list must already have layer_num entries; only the missing-list default becomes an all-false list. Skipped positions remain in the layer structure, and an all-skipped index cache is omitted from host declarations.

Compression tails are allocated only when index_kpool is greater than one and compression is enabled, with max_running_requests required. Key and score tails use bf16, an extra request row and width index_kpool plus tail_extra_slots; skipped layers use zero-row placeholders. A nonempty tail write indexes modulo that actual width. A zero remainder performs no write, rather than explicitly clearing the buffer. get_cpu_copy and load_cpu_copy carry index cache and, when present, request tails alongside parent KV state. This coupling preserves the state the source identifies as needed across retraction; it is not an executed retraction-correctness result.

Reviewed limits and corrections:

- A supplied skip list is length-checked, not automatically expanded.
- The no-remainder tail branch does not clear storage; the no-carryover statement is a comment-level contract.

Omitted scope:

- Detailed index quantization kernels, host-transfer implementations and platform acceptance are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/memory_pool.py:5331-5379](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5331-L5379)
- [python/sglang/srt/mem_cache/memory_pool.py:5398-5464](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5398-L5464)
- [python/sglang/srt/mem_cache/memory_pool.py:5530-5550](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5530-L5550)
- [python/sglang/srt/mem_cache/memory_pool.py:5627-5659](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L5627-L5659)
