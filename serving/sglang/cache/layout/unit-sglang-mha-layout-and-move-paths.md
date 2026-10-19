---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:eb42ca4fa0a2e0fea9e1015360790542b77c4ab2eb198234d4d7ccd2c2b190e0
domain_path: serving/sglang/cache/layout
evidence_strength: single_observation
id: unit-sglang-mha-layout-and-move-paths
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-mha-layout-and-move-paths-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2268-L2275
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2276-L2342
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2591-L2611
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L3430-L3469
  source_digest: f941b72fb5e7065071fe516cf8f5e704b5c842698812852f2e3ff0fa5ef22e8b
  source_locator: docs/evidence/unit-sglang-mha-layout-and-move-paths-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: MHA KV layout selection and move paths
triggers:
  error_signatures: []
  keywords:
  - sglang
  - mha
  - kv
  - layout
  - selection
  - and
  - move
  - paths
---
MHATokenToKVPool treats an explicit kv_cache_layout as a label and disables the previously selected HND route for that case; identity-changing layouts are assigned to subclasses by the comment. Otherwise HND requires size plus page_size to be page-divisible. The optional AITER vectorized layout requires page size and both head dimensions to be divisible by its element-size-derived vector width. Non-HND buffers include a padded page; the source comment assigns it to dummy padded-token writes. Tiled-copy configuration is created only when KV-copy support is enabled and HND is off.

move_kv_cache returns immediately for a zero-layer pool, calls the optional out-of-bounds detection helper, and handles HND by converting source and target slots into page/offset coordinates for assignment. The other path selects configured native movement or the tiled-copy implementation. The latter returns for empty locations and otherwise requires _kv_copy_config. These branches explain which movement implementation is reached; merely calling maybe_detect_oob does not establish that bounds checks are always active.

Reviewed limits and corrections:

- Name the configured native branch without inferring an environment-setting origin from this method alone.
- Preserve the optional nature of the out-of-bounds helper.

Omitted scope:

- NoOp and PageMajor subclasses, offload paths and measured transfer throughput are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/memory_pool.py:2268-2275](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2268-L2275)
- [python/sglang/srt/mem_cache/memory_pool.py:2276-2342](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2276-L2342)
- [python/sglang/srt/mem_cache/memory_pool.py:2591-2611](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L2591-L2611)
- [python/sglang/srt/mem_cache/memory_pool.py:3430-3469](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/memory_pool.py#L3430-L3469)
