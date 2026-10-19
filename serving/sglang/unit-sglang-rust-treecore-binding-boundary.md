---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:21b9bffce5bb5d9643d1c217c705a72606ca933377583b83b6d24b26cb51eb0b
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-rust-treecore-binding-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-rust-treecore-binding-boundary-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L371-L468
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L499-L503
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L625-L681
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L749-L758
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L787-L821
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L847-L848
  source_digest: 8aa27b5fadcd9dd113383ab09d2fe754012b58ea08c7b357fd309697ff720c80
  source_locator: docs/evidence/unit-sglang-rust-treecore-binding-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Rust TreeCore binding boundaries and Python orchestration
triggers:
  error_signatures: []
  keywords:
  - sglang
  - rust
  - treecore
  - binding
  - boundaries
  - and
  - python
  - orchestration
---
RustUnifiedTreeCore rejects session radix caching, component-registry overrides and components outside FULL, SWA and MAMBA at construction. SWA requires a positive sliding window. The bigram binding is selected only for an EAGLE configuration without MAMBA; a bare cuda device resolves to the current device. The native initializer starts with write-back and HiCache disabled, but these are initial values rather than permanent capability exclusions.

The adapter exposes stepwise eviction: evict_device_next_node asks the native core to choose a candidate and returns a Python result containing node_id, progress and backup metadata for the surrounding loop. evict_device_leaf checks that the requested write-back mode agrees with the current configuration. match_prefix delegates its tree matching through one binding call, while insertion uses begin, optional resume and end calls to coordinate actions around the native operation. These Python entrypoints establish the ownership boundary and protocol shape; they do not reveal or verify the native Rust selection algorithm, memory safety or performance.

Reviewed limits and corrections:

- Initial enable_hicache=False is mutable; set_hicache changes it later.
- Describe native selection versus Python orchestration without treating the adapter as evidence for the Rust algorithm.

Omitted scope:

- T-LRU arithmetic, storage integration and measured Rust speedups are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:371-468](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L371-L468)
- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:499-503](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L499-L503)
- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:625-681](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L625-L681)
- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:749-758](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L749-L758)
- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:787-821](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L787-L821)
- [python/sglang/srt/mem_cache/rust_tree_core/adapter.py:847-848](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/rust_tree_core/adapter.py#L847-L848)
