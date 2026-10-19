---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:5182326dd23d7c36db3eb13bcb7455328c5c8e5bc3d2797b71047be33239f9e0
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-token-major-strided-views
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-token-major-strided-views-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L20-L113
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L116-L153
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L172-L258
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/paged_view.py#L11-L43
  source_digest: e5106e1e93f8c3470449fc80c90cffb92c967a5b340994a4131a4eb7df34d396
  source_locator: docs/evidence/unit-sglang-token-major-strided-views-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Token-major byte layouts create aliased strided views
triggers:
  error_signatures: []
  keywords:
  - sglang
  - token-major
  - byte
  - layouts
  - create
  - aliased
  - strided
  - views
---
DenseEntryLayout describes fixed byte spans inside each token entry. Validation requires entry_bytes to be 32-byte aligned; each part's offset, layer stride and row byte count must be 16-byte aligned. Every layer span must fit within the entry, and sorted spans must not overlap. build_dense_views additionally checks item-size divisibility and total raw-buffer bounds, then creates per-layer as_strided views whose slot stride is entry_bytes divided by the element size. These views alias the supplied raw storage rather than copying each layer into a separate allocation.

The Mamba view builder has different checks: convolution views check entry and anchor divisibility but integer-divide the initial convolution offset without a separate assertion for it; temporal views check their combined offset. Do not generalize dense-layout validation to every Mamba offset. paged_view groups slots using page stride equal to slot_stride times page_size and retains the slot stride inside a page, including page_size one. The HND helper permutes axes after that regrouping.

Reviewed limits and corrections:

- Include the actual alignment constants with the validator citation.
- Page stride is slot_stride multiplied by page_size, not always the original slot stride.
- The empty layout/__init__.py source unit contributes no line-1 mechanism and is omitted.

Omitted scope:

- Dense draft-region sizing and runtime aliasing or device-transfer tests are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/layout/token_major.py:20-113](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L20-L113)
- [python/sglang/srt/mem_cache/layout/token_major.py:116-153](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L116-L153)
- [python/sglang/srt/mem_cache/layout/token_major.py:172-258](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/token_major.py#L172-L258)
- [python/sglang/srt/mem_cache/layout/paged_view.py:11-43](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/layout/paged_view.py#L11-L43)
