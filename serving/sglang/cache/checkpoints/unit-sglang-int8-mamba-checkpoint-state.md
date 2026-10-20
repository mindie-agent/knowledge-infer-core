---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:81ad76aaa5dd6dfa8922deaef2a72a60f82de2b1acc4121d5d87c714f3fc78c6
domain_path: serving/sglang/cache/checkpoints
evidence_strength: single_observation
id: unit-sglang-int8-mamba-checkpoint-state
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-int8-mamba-checkpoint-state-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L74-L139
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L143-L175
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L203-L226
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L239-L260
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L318-L330
  source_digest: d6bbf8217d718462262673da0ec8d9796f6d165dbf9ecb70a8fe35712c215a62
  source_locator: docs/evidence/unit-sglang-int8-mamba-checkpoint-state-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Int8 Mamba checkpoints retain rounded scales and native convolution state
triggers:
  error_signatures: []
  keywords:
  - sglang
  - int8
  - mamba
  - checkpoints
  - retain
  - rounded
  - scales
  - and
  - native
  - convolution
  - state
---
Int8CheckpointStore stores temporal state as int8 data with one scale per head and key channel. Quantize computes amax across the value dimension in float32, floors it at 1e-8, divides by 127 and rounds that scale to the source-state dtype before dividing the float32 state by it. The result is rounded, clamped to -127 through 127 and cast to int8. Dequantize multiplies data and scale in float32 before converting to the requested output dtype. The early scale conversion makes quantization use the same rounded scale that dequantization receives.

MambaCheckpointPool reserves an extra slot, retains convolution buffers in their native dtype and owns a separate checkpoint allocator. store_from_active quantizes temporal state and copies convolution state into the selected checkpoint; load_to_active restores both. clear resets slot allocation and deliberately leaves stored bytes untouched. The optional initialization path returns no pool unless enabled and chooses configured capacity or twice mamba_size. Those capacity and representation choices are source behavior, not evidence of a measured memory saving or numerical-accuracy bound.

Scope and caveats:

- The scale is rounded before division, not merely when finally stored.
- Checkpoint clearing resets allocation bookkeeping; it is not zeroization of qdata.
- Convolution state remains native precision and must be included in any later memory or restore measurement.

Omitted scope:

- Device-memory preflight, budget estimates and hardware quality/performance claims are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:74-139](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L74-L139)
- [python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:143-175](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L143-L175)
- [python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:203-226](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L203-L226)
- [python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:239-260](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L239-L260)
- [python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:318-330](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L318-L330)
