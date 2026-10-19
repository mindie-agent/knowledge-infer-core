---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:c628cfdeb7946ef832cdbd251009aa0e49bcbb1ec6b5aae7155fdf76f6d95b78
domain_path: serving/sglang
evidence_strength: single_observation
id: unit-sglang-hf3fs-batch-confirmation
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-hf3fs-batch-confirmation-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L388-L434
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L436-L514
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L523-L537
  source_digest: d8cda841c95013a57faef8adc0ee7a46862efade1231e24f4e2e3c32bdbdd1d1
  source_locator: docs/evidence/unit-sglang-hf3fs-batch-confirmation-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: HF3FS batch writes distinguish existing pages from newly confirmed IO
triggers:
  error_signatures: []
  keywords:
  - sglang
  - hf3fs
  - batch
  - writes
  - distinguish
  - existing
  - pages
  - from
  - newly
  - confirmed
  - io
---
HiCacheHF3FS._batch_set returns scalar True immediately when skip_backup is enabled, before reserving or writing pages. Otherwise it reserves page indices, skips already-written entries and entries marked -1, and compacts new writes together with their source tensors. Results start from the reservation's is_written flags. New writes count as successful only when the future returns bytes_per_page; successful keys and short-write pages are then passed to confirm_write for confirmation or release. A raised future exception occurs before that confirmation path, so the shown code does not guarantee cleanup for every failure. A pre-existing page can report success without any local write in this call.

The V1 read path compacts hit offsets but leaves destination values unfiltered, slicing both with dense hit positions. If a miss precedes a later hit, the shown indexing can pair that hit's offset with an earlier destination tensor; prefix-shaped hits hide this discrepancy. This is a source-level conditional finding, not a reproduced storage-corruption result. batch_exists reports the leading existing prefix, adjusted for doubled MHA keys.

Reviewed limits and corrections:

- skip_backup reports success without proving a local durable write.
- Read offsets are hit-compacted while destinations remain full; the issue is their pairing, not merely the returned batch_indices.
- Confirmation covers returned short writes but is bypassed by an uncaught future exception.

Omitted scope:

- HF3FS server behavior, fault injection and the other storage backends grouped in this source record are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py:388-434](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L388-L434)
- [python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py:436-514](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L436-L514)
- [python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py:523-537](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py#L523-L537)
