---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:03a9104a79d14d4b5c69981d59b45141ac68e1877c16c5c356854d4649e87f1c
domain_path: serving/sglang/cache/layout
evidence_strength: single_observation
id: unit-sglang-kv-read-write-translation-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-kv-read-write-translation-boundary-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/kv_index_translator.py#L121-L224
  source_digest: f48fa5fb5bb8da2c067afcb1c6691637852e2aeb5584afbe32e674f63f51c447
  source_locator: docs/evidence/unit-sglang-kv-read-write-translation-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: KV read and write translations have different activation rules
triggers:
  error_signatures: []
  keywords:
  - sglang
  - kv
  - read
  - and
  - write
  - translations
  - have
  - different
  - activation
  - rules
---
KVIndexTranslator enables its main translation mode when a supported unified allocator owns the supplied KV pool or a fused draft's host allocator is exactly that allocator. It captures the allocator's full virtual-to-physical table and uses translate_write_loc for writes. Under attention DCP greater than one, read_v2p is left unset so the consumer can select the rank's widened read IDs at the production site. A UnifiedSWA allocator registers the window space only when it routes window layers.

Main translation mode being false does not imply identity writes. For a non-draft NPU DCP target, the constructor installs rank-local write localization. A static BaseSWAKVPool can additionally install full-to-SWA write mapping, composed with localization where needed; DeepSeekV4TokenToKVPool with request_window is excluded from that path. Reads remain in req_to_token in this mode and the comment assigns their mapping to the backends. Inspect the relevant IdSpace and read consumer separately when diagnosing address mismatches.

Scope and caveats:

- Do not equate is_translating=False with identity write locations.
- DCP read deferral and write localization are distinct contracts; this constructor does not prove downstream kernel handling.

Omitted scope:

- Draft-SWA configurator alternatives, lazy read-plan caching and FP4 backend capability claims are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/kv_index_translator.py:121-224](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/kv_index_translator.py#L121-L224)
