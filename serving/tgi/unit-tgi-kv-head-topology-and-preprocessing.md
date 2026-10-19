---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:a2aaf5f7f3a1d6e91061f2a051c93375c91601ce1b4dbd6fe3102ea780cff652
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-kv-head-topology-and-preprocessing
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-kv-head-topology-and-preprocessing-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L150-L175
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L249-L296
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L442-L463
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py#L244-L295
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py#L603-L610
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py#L72-L103
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py#L151-L175
  source_digest: 480f0e3e545ca257a933f26bcbbe0709ee18a7c70d85dadd9de54112dc5fb5d6
  source_locator: docs/evidence/unit-tgi-kv-head-topology-and-preprocessing-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: KV-head topology and preprocessing configuration need executable-path checks
triggers:
  error_signatures: []
  keywords:
  - tgi
  - kv-head
  - topology
  - and
  - preprocessing
  - configuration
  - need
  - executable-path
  - checks
---
RW attention uses separate MQA/MHA and grouped-head mappings; its parallel-attention layer combines attention and MLP outputs before all-reduce. StarCoder2 applies prefill_cache_indices to both stored K/V rows and outer slots, while its KV-head count uses integer division by world size. Gemma3's processor constructs its own fixed image processor and expands image placeholders, but its Pan-and-Scan crop-description assignment does not update the text list later tokenized.

Reviewed limits and corrections:

- StarCoder2 sequence-length clamping at603-610 is guarded by prefill_cache_indices is None and max_past is not None, not explicitly by decode mode. The comment alone does not prove prefill can never reach it.
- StarCoder2 checks query-head divisibility but floor-divides KV heads; downstream accepted configurations and actual device failures were not tested.
- RWLarge checks group/world-size compatibility; head/group division at268 has no local divisibility check.
- Gemma3 processor rebinds prompt at166-170 without updating text before173-175. Images with text=None also reach text[0] at126; text=[] bypasses that operand and can reach placeholder generation.
- A non-dict vision_config is replaced by a new SiglipVisionConfig at293-300. Do not infer effective settings from docstring examples.
- SantaCoder's discarded initial bias slice at118 is overwritten at121; classify it as redundant source work without a measured performance claim.

Evidence scope: All 19 proposed reference ranges read from pinned Git; compact mechanism excludes uninspected docstring-default and external-kernel claims.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/flash_rw_modeling.py:150-175](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L150-L175)
- [server/text_generation_server/models/custom_modeling/flash_rw_modeling.py:249-296](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L249-L296)
- [server/text_generation_server/models/custom_modeling/flash_rw_modeling.py:442-463](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_rw_modeling.py#L442-L463)
- [server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py:244-295](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py#L244-L295)
- [server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py:603-610](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_starcoder2_modeling.py#L603-L610)
- [server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py:72-103](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py#L72-L103)
- [server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py:151-175](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/gemma3/processing_gemma3.py#L151-L175)
