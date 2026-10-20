---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:1ffc6b3a1abe27fe2386ec32e62d577a59361cd72fdd779a2ebec9020bfb219c
domain_path: serving/tgi/models/attention
evidence_strength: single_observation
id: unit-tgi-expanded-deepseek-kv-storage
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-expanded-deepseek-kv-storage-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_deepseek_v2_modeling.py#L270-L354
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma2_modeling.py#L448-L451
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py#L475-L485
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py#L251-L295
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gpt2_modeling.py#L450-L452
  source_digest: 2e2d0d78d1c1a3545022645466272d3f7507923041e3d5a30180dd21f4b2c937
  source_locator: docs/evidence/unit-tgi-expanded-deepseek-kv-storage-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Compressed attention projections do not imply a compressed runtime KV cache
triggers:
  error_signatures: []
  keywords:
  - tgi
  - compressed
  - attention
  - projections
  - do
  - not
  - imply
  - a
  - runtime
  - kv
  - cache
---
The pinned DeepSeek V2 implementation expands compressed KV through kv_b_proj, combines non-RoPE and rotated key components, pads Q/K/V to a common head size, and stores the expanded keys and values. Its attention then separates prefill and paged decode. Gemma2 selects sliding layers by parity; Gemma3 uses sliding_window_pattern and per-layer rotary bases, with explicit masks consumed by its non-flashinfer SDPA prefill branch. GPT-2 instead adds learned position embeddings.

Scope and caveats:

- DeepSeek V2 padding fields:171-173; rotary assembly:305-311; expanded store:313-330; MoE reduction:472-481. Config rejection ends at146.
- DeepSeek V3 scoring/correction-bias arguments occur at455-456, outside the proposed ending454.
- Gemma2 RMSNorm multiplication/downcast ends126; store starts256. Gemma3 sliding selection is475-485, not486-492; per-layer cos/sin is516-519.
- Gemma3 projector constructor/forward is611-651; image-mask OR is741-751, not758-768.
- Source comment and symbol names do not establish latent-cache memory savings, TP efficiency or speedup. Claims here describe this implementation only.
- Gemma v1 scaling at426-435 and GPT-2 learned-position addition at450-452 are source facts; no numerical model equivalence test was run.

Evidence scope: All 22 proposed references read and every truncated/misaligned substantive operation above recovered from pinned Git. Compact accepted mechanism omits unreviewed broad wrapper and ROCm-path generalizations.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/flash_deepseek_v2_modeling.py:270-354](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_deepseek_v2_modeling.py#L270-L354)
- [server/text_generation_server/models/custom_modeling/flash_gemma2_modeling.py:448-451](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma2_modeling.py#L448-L451)
- [server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py:475-485](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py#L475-L485)
- [server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py:251-295](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gemma3_modeling.py#L251-L295)
- [server/text_generation_server/models/custom_modeling/flash_gpt2_modeling.py:450-452](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/flash_gpt2_modeling.py#L450-L452)
