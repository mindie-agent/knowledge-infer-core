---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:30722f9b89626608b2961b12f34fe4427840015bd08ac1cbd03c5deb0f698a41
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-multimodal-cache-and-cross-attention
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-multimodal-cache-and-cross-attention-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_processing.py#L357-L414
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L840-L874
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1047-L1054
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1321-L1342
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1190-L1258
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/llava_next.py#L205-L252
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/llava_next.py#L261-L303
  source_digest: 25ddc4142049953e86b0b15d72bbceb6d20cc4d2c22547704e117a30a80a6e75
  source_locator: docs/evidence/unit-tgi-multimodal-cache-and-cross-attention-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Multimodal cache reuse still depends on image metadata and cross-attention
  scheduling
triggers:
  error_signatures: []
  keywords:
  - tgi
  - multimodal
  - cache
  - reuse
  - still
  - depends
  - 'on'
  - image
  - metadata
  - and
  - cross-attention
  - scheduling
---
The legacy IDEFICS processor pads text and image slots and builds image-attention masks. Its decoder inserts gated cross-attention at multiples of cross_layer_interval, but creates floor(num_hidden_layers/interval) gated blocks. The no-images flag zeroes a residual gate after cross-attention computation; it does not skip that computation. Precomputed image_hidden_states still require pixel_values-derived image-count metadata, and image_attention_mask is dereferenced before the fallback check. LLaVA-NeXT instead projects/spatially unpads image features and merges them into token embeddings before paged decoder execution.

Reviewed limits and corrections:

- For positive layer/interval counts with a non-divisible remainder, created gated-block count can be smaller than the indexed schedule; do not describe this as silently omitting layers.
- IDEFICS cross-attention rejects past_key_value; ordinary decoder past and precomputed image features are different caches.
- A generated image-token comment is not a guard: LLaVA merges whenever vision_embeds is non-None, and passes prefill_cache_indices=None into its decoder.
- The k_layer_norm loader uses a q_layer_norm weight prefix; whether the sharing is intentional is unresolved.
- SDPA receives dropout_p=self.dropout directly; no local evaluation-mode switch is shown. Runtime inference configurations were not inspected.
- RMSNorm's IPEX/CUDA/ROCm/large-width branches and lack of a vision attention-mask argument are source facts, not device compatibility or performance measurements.

Evidence scope: All 22 corrected references read from pinned Git; actual requests, generation expansion and imported-kernel behavior were not executed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/idefics_processing.py:357-414](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_processing.py#L357-L414)
- [server/text_generation_server/models/custom_modeling/idefics_modeling.py:840-874](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L840-L874)
- [server/text_generation_server/models/custom_modeling/idefics_modeling.py:1047-1054](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1047-L1054)
- [server/text_generation_server/models/custom_modeling/idefics_modeling.py:1321-1342](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1321-L1342)
- [server/text_generation_server/models/custom_modeling/idefics_modeling.py:1190-1258](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics_modeling.py#L1190-L1258)
- [server/text_generation_server/models/custom_modeling/llava_next.py:205-252](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/llava_next.py#L205-L252)
- [server/text_generation_server/models/custom_modeling/llava_next.py:261-303](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/llava_next.py#L261-L303)
