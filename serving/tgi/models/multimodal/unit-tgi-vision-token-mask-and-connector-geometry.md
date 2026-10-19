---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:526b6338bf3b917b59f90a3153e053fa4ea8f5a2a2f9cfdb4089a69df9c55fed
domain_path: serving/tgi/models/multimodal
evidence_strength: single_observation
id: unit-tgi-vision-token-mask-and-connector-geometry
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-vision-token-mask-and-connector-geometry-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L685-L714
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L729-L734
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L747-L803
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L638-L661
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L401-L420
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L489-L545
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L579-L595
  source_digest: 5956af13cb3d09558ee4edb5867e6bb48237ee18f86a58820d4f64cf5b114aed
  source_locator: docs/evidence/unit-tgi-vision-token-mask-and-connector-geometry-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Vision-token merging depends on batch masks and connector geometry
triggers:
  error_signatures: []
  keywords:
  - tgi
  - vision-token
  - merging
  - depends
  - 'on'
  - batch
  - masks
  - and
  - connector
  - geometry
---
Idefics2/3 load vision and connector weights through an unquantized loader while preserving a separate text-model quantization configuration. Both filter tensor-valued padding images, run vision and connector stages, stack per-row features, then assign flattened image features into image-token embedding positions. Idefics2 uses learned Perceiver latents; Idefics3 folds a square patch grid with pixel shuffle before projection. Their text forwards pass paged-attention metadata but replace prefill_cache_indices with None.

Reviewed limits and corrections:

- The all-zero test is a tensor equality heuristic, not a semantic test for black images.
- When the initial pixel_attention_mask is None and a second row is processed, the loop uses the originally captured all_pixel_mask=None after mutating the local mask on the first row. Preserve this source-level failure path; actual caller reachability was not run.
- torch.stack requires consistent connector-output shape across rows. Pixel shuffle requires a square sequence grid and compatible scale-factor divisibility.
- Masked assignment does not explicitly check token/feature counts. Do not claim every mismatch necessarily raises; tensor broadcasting can also matter. Correctness requires aligned features and placeholder positions.
- The loaded Perceiver latent count is not locally asserted equal to configured n_latents; do not turn configured intent into a verified output count.
- Do not apply the separate legacy Idefics processor/config defaults to Idefics2/3 without caller evidence.

Evidence scope: All 19 proposed reference ranges read at the pinned source revision; no multimodal model execution or mask fixture run.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/idefics2.py:685-714](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L685-L714)
- [server/text_generation_server/models/custom_modeling/idefics2.py:729-734](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L729-L734)
- [server/text_generation_server/models/custom_modeling/idefics2.py:747-803](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L747-L803)
- [server/text_generation_server/models/custom_modeling/idefics2.py:638-661](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics2.py#L638-L661)
- [server/text_generation_server/models/custom_modeling/idefics3.py:401-420](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L401-L420)
- [server/text_generation_server/models/custom_modeling/idefics3.py:489-545](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L489-L545)
- [server/text_generation_server/models/custom_modeling/idefics3.py:579-595](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/idefics3.py#L579-L595)
