---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:84e883d393262dbd8d2b3990089b3969dea71f7a35f34352990884b7741c97c3
domain_path: serving/tgi/models/attention
evidence_strength: single_observation
id: unit-tgi-cache-axes-and-mask-semantics
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-cache-axes-and-mask-semantics-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L210-L229
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L498-L515
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L613-L627
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/phi_modeling.py#L164-L183
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/phi_modeling.py#L273-L287
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/qwen2_5_vl.py#L774-L813
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/qwen2_vl.py#L378-L386
  source_digest: 82d4316c6886fcdf947bef54a29bb796409f7853ce3d1956d6a76974856e4be0
  source_locator: docs/evidence/unit-tgi-cache-axes-and-mask-semantics-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Cache axes and attention-mask meaning are part of each model's serving contract
triggers:
  error_signatures: []
  keywords:
  - tgi
  - cache
  - axes
  - and
  - attention-mask
  - meaning
  - are
  - part
  - of
  - each
  - model
  - s
  - serving
  - contract
---
OPT appends K/V along axis2 and reads its cached sequence length from shape[2], while Phi uses axis1 and shape[1]. OPT derives positions from the attention mask and combines causal/padding masks. Phi uses the mask only as a non-None flag for a fixed triangular penalty and disables that path for single-token steps. Qwen2.5 vision reorders tokens into windows, selects full or window attention by layer, merges patches, then restores ordering; Qwen2 uses full packed sequence boundaries for every vision block.

Scope and caveats:

- Qwen2-VL locally divides config.embed_dim, not config.hidden_size, at74. Its separately assigned head_dim is not used in the shown attention shaping/scaling path; do not report a demonstrated shape failure from that assignment alone.
- Phi does not read supplied padding-mask values, and its upper-triangular mask has no offset for a nonempty past. Do not infer generic cached multi-token correctness from one-token decode behavior.
- Qwen2.5 restores order after the patch merger at811-813, not before.
- Qwen processor expands both image and video placeholders, but the shown embedding merge replaces only image_token_id rows. Complete video handling remains dependent on external preprocessing/callers.
- OPTForCausalLM directly reads decoder object attributes and returns a fixed two-tuple. Return-dict=False compatibility requires checking the decoder's actual return path; no runtime invocation was performed here.
- SigLIP's attention accepts an additive mask and uses fp32 softmax, while its shown transformer forward supplies no mask. A pooling-head definition alone does not establish its use.

Evidence scope: All 24 indexed source-reference ranges read from pinned Git; no model loading, cache replay or hardware benchmark executed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/opt_modeling.py:210-229](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L210-L229)
- [server/text_generation_server/models/custom_modeling/opt_modeling.py:498-515](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L498-L515)
- [server/text_generation_server/models/custom_modeling/opt_modeling.py:613-627](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/opt_modeling.py#L613-L627)
- [server/text_generation_server/models/custom_modeling/phi_modeling.py:164-183](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/phi_modeling.py#L164-L183)
- [server/text_generation_server/models/custom_modeling/phi_modeling.py:273-287](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/phi_modeling.py#L273-L287)
- [server/text_generation_server/models/custom_modeling/qwen2_5_vl.py:774-813](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/qwen2_5_vl.py#L774-L813)
- [server/text_generation_server/models/custom_modeling/qwen2_vl.py:378-386](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/qwen2_vl.py#L378-L386)
