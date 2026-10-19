---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:995c689ad0ce417019829312eebe3b34cd0f219397acd249db5475b7160f77fd
domain_path: serving/tgi/models/forward
evidence_strength: single_observation
id: unit-tgi-model-state-layout-and-backend-guards
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-model-state-layout-and-backend-guards-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mamba_modeling.py#L94-L167
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mamba_modeling.py#L226-L237
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L81-L91
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L371-L378
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L867-L886
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/neox_modeling.py#L171-L218
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/neox_modeling.py#L536-L561
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mllama.py#L857-L885
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mllama.py#L970-L1027
  source_digest: 8c6bdc1a2f579ec1bb6139c1dd2729083abe4f7c731d98503f0ca1a99a5a8789
  source_locator: docs/evidence/unit-tgi-model-state-layout-and-backend-guards-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: State layout and backend guards must be checked per model path
triggers:
  error_signatures: []
  keywords:
  - tgi
  - state
  - layout
  - and
  - backend
  - guards
  - must
  - be
  - checked
  - per
  - model
  - path
---
Mamba writes convolution/SSM results back to external InferenceParams and advances seqlen_offset; subsequent calls use the step path. MPT selects flash, Triton or Torch attention, keeps rank-sliced bias and enforces explicit model/sequence restrictions. NeoX applies rotary to a partial head dimension and its manual attention path grows K/V along the sequence dimension. MLLama builds variable-length cross-attention metadata and gathers/scatters only selected image-associated text rows.

Reviewed limits and corrections:

- MPT's unequal query/key causal guard runs only when original_is_causal is true; do not apply it to every backend/configuration.
- The MPT dtype/CUDA check belongs to its flash/Triton wrappers, not to NeoX.
- MPT LPLayerNorm replaces normalized_shape with the loaded sharded weight shape, so its constructor argument alone does not prove a mismatch.
- NeoX positions read past dimension -2 while its mask reads -1; this conflicts with the manual path's [batch,heads,sequence,head_dim] cache. The imported fused kernel's returned layout was not inspected.
- MPT lists MultiQueryAttention but MPTBlock rejects that configuration; its self.head_dim references do not establish a reachable configured-model failure through MPTBlock.
- MLLama projects K/V from the tensor unpacked from cross_attention_states, not from the whole tuple. The zero-padding slice at58-59 clears all entries when pad_patches==0; occurrence with a deployed configuration remains unverified.
- Mamba step squeezes axis1 and does not provide a continued multi-token prefill path in this file; actual external kernel shape validation was not executed.

Evidence scope: All 46 indexed source-reference ranges read from pinned Git, including manual cache construction and caller dimension reads. No imported native-kernel execution.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/mamba_modeling.py:94-167](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mamba_modeling.py#L94-L167)
- [server/text_generation_server/models/custom_modeling/mamba_modeling.py:226-237](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mamba_modeling.py#L226-L237)
- [server/text_generation_server/models/custom_modeling/mpt_modeling.py:81-91](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L81-L91)
- [server/text_generation_server/models/custom_modeling/mpt_modeling.py:371-378](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L371-L378)
- [server/text_generation_server/models/custom_modeling/mpt_modeling.py:867-886](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L867-L886)
- [server/text_generation_server/models/custom_modeling/neox_modeling.py:171-218](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/neox_modeling.py#L171-L218)
- [server/text_generation_server/models/custom_modeling/neox_modeling.py:536-561](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/neox_modeling.py#L536-L561)
- [server/text_generation_server/models/custom_modeling/mllama.py:857-885](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mllama.py#L857-L885)
- [server/text_generation_server/models/custom_modeling/mllama.py:970-1027](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/mllama.py#L970-L1027)
