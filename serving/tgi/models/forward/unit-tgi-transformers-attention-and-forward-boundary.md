---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:11b92bce5376cab469d035f4bd1db366ec8fed62cd1cc99480e28722abacbe69
domain_path: serving/tgi/models/forward
evidence_strength: single_observation
id: unit-tgi-transformers-attention-and-forward-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-transformers-attention-and-forward-boundary-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_causal_lm.py#L20-L84
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_causal_lm.py#L255-L290
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L69-L143
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L311-L315
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L358-L415
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/vlm_causal_lm.py#L551-L617
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/vlm_causal_lm.py#L959-L1077
  source_digest: 7258c459d3634505d4046e8536528847bb8e20fa9dd0aa2df53614e37888f185
  source_locator: docs/evidence/unit-tgi-transformers-attention-and-forward-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Replacing Transformers attention leaves a second contract at the model-forward
  boundary
triggers:
  error_signatures: []
  keywords:
  - tgi
  - replacing
  - transformers
  - attention
  - leaves
  - a
  - second
  - contract
  - at
  - the
  - model-forward
  - boundary
---
The registered TGI text attention hook stores each layer's K/V into supplied slots and dispatches prefill to variable-length attention and decode to paged attention, while the bridge disables the Transformers cache. VLM text prefill can alternatively pad and use SDPA; vision is configured for SDPA. Image encoding is scheduled for token spans intersecting the current prefill window, with per-request encoder caching. These mechanisms require compatible forward signatures and model-specific mask/position methods.

Scope and caveats:

- Device selection permits IPEX CPU when XPU is absent. Both Transformers loaders explicitly reject speculator.
- The graph warmup always creates and captures a graph; its buffer reuse is not capture reuse.
- The patched Transformers VLM _model_forward requires input_ids and has no attention_mask parameter; base VLM forward passes attention_mask and omits input_ids. Constructor:311-315 confirms the forward patch. This is a source-level interface mismatch; no deployed model was run.
- Do not compare self.model.get_attention_mask directly with wrapper self.get_attention_mask: no mask-method patch is shown at 311-315, so the called model object's method remains unverified.
- Restrict the prefix-caching rejection claim to VlmCausalLM.__init__:709-710. The Transformers constructor skips directly to Model via super(FlashCausalLM,self), so this guard cannot be assumed to run there.
- The speculative branch leaves inputs_embeds unset, but loader-level speculator rejection limits reachable combinations.
- Text-hook attention_mask is unused; adapter_data and prefill_cache_indices are accepted but unused in bridges. No general model-compatibility or performance claim follows.

Evidence scope: Pinned text hook/loader/bridge and VLM SDPA, masks, constructor patch, image-window, cache-release, graph and forward ranges read; Qwen position implementation read. Dataclass boilerplate was not independently reviewed for this narrowed unit.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/transformers_flash_causal_lm.py:20-84](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_causal_lm.py#L20-L84)
- [server/text_generation_server/models/transformers_flash_causal_lm.py:255-290](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_causal_lm.py#L255-L290)
- [server/text_generation_server/models/transformers_flash_vlm.py:69-143](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L69-L143)
- [server/text_generation_server/models/transformers_flash_vlm.py:311-315](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L311-L315)
- [server/text_generation_server/models/transformers_flash_vlm.py:358-415](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/transformers_flash_vlm.py#L358-L415)
- [server/text_generation_server/models/vlm_causal_lm.py:551-617](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/vlm_causal_lm.py#L551-L617)
- [server/text_generation_server/models/vlm_causal_lm.py:959-1077](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/vlm_causal_lm.py#L959-L1077)
