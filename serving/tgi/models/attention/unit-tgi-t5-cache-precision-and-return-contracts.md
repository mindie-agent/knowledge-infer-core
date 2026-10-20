---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:032056ab8537ab79692971be4da5c28a42bb703757ed3a5ecd2c66c88341350d
domain_path: serving/tgi/models/attention
evidence_strength: single_observation
id: unit-tgi-t5-cache-precision-and-return-contracts
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-t5-cache-precision-and-return-contracts-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L355-L501
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L641-L716
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L786-L808
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L971-L981
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L1016-L1054
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L1148-L1165
  source_digest: 82e985708747cc12387b83e0ece18f2bcac1a96e5a5493b1d165532795dd1893
  source_locator: docs/evidence/unit-tgi-t5-cache-precision-and-return-contracts-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: T5 cache shape, precision and return conventions are independent contracts
triggers:
  error_signatures: []
  keywords:
  - tgi
  - t5
  - cache
  - shape
  - precision
  - and
  - return
  - conventions
  - are
  - independent
  - contracts
---
This TGI T5 implementation partitions attention heads, uses per-head relative position bias, concatenates self-attention past K/V and can reuse cross-attention K/V by source length. The standard stack constructs relative bias only for its first layer and forwards it to later blocks. Past-state cardinality is selected by encoder_hidden_states presence, while execution of cross-attention additionally requires a decoder. Both output modes contain logits; only return_dict=True includes the separate speculative-logits result.

Scope and caveats:

- Correct constructor citation to 1016-1054, including the shared-head RuntimeError fallback; forward to 1056-1165 and generation/reorder to 1167-1227.
- Use T5LayerFF:204-227 for dense-layer selection and residual; the float32 wo load/cast source does not itself quantify accuracy or throughput.
- The local layer_norm helper is RMS-style, but T5LayerNorm can be replaced by apex FusedRMSNorm at 99-112; do not describe the helper as the unconditional runtime implementation.
- The reorder implementation uses dimension 0 despite an adjacent comment saying the second position; cite code and retain the comment inconsistency.
- VLM load_text_model forwards name to llama/mistral; it does not ignore name universally. Unsupported text/vision model types raise.

Evidence scope: All corrected references read from pinned Git, with truncated block, constructor and file-tail ranges independently recovered. No T5 model execution or kernel timing performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/custom_modeling/t5_modeling.py:355-501](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L355-L501)
- [server/text_generation_server/models/custom_modeling/t5_modeling.py:641-716](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L641-L716)
- [server/text_generation_server/models/custom_modeling/t5_modeling.py:786-808](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L786-L808)
- [server/text_generation_server/models/custom_modeling/t5_modeling.py:971-981](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L971-L981)
- [server/text_generation_server/models/custom_modeling/t5_modeling.py:1016-1054](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L1016-L1054)
- [server/text_generation_server/models/custom_modeling/t5_modeling.py:1148-1165](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/custom_modeling/t5_modeling.py#L1148-L1165)
