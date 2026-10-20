---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:b55298a72793912443a10a9e3ed37bd38d3095dc3bd1852d3828e0c42187a610
domain_path: serving/tgi/models/forward
evidence_strength: single_observation
id: unit-tgi-model-family-batch-state
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-model-family-batch-state-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/model.py#L144-L171
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/idefics_causal_lm.py#L280-L303
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/idefics_causal_lm.py#L404-L422
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/seq2seq_lm.py#L155-L165
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/seq2seq_lm.py#L263-L273
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L36-L74
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L417-L433
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L569-L614
  source_digest: 3e4eb4fbb482941e30cb4590bc68fd630fb16152339d230ab186fb2878b69989
  source_locator: docs/evidence/unit-tgi-model-family-batch-state-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Batch mutation must preserve each model family's state representation
triggers:
  error_signatures: []
  keywords:
  - tgi
  - batch
  - mutation
  - must
  - preserve
  - each
  - model
  - family
  - s
  - state
  - representation
---
TGI uses different state layouts for Idefics, encoder-decoder generation, Mamba and MLLama. Idefics filter/concatenate paths index both pixel values and image masks; Seq2Seq starts with BOS and no past state, then filters four per-layer past tensors after execution. Mamba allocates fixed convolution/SSM state, rejects tensor parallel world sizes above one and gates graph replay on non-prefill state. Incremental decoding withholds a growing suffix ending in U+FFFD.

Scope and caveats:

- Seq2Seq from_pb sets past_key_values=None at 165; it does not populate four past tensors before the first forward.
- Seq2Seq filter computes total_remaining_decode_tokens at 207-237 but its max_tokens expression uses only the last remaining_decode_tokens at 270-273; downstream enforcement and impact are unverified.
- Include Idefics:799 for the rank condition and Mamba:721 for its corresponding condition. The per-row updates have different indentation.
- Reject the reviewer's assertion that repeated MLLama image indices always receive distinct successive offsets: source 69-77 produces [0,1,1] for three repeated kept indices. The shown builder retains only the last image per request, so that synthetic shape's reachability is not established.
- MLLama concatenation uses an image-entry count as offset. Preserve request/image-index alignment as unresolved; no production failure or performance outcome was reproduced.
- Idefics text-only compatibility and token 32000 semantics are not established by these ranges.

Evidence scope: All 21 corrected reference ranges read plus rank-condition, initial-state and accumulation definitions; source-level conclusions only.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/model.py:144-171](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/model.py#L144-L171)
- [server/text_generation_server/models/idefics_causal_lm.py:280-303](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/idefics_causal_lm.py#L280-L303)
- [server/text_generation_server/models/idefics_causal_lm.py:404-422](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/idefics_causal_lm.py#L404-L422)
- [server/text_generation_server/models/seq2seq_lm.py:155-165](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/seq2seq_lm.py#L155-L165)
- [server/text_generation_server/models/seq2seq_lm.py:263-273](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/seq2seq_lm.py#L263-L273)
- [server/text_generation_server/models/mamba.py:36-74](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L36-L74)
- [server/text_generation_server/models/mamba.py:417-433](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L417-L433)
- [server/text_generation_server/models/mamba.py:569-614](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/mamba.py#L569-L614)
