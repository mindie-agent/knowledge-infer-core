---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:4ef2c50f8b4eb79b46bf02116e3106a73c56b1b2199c4f85b16ab9d1db3c1172
domain_path: serving/tgi/runtime
evidence_strength: single_observation
id: unit-tgi-prefill-speculation-and-graph-eligibility
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-prefill-speculation-and-graph-eligibility-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L244-L362
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L891-L952
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L1507-L1584
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L1792-L1888
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/globals.py#L8-L28
  source_digest: ba0343ad3c127d06f1afad8266631298de3883868c9a486b7e90721bc7185a77
  source_locator: docs/evidence/unit-tgi-prefill-speculation-and-graph-eligibility-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Prefill chunking, speculative tokens and decode graphs use separate eligibility
  checks
triggers:
  error_signatures: []
  keywords:
  - tgi
  - prefill
  - chunking
  - speculative
  - tokens
  - and
  - decode
  - graphs
  - use
  - separate
  - eligibility
  - checks
---
FlashCausalLMBatch maps requests to paged block/slot tables and requires at least one uncached token. An incomplete chunk requires speculate==0; prepare_for_prefill separately requires no speculative_ids. Concatenation retains speculative IDs only when all member batches have them. The forward path uses eager execution whenever cu_seqlen_prefill is present, and otherwise can replay the smallest captured decode graph covering the expanded batch. Warmup estimates bytes per KV block and rebuilds the cache from observed free memory.

Scope and caveats:

- AST-verified methods: filter:485-687; concatenate:691-946; prepare_for_prefill:948-1175; warmup:1507-1684; forward:1726-1888. Replace overlapping approximate ranges.
- The cache formula is bytes per block, not bytes per token. max_total_tokens=sum(batch.cache_lengths) is the no-chunking warmup fallback.
- The prefix-cache condition permits flashinfer and both flashdecoding variants despite narrower error text.
- Graph padding writes slot zero; the healthcheck block-0 reservation remains an external invariant, not proved by this model file.
- GalacticaCausalLMBatch uses padded CausalLMBatch fields; do not generalize the Flash path to every model.
- Drop unverified detailed claims about generation sampling on every rank from the compact accepted unit; no throughput gain follows from the code's TunableOp comment.

Evidence scope: Critical construction, mutation, warmup, forward, context, globals and Galactica ranges read from pinned Git; method boundaries parsed from actual source. No CUDA/IPEX execution.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/models/flash_causal_lm.py:244-362](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L244-L362)
- [server/text_generation_server/models/flash_causal_lm.py:891-952](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L891-L952)
- [server/text_generation_server/models/flash_causal_lm.py:1507-1584](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L1507-L1584)
- [server/text_generation_server/models/flash_causal_lm.py:1792-1888](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/flash_causal_lm.py#L1792-L1888)
- [server/text_generation_server/models/globals.py:8-28](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/models/globals.py#L8-L28)
