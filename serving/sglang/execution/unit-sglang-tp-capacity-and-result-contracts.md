---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:332d8f12e90715fe24236a2daed8635214538a826e334bbdfa2ed5797fa1aea4
domain_path: serving/sglang/execution
evidence_strength: single_observation
id: unit-sglang-tp-capacity-and-result-contracts
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-tp-capacity-and-result-contracts-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L354-L386
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L393-L418
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L543-L555
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L701-L745
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/utils.py#L136-L161
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/utils.py#L479-L495
  source_digest: 7bafbfd21297fe0bdd8149690d2ceafd9452170197d666dc56e38a2630c74696
  source_locator: docs/evidence/unit-sglang-tp-capacity-and-result-contracts-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: TP-worker capacity reports and token-result types are conditional contracts
triggers:
  error_signatures: []
  keywords:
  - sglang
  - tp-worker
  - capacity
  - reports
  - and
  - token-result
  - types
  - are
  - conditional
  - contracts
---
TpModelWorker seed selection first honors an explicit seed, then reuses the elastic joiner's configured seed, then broadcasts a PP-speculative draft worker's seed inside its TP group, otherwise using the world group. The source explains that first PP stages do not construct that draft worker and cannot join its world broadcast. Memory-pool validation uses effective logical token capacity directly, but get_worker_info multiplies it by page_interleave_shard_size before computing the reported request limit and subtracting the input reserve. EAGLE alone reserves output slots using max(top_k * num_steps, max_speculative_num_draft_tokens); these excerpts do not establish one combined admission formula. Delayed sampling finalizes elastic CUDA-graph scaling inside its deferred callback and returns before the immediate/prefill-only finalization line. Prefill-only execution creates dummy zero token IDs as a tensor, so GenerationBatchResult.has_sampled_token_ids, which only tests isinstance(torch.Tensor), cannot prove that sampling actually happened. extra_keep_alive_refs is a documented two-iteration lifetime handoff to the scheduler, not a demonstrated drain protocol.

Reviewed limits and corrections:

- Do not include delayed sampling in the branch that reaches the later immediate finalization call.
- Distinguish memory-pool validation from reported capacity instead of silently equating their formulas.
- The tensor type check is not evidence of a semantic sample.
- No end-to-end distributed deadlock, cancellation, capacity or performance experiment was run.

Evidence scope: Manually read the listed exact Git ranges for the final corrected mechanism. All other model claims remain research notes; no runtime experiment.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/managers/tp_worker.py:354-386](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L354-L386)
- [python/sglang/srt/managers/tp_worker.py:393-418](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L393-L418)
- [python/sglang/srt/managers/tp_worker.py:543-555](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L543-L555)
- [python/sglang/srt/managers/tp_worker.py:701-745](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/tp_worker.py#L701-L745)
- [python/sglang/srt/managers/utils.py:136-161](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/utils.py#L136-L161)
- [python/sglang/srt/managers/utils.py:479-495](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/managers/utils.py#L479-L495)
