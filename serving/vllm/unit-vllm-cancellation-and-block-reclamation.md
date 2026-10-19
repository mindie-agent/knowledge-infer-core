---
category: sop
conditions:
  repository: vllm-project/vllm
  revision: 752a3a504485790a2e8491cacbb35c137339ad34
content_digest: sha256:c727bac94c37140e958b7164975c02a831827612b0126554e16e0306fb441578
domain_path: serving/vllm
evidence_strength: single_observation
id: unit-vllm-cancellation-and-block-reclamation
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/vllm-reviewed-source.json
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2036-L2097
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2099-L2116
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2130-L2143
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2145-L2157
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2352-L2385
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2474-L2501
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2118-L2121
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L1499-L1517
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L614-L635
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L542-L572
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L574-L595
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/single_type_kv_cache_manager.py#L402-L410
  - https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/single_type_kv_cache_manager.py#L384-L400
  source_digest: 4aa96aa6375feeb4dd9349a7cdfe7866897ea0355a9b336819de46ea6873f6bf
  source_locator: docs/evidence/vllm-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Trace vLLM v1 cancellation, deferred block return and cache ownership
triggers:
  error_signatures: []
  keywords:
  - vllm
  - cancellation
  - reclamation
  - ownership
  - cache
---
Use this diagnostic in the v1 scheduler and KV-cache paths at the exact
repository revision in conditions. Keep request termination, ownership release,
and reusable block-pool capacity as separate observations.

1. Trace `finish_requests` for the affected request: queue removal and terminal
   status lead to `_free_request`. Check remote-KV receive state and connector
   return values before concluding that `_free_blocks` ran.
   [Request finishing](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2036-L2097),
   [Conditional request release](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2099-L2116)
2. Follow `_free_blocks` into `_free_request_blocks`. Inspect whether the blocks
   were returned immediately or placed behind the processed-step fence. Follow
   `update_from_output` advancing the fence before its deferred-free drain.
   [Block-release caller](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2118-L2121),
   [Deferred block-return path](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2474-L2501),
   [Output processing and deferred-free drain](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L1499-L1517)
3. Inspect request bookkeeping separately from pool and cache state:
   `pop_blocks_for_free` removes request bookkeeping without returning blocks to
   the pool. `BlockPool.free_blocks` decrements references and requeues
   zero-reference non-null blocks. Cached hashes are evicted when
   `get_new_blocks` selects a block with caching enabled.
   [Request bookkeeping](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/single_type_kv_cache_manager.py#L402-L410),
   [Reference decrement and free queue](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L614-L635),
   [Block allocation](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L542-L572),
   [Cached-hash eviction](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py#L574-L595)
4. Follow `finished_recving` and `finished_sending` separately. A receive
   completion may make a waiting request ready or release a terminal request;
   deferred block return still has its own processed-step condition.
   [Receive completion branch](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2130-L2143),
   [Send completion branch](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L2145-L2157)

These are source-based diagnostic checks. The inspected methods do not prove
physical device-memory deallocation, immediate termination of launched device
work, a reproduced fault, or a measured performance result. Retain pending state
until the particular queue or completion condition has been observed. The
accepted source subset and exact file/range hashes are in `docs/evidence/vllm-reviewed-source.json`.
