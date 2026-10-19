---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:7b1feec659814f4c38d98c258195805f967b465fb137c22e7b241428c87b1147
domain_path: serving/tgi-v3
evidence_strength: single_observation
id: unit-tgi-v3-cancellation-and-batch-reclamation
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/text-generation-inference-reviewed-source.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/backend.rs#L1-L572
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L84-L89
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L91-L98
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L1-L221
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L1-L1041
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/client/sharded_client.rs#L1-L259
  source_digest: 8519a8d038aa254aa7c718c55e57d58d9357a4d46c48cc365d8232c7258233d4
  source_locator: docs/evidence/text-generation-inference-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Trace TGI v3 channel closure, cached batches and shard result handling
triggers:
  error_signatures: []
  keywords:
  - text-generation-inference
  - cancellation
  - reclamation
  - ownership
  - cache
---
Use this diagnostic for the v3 backend at the exact repository revision in
conditions. Observe response-channel state, cached-batch membership, allocator
ownership and shard results independently.

1. Trace the response channel attached to the queued `Entry`. In the inspected
   prefill path, `client.prefill` returns before `send_responses` observes channel
   closure and `filter_send_generations` removes entries. Follow `filter_batch`
   updating or clearing the next cached batch. Do not label this sequence a
   pre-forward cancellation acknowledgement.
   [Prefill ordering](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/backend.rs#L297-L340),
   [Batch filtering and response-channel checks](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/backend.rs#L389-L475)
2. Inspect `FilterBatch` popping, filtering and reinserting cached batch state.
   Distinguish that state from allocator capacity: `SimpleAllocator.free`
   extends a free-block list; `RadixAllocator.free` decreases a prefix reference,
   may insert uncached prefill tokens into the trie, returns duplicate or
   non-prefill blocks to the free list, and leaves retained prefix blocks for
   later trie eviction.
   [Cached batch filtering](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L91-L98),
   [Simple allocator free list](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/block_allocator.rs#L218-L221),
   [Radix allocator and prefix ownership](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/radix.rs#L157-L209)
3. Verify both completion and error aggregation for shard operations.
   `ShardedClient.clear_cache` joins all shard futures and collects their
   `Result` values. `filter_batch` also joins all futures, but returns only the
   last `Result`. Waiting for all futures and propagating every shard error are
   different properties in these inspected methods.
   [Shard completion and result aggregation](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/v3/src/client/sharded_client.rs#L74-L101)

These source checks do not establish measured HBM release, a reproduced
distributed or memory-safety failure, immediate device-work cancellation, or
global quiescence. The accepted source subset and exact file/range hashes are
in `docs/evidence/text-generation-inference-reviewed-source.json`.
