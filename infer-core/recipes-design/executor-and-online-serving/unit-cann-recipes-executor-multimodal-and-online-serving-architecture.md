---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - LLM, VLM, and DiT model execution and online serving in cann-recipes-infer
domain_path: infer-core/recipes-design/executor-and-online-serving
evidence_strength: software_contract
id: unit-cann-recipes-executor-multimodal-and-online-serving-architecture
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/executor_design.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/online_inference_design.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/design/mm_inference_design.md
  source_digest: 96aab0b64afc449cf0e82148f492d91b10ec84ad44ea3a069d11de0a4774ea93
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/design/executor_design.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Architectural specification of the cann-recipes-infer Executor, Online Inference
  Engine (continuous batching, chunked prefill, multi-rank SPMD worker loop), Multimodal
  DiT Runner, and New Model Checklist.
tier: L2
title: CANN-Recipes-Infer Executor, Online Serving Engine, and Multimodal DiT Design
  Architecture
triggers:
  error_signatures:
  - cann recipes infer executor design
  - online inference continuous batching ascend
  keywords:
  - cann recipes infer executor design
  - online inference continuous batching ascend
  - multimodal dit vae executor design
  - new model adaptation checklist cann recipes
  - executor-design
  - online-serving
  - mm-inference-design
  - continuous-batching
  - chunked-prefill
  - new-model-checklist
---
## 1. Unified Executor Architecture (`docs/design/executor_design.md`)

`cann-recipes-infer` decouples model definition from distributed runtime orchestration via a 4-layer `Executor` hierarchy:
1. **`ModelRunner` / `Executor` Lifecycle**:
   - `init_distributed()` -> `load_model_and_weights()` -> `init_kv_cache()` -> `warmup_and_compile_graph()` -> `execute_step(batch_input)`.
2. **SPMD Multi-Rank Coordination**:
   - Rank 0 (`Driver / API Server`) tokenizes requests, runs the `Scheduler`, and broadcasts compact `ScheduleMetadata` (token IDs, positions, block tables, `actual_seq_lengths`) to worker ranks via shared memory or HCCL broadcast so all ranks execute identical control flow through `Model.forward()`.
3. **Graph Compilation & Static Buffer Binding**:
   - Pre-allocates static input buffers (`input_ids`, `positions`, `slot_mapping`, `block_tables`, `actual_seq_lengths_kv`) for each batch-size bucket (`1, 2, 4, 8, 16, 32, 64`).
   - During `execute_step`, copies dynamic batch tensors into the matching static bucket buffer (`buffer[:N].copy_(batch)`) and replays the captured `npugraph_ex` / `torchair` graph without re-allocating device memory.

## 2. Online Inference Serving Engine (`docs/design/online_inference_design.md`)

1. **Asynchronous Frontend & Continuous Batching Scheduler**:
   - OpenAI-compatible HTTP/SSE server (`FastAPI` + `uvicorn`) communicates with the backend engine process via zero-copy IPC queues (`ZMQ` / `multiprocessing.Queue`).
   - The `Scheduler` maintains three queues (`waiting`, `running`, `swapped`) and forms hybrid or phase-split batches every iteration according to available KV cache blocks in `BlockManager`.
2. **Prefix Caching (Radix / Hash Block Reuse) & Chunked Prefill**:
   - Hashes token blocks (`block_size=16` or `128`) so shared system prompts reuse existing physical KV blocks (`ref_count` increment) without recomputation.
   - Splits long incoming prompts into fixed token chunks (`max_num_batched_tokens`) so ongoing decode requests maintain stable `TPOT` SLAs.

## 3. Multimodal & DiT Inference Architecture (`docs/design/mm_inference_design.md`)

1. **Three-Stage Pipeline (`Text Encoder -> DiT Denoising Loop -> 3D/2D VAE Decoder`)**:
   - Supports stage-wise memory offloading or stage-specific parallelism: e.g., `CFG Parallel (2x) + Ulysses SP (4x) + Ring Attention (2x)` inside the DiT loop, switching to spatial tile parallelism (`VAE Patch Parallel`) during high-resolution video VAE decoding.
2. **Declarative Optimization Patching**:
   - Applies non-intrusive monkey-patches (`patch_attention`, `patch_rope`, `patch_norm`, `patch_fbcache`, `patch_quant`) onto upstream `diffusers` / model pipelines during initialization.

## Verification Checklist

- Confirm every new model recipe implements the 5-step gate (`Precision Check vs. Golden` -> `Parallelism Split` -> `Static KV Cache` -> `Fused Operator Replacement` -> `Graph Mode Capture & Benchmark`).
