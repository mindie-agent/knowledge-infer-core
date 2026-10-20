---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/frameworks-and-platforms/executors-and-serving
evidence_strength: software_contract
id: unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/cann_executor.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/kv_cache_manager.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/online_pd_inference.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/mm_inference_launch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/sglang.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/hixl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/cann_shmem.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/online_weight_split.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/inference_config.md
  source_digest: 61733f6bd867914c8706a3e07ae69afcaf1b5fcfc4393a496830f873911b6ee9
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/frameworks/cann_executor.md
  source_type: cannbot_concept_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Execution model invariants of CANN Executor, SGLang, Online PD Disaggregation,
  KVCacheManager, HIXL, and CANN SHMEM
triggers:
  error_signatures:
  - Disaggregated PD serving deadlocks when Prefill ZMQ push and HIXL one-sided RDMA
    KV transfer mismatch block indices
  - CANN SHMEM symmetric heap allocation fails when symmetric memory pool size is
    not identical across all PE ranks
  - Online weight splitting OOMs on rank 0 when full unsharded safetensors tensor
    is loaded onto NPU before TP/EP slicing
  keywords:
  - cann_executor
  - kv_cache_manager
  - online_pd_inference
  - HIXL
  - cann_shmem
  - online_weight_split
  - InferenceConfig
  - mm_inference_launch
---
# Execution model invariants of CANN Executor, SGLang, Online PD Disaggregation, KVCacheManager, HIXL, and CANN SHMEM

## Condition

Applies to serving engine architecture, KV memory management, and one-sided communication layers across `cann-recipes-infer` (`CANN Executor`), `vllm-ascend`, and `SGLang` on Ascend NPU.

## Constraint

1. **CANN Executor & `InferenceConfig` Lifecycle (`cann_executor.md`, `inference_config.md`, `online_weight_split.md`, `mm_inference_launch.md`):**
   - `CANN Executor` separates execution into deterministic phases: `InferenceConfig` YAML resolution -> ProcessGroup initialization (`TP`/`DP`/`EP`/`CP`/`MC2`) -> **Online Weight Splitting** (streaming only the rank-local slice `[start:end]` from `safetensors` on CPU directly to NPU HBM and applying `process_weights_after_loading` for `FRACTAL_NZ` / W4A8 packing) -> `profile_run` peak activation probe -> `KVCacheManager` block pool allocation -> Decode graph warmup/capture (`GraphWrapper`).
   - `mm_inference_launch` unifies multimodal diffusion/VLM startup with decoupled encoder/backbone device placement and warm-cache reuse.

2. **Heterogeneous `KVCacheManager` Block Pool (`kv_cache_manager.md`, `sglang.md`):**
   - Block `0` (`null_block`) in `FreeKVCacheBlockQueue` is permanently reserved as a safe sink for padded/masked tokens (`slot_mapping` pointing to block `0` offset `0`) so graph-mode padded tokens never corrupt live request KV data.
   - Hybrid models simultaneously register `FullAttentionSpec`, `SlidingWindowSpec` (evicting blocks below `seq_len - window_size` back to the shared pool via `free_then_allocate_blocks`), and fixed-footprint `MambaSpec` state blocks (`1` or `2` blocks per active request slot).

3. **Online PD Disaggregation & HIXL / CANN SHMEM One-Sided Transport (`online_pd_inference.md`, `hixl.md`, `cann_shmem.md`):**
   - **Online PD Control vs. Data Plane:** ZMQ sockets carry lightweight request metadata (`req_id`, first sampled token, remote KV block IDs), while **HIXL (`Huawei Xfer Library`)** executes one-sided zero-copy `Write`/`Read` of `2 MiB`-aligned HBM KV pages over HCCS or RoCE RDMA without involving the remote NPU's compute stream.
   - **CANN SHMEM (`cann_shmem.md`):** Maps peer NPUs' symmetric HBM heaps into a global virtual address space over HCCS (`shmem_putmem_nbi` / `shmem_getmem_nbi` / `shmem_barrier_all`), enabling AscendC kernels (such as `shmem_dispatch_combine`) to write routed token payloads directly into remote expert buffers from inside the AIC/AIV kernel without host collective launch overhead.
