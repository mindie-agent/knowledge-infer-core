---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/runbooks/precision-and-runtime/profiling-and-latency-spikes
evidence_strength: software_contract
id: unit-infer-core-npu-profiling-kernel-breakdown-and-latency-spike-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-perf-breakdown/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-profiling/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-runtime-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-recommend-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/plugins-community/triton-optimizer/skills/triton-npu-optimize/SKILL.md
  source_digest: 3c23d1a7a1893f705c16e3ecc89e40e7eda1460a478d9527acf8309fb7ffcaf9
  source_locator: cann/cannbot-skills@f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:model/model-infer-perf-breakdown/SKILL.md
  source_type: cannbot_skill_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: NPU profiling, kernel breakdown, and tail-latency spike diagnosis across Ascend
  inference engines and Triton-NPU kernels
triggers:
  error_signatures:
  - Profiler trace shows large gaps between AICore kernels or unexpected TransData/Cast
    insertions
  - Triton-NPU kernel suffers high scalar latency or low MTE2/UB bandwidth utilization
  - Periodic P99 TPOT latency spikes occur during steady-state serving on Ascend NPU
  keywords:
  - model-infer-perf-breakdown
  - model-infer-profiling
  - triton-npu-optimize
  - kernel_details.csv
  - step_trace_time.csv
  - aic_mac_ratio
  - aiv_vec_ratio
  - mte2_ratio
---
# NPU profiling, kernel breakdown, and tail-latency spike diagnosis across Ascend inference engines and Triton-NPU kernels

## Applicability and symptom scope

Use this SOP to capture low-overhead `torch_npu.profiler` traces, classify bottlenecks from `kernel_details.csv` / `step_trace_time.csv`, optimize Triton-NPU kernels, and eliminate P99 tail-latency spikes across Ascend inference engines.

## Procedure

1. **Capture Stage-Isolated NPU Profiles After Warmup (`model-infer-profiling/SKILL.md`, `model-infer-perf-breakdown/SKILL.md`).**
   - Always warm up `2..5` steps before activating `torch_npu.profiler.profile` so JIT compilation, graph capture, and HCCL communicator initialization do not pollute the trace.
   - Configure `ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1)` with `with_stack=False` for accurate hardware duration measurements (`with_stack=True` inflates host dispatch overhead and is used only in a separate symbol-attribution pass).
   - Separate Prefill and Decode into distinct profiler windows (`prof/prefill` and `prof/decode`) and ensure `input_max_len <= moe_chunk_max_len` so prefill profiling is not silently skipped.

2. **Classify Bottlenecks via Roofline & PipeUtilization Metrics (`model-infer-perf-breakdown/SKILL.md`, `model-recommend-analysis/SKILL.md`).**
   - From `step_trace_time.csv` and `kernel_details.csv`, compute the four primary breakdown buckets:
     1. **Host Dispatch Bound (`Free` time > `15%` of step):** Host Python/C++ launch rate lags behind NPU execution. Fix via decode graph capture (`ACLGraph` / `TorchAir`), `SuperKernel`, or eliminating per-layer `.item()` / scalar tensor construction.
     2. **Memory Bandwidth Bound (`aic_mte2_ratio` or `aiv_mte2_ratio > 0.65`, low `mac_ratio`):** HBM-to-UB/L1 transfer dominates (typical of decode projections and paged attention). Fix via weight/KV quantization (`W8A8`, `W4A8`, `C8`), MLA weight absorption, or `npu_prefetch`.
     3. **Cube/Vector Compute Bound (`aic_mac_ratio > 0.70` or `aiv_vec_ratio > 0.70`):** Saturating arithmetic pipes (typical of prefill GEMMs and DiT attention) or suffering hidden `TransData` / `Cast` insertions. Inspect for unaligned shapes causing `TransData` and fuse adjacent Vector ops (`AddRMSNormQuant`, `SwiGLU`).
     4. **Communication Exposed (`Hccl*` non-overlapped duration > `15%`):** Overlap via dual-stream Shared-Expert / micro-batch ping-pong, MC2 fused collectives, or `SP-TP-SP`.

3. **Eliminate Triton-NPU Kernel Bottlenecks (`triton-npu-optimize/SKILL.md`).**
   - When profiling Triton-on-Ascend kernels via `kernel_details.csv`:
     - **High Scalar Overhead (`scalar_ratio` high):** Replace scalar index loops and per-element conditionals with vectorized 1D/2D `tl.arange` masks and hoist loop-invariant pointer arithmetic outside the inner loop.
     - **High Transfer Pressure / Poor Locality (`mte2_ratio` / `mte3_ratio` high):** Coalesce inner-dimension HBM loads to `32-byte` aligned contiguous blocks, eliminate implicit transposes before store, and size block tiles so double-buffered tensors fit inside the `192 KiB` UB limit.

4. **Triage Periodic P99 Tail-Latency Spikes (`model-infer-runtime-debug/SKILL.md`).**
   - Check for: (a) un-bucketed decode batch sizes triggering synchronous `torch.compile` / `ACLGraph` recompilation; (b) `torch.npu.empty_cache()` or caching-allocator segment compaction triggered at high `gpu_memory_utilization` (mitigate with `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`); and (c) Python GC pauses on worker processes (disable automatic GC during active batch forward and run `gc.collect()` on controlled idle intervals).
