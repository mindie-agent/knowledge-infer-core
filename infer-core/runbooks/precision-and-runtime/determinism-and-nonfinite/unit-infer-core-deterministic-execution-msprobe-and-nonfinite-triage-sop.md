---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/runbooks/precision-and-runtime/determinism-and-nonfinite
evidence_strength: software_contract
id: unit-infer-core-deterministic-execution-msprobe-and-nonfinite-triage-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-train-precision-determinism/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-train-precision-nonfinite/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-train-precision-numerical-mismatch/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-precision-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/training_debug_observability.md
  source_digest: 07a5914482d5c620b0cd21dc1b1b5569c9f9c61aed265d2eac57bf12f29a7a5c
  source_locator: cann/cannbot-skills@f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:model/model-train-precision-determinism/SKILL.md
  source_type: cannbot_skill_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Precision & determinism triage across Ascend engines: HCCL/LCC/MatMul deterministic
  flags, msprobe layerwise bisect, and NaN/Inf overflow isolation'
triggers:
  error_signatures:
  - Two identical NPU inference or training runs with fixed seed diverge bit-by-bit
    after AllReduce or MatMul K-axis reduction
  - Layer output suddenly emits NaN or Inf during FP16/BF16/W8A8 attention or MoE
    forward pass
  - NPU fused operator output cosine similarity drops below 0.99 against CPU FP32
    golden reference
  keywords:
  - HCCL_DETERMINISTIC
  - LCC_DETERMINISTIC
  - CLOSE_MATMUL_K_SHIFT
  - CUBLAS_WORKSPACE_CONFIG
  - msprobe
  - PrecisionDebugger
  - use_deterministic_algorithms
  - cosine_similarity
---
# Precision & determinism triage across Ascend engines: HCCL/LCC/MatMul deterministic flags, msprobe layerwise bisect, and NaN/Inf overflow isolation

## Applicability and symptom scope

Use this SOP to enforce bitwise repeatability across runs, isolate the first operator producing `NaN`/`Inf`, or bisect numerical mismatches between NPU (`MindIE-LLM`, `vLLM-Ascend`, `SGLang`, `MindSpeed`) and a golden reference (`CPU FP32` or baseline eager).

## Procedure

1. **Lock Full-Stack Bitwise Determinism before Bisecting Numerical Drift (`model-train-precision-determinism/SKILL.md`, `model-infer-precision-debug/SKILL.md`).**
   - Floating-point addition non-associativity in multi-stream reductions, atomic scatter-adds, and ring/tree collectives causes run-to-run variance that masks real precision bugs. Export the complete deterministic environment contract across all ranks:
     ```bash
     export HCCL_DETERMINISTIC=true
     export LCC_DETERMINISTIC=1
     export CLOSE_MATMUL_K_SHIFT=1
     export PYTHONHASHSEED=0
     ```
   - In Python initialization, call `torch.use_deterministic_algorithms(True)` and `torch_npu.npu.manual_seed_all(seed)`, and disable asynchronous multi-stream overlap (`enable_multi_stream=False`) during precision bisection so stream scheduling jitter cannot alter reduction order.

2. **Isolate `NaN`/`Inf` Non-Finite Origins via `msprobe` Summary Statistics (`model-train-precision-nonfinite/SKILL.md`, `training_debug_observability.md`).**
   - Configure `msprobe` (`PrecisionDebugger` with `task="statistics"`, `summary_mode="statistics"`) to record lightweight per-module `max`, `min`, `mean`, `norm`, and `nan/inf` counts without dumping multi-terabyte raw tensors.
   - Locate the **first** module in topological execution order whose inputs are finite (`is_finite == True`) and whose output contains `NaN` or `Inf`:
     - **FP16 Overflow (`> 65504`):** Upcast residual stream or attention score accumulation to `BF16` / `FP32`, or clamp pre-activation norms.
     - **Attention Online-Softmax `0 * inf` / `-inf - (-inf)`:** Check whether an all-masked query row (e.g., empty CP shard or sliding-window mask) produced `LSE = -inf` without zero-masking the output tensor.
     - **Dynamic Quantization Divide-by-Zero:** Ensure per-token `dynamic_quant` clamps `absmax` to `eps = 1e-12` on all-zero padded token rows.

3. **Bisect Operator-Level Numerical Mismatch Against CPU FP32 Golden (`model-train-precision-numerical-mismatch/SKILL.md`, `model-infer-precision-debug/SKILL.md`).**
   - Binary-search across Transformer layers (`layer 0 -> layer N/2 -> ...`) using single-step teacher-forced inputs:
     1. Feed the **exact same saved input tensor** from the baseline run into the suspect NPU layer/operator in isolation so upstream rounding differences do not compound;
     2. Evaluate **Cosine Similarity** ($\ge 0.999$ for BF16/FP16, $\ge 0.99$ for W8A8/W4A8), **Max Absolute Error**, and **Root-Mean-Square Relative Error** against a `torch.float32` CPU reference;
     3. When a fused operator (`FIA`, `mla_prolog`, `npu_grouped_matmul`) diverges while its unfused PyTorch decomposition passes, audit layout strides (`contiguous()` vs. strided view), `actual_seq_lengths` semantics (`cumsum` vs. raw lengths), and scale/offset dtypes (`FP32` vs. `BF16`).
