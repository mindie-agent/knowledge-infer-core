---
category: sop
conditions:
  cann_versions:
  - '>=8.0.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - Precision debugging, NaN/Inf root-cause isolation, and GPU-vs-NPU operator alignment
domain_path: infer-core/quantization-and-accuracy/msprobe-nan-and-determinism
evidence_strength: software_contract
id: unit-msprobe-nan-overflow-detection-and-deterministic-alignment-sop
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/msagent/blob/521f8ead0ebda5db5b282f0c3c4147006e01e016/skills/accuracy/nan-overflow-detection/SKILL.md
  source_digest: 4d111b48d201815778fad61944952923d64972dc8f1fcdc86e1917d36bb1f542
  source_locator: Ascend/msagent@521f8ead0ebda5db5b282f0c3c4147006e01e016:skills/accuracy/nan-overflow-detection/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Diagnostic SOP from Ascend/msagent (`nan-overflow-detection`, `deterministic-execution`,
  `msprobe-usage`) for isolating the exact first operator producing NaN/Inf, enforcing
  full NPU determinism, and running two-stage statistics/tensor dumps.
tier: L2
title: msprobe NaN/Inf Overflow Detection, Cross-Platform Dump Comparison, and Determinism
  SOP
triggers:
  error_signatures:
  - nan overflow detection skill msagent
  - msprobe precisiondebugger dump compare
  keywords:
  - nan overflow detection skill msagent
  - msprobe precisiondebugger dump compare
  - npu deterministic mode hccl deterministic
  - locate first nan inf operator torch npu
  - msprobe
  - nan-overflow
  - deterministic-execution
  - accuracy-comparison
  - hccl-deterministic
  - precision-debugger
---
## 1. Full NPU Determinism Configuration (`deterministic-execution`)

Before comparing two runs or debugging intermittent `NaN` / precision drift, eliminate non-deterministic reduction order in AICore and HCCL collectives:
```python
import os, torch, torch_npu

os.environ["HCCL_DETERMINISTIC"] = "true"
os.environ["LCCL_DETERMINISTIC"] = "1"
os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
torch.use_deterministic_algorithms(True)
torch_npu.npu.matmul.allow_hf32 = False
```
- **Diagnostic Rule**: If enabling `HCCL_DETERMINISTIC=true` resolves a multi-run divergence, the variance comes from ring/tree floating-point associative order rather than an operator bug.

## 2. Two-Stage `msprobe` Precision Diagnosis (`nan-overflow-detection` & `msprobe-usage`)

Never dump full `[B, S, H]` tensors across all 60+ layers on the first pass (causes multi-terabyte disk exhaustion). Follow the two-stage funnel:
1. **Stage 1 — Lightweight `statistics` Mode (`summary_mode="statistics"`)**:
   - Configure `config.json` with `"task": "statistics"`, `"level": "L1"` (API + Module level), `"summary_mode": "statistics"`.
   - Wrap the target step with `PrecisionDebugger(config_path="config.json")`: `debugger.start()` -> `model(inputs)` -> `debugger.stop(); debugger.step()`.
   - `msprobe` records `max`, `min`, `mean`, `l2norm`, `nan_count`, and `inf_count` in `dump.json` with `< 5%` overhead.
2. **Stage 2 — Pinpoint First Anomaly (`First-Fault Rule`)**:
   - Scan `dump.json` in execution order for the **first** API call where `Output` contains `NaN`/`Inf` while all `Input` tensors are finite (or where `CosineSimilarity < 0.99` against GPU reference).
   - Re-run with `"task": "tensor"` scoped strictly to that module/API (`"scope": ["Model.layers.43.mlp"]`) to capture exact `.pt`/`.npy` input tensors for standalone single-op reproduction.

## 3. Common NPU `NaN` / Overflow Root Causes & Fixes

1. **`FP16` Dynamic Range Overflow (`max = 65504`)**:
   - Residual stream accumulation or SwiGLU intermediate products (`gate * up`) in deep layers (`layer >= 40`) frequently exceed `65504`.
   - **Fix**: Switch activation dtype to `BF16`, clamp SwiGLU intermediates (e.g., `limit=7.0` in `Step-3.7-Flash` layers 43–44), or scale residuals.
2. **Division-by-Zero in `RMSNorm` / `Softmax` / All-Masked Attention Rows**:
   - When an attention query row is completely masked (`-inf` across all keys), `exp(-inf - (-inf))` yields `NaN`.
   - **Fix**: Ensure causal/sliding-window masks leave at least one valid token (such as attention sink or self-token) unmasked, and verify `eps >= 1e-6` in `npu_rms_norm`.
3. **Uninitialized Workspace or Padding Tail in Ragged/MoE Tensors**:
   - When `expanded_x` has unused drop/pad rows containing garbage/`NaN` memory, multiplying `0.0 * NaN` produces `NaN`. Zero-initialize padded rows before `GMM`.

## Verification Checklist

- Confirm `Input` statistics of the suspected operator are completely free of `NaN`/`Inf` before blaming the kernel implementation.
