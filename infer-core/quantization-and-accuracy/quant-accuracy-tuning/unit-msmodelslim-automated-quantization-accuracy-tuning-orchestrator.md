---
category: workflow
conditions:
  cann_versions:
  - '>=8.0.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - End-to-end automated quantization and accuracy recovery for LLM/VLM/MoE models
domain_path: infer-core/quantization-and-accuracy/quant-accuracy-tuning
evidence_strength: software_contract
id: unit-msmodelslim-automated-quantization-accuracy-tuning-orchestrator
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/msagent/blob/521f8ead0ebda5db5b282f0c3c4147006e01e016/skills/quantizer/quantization-accuracy-tuning-orchestrator/SKILL.md
  source_digest: aa0b66a5acd2651c046c43086fff5e28f560941bbebe3ce460d91f3284449e96
  source_locator: Ascend/msagent@521f8ead0ebda5db5b282f0c3c4147006e01e016:skills/quantizer/quantization-accuracy-tuning-orchestrator/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Closed-loop workflow from Ascend/msagent (`quantization-accuracy-tuning-orchestrator`)
  for diagnosing quantization accuracy drops via `msmodelslim analyze` and iterating
  outlier suppression (`iter_smooth`, `QuaRot`, `AWQ`, `LAOS`) and sensitive-layer
  rollback.
tier: L3
title: 'Automated Quantization Accuracy Tuning Orchestrator: Outlier Suppression,
  Rollback, and Search Loop'
triggers:
  error_signatures:
  - quantization accuracy tuning orchestrator
  - msmodelslim quant analyze tuning
  keywords:
  - quantization accuracy tuning orchestrator
  - msmodelslim quant analyze tuning
  - w8a8 w4a8 accuracy drop recovery
  - quarot awq iter_smooth laos rollback
  - disable_names sensitive layer rollback
  - quantization-tuning
  - msmodelslim-analyze
  - awq
  - quarot
  - iter-smooth
  - laos
  - aisbench
  - layer-rollback
---
## 1. Closed-Loop Quantization & Accuracy Tuning Workflow

When a baseline quantized checkpoint (`W8A8` static/dynamic, `W8A16`, `W4A16`, or `W4A8`) exhibits an accuracy drop exceeding the target budget (typically `<= 1.0%` on GSM8K / CEVAL / MMLU / MATH_500 via `AISBench`):

```
[1. Baseline FP16/BF16 Eval] -> [2. Layer Sensitivity Analysis (msmodelslim analyze)]
       -> [3. Generate Candidate YAML Config (Outlier Suppression + Rollback)]
       -> [4. Execute msmodelslim quant] -> [5. Serve via vLLM-Ascend & Eval via AISBench]
       -> [6. Converged? Export Recipe : Update Search State & Iterate]
```

## 2. Layer Sensitivity & Outlier Diagnosis (`msmodelslim analyze`)

1. Run `msmodelslim analyze` (using `std`, `quantile`, or `kurtosis` metrics across calibration activations) to rank linear modules by quantization sensitivity score.
2. Inspect structural sensitivity patterns:
   - **`down_proj` / `o_proj` Activation Outliers**: Massive channel spikes caused by SwiGLU or attention sinks.
   - **First 2 & Last 2 Decoder Layers**: Disproportionately impact initial representation formation and final logit projection.
   - **MoE Router Gate (`gate` / `router`) & MLA `kv_a_proj_with_mqa`**: Must **always** remain in `BF16`/`FP16` (`disable_names`).

## 3. Progressive Tuning Ladder (Ordered by Inference Performance Cost)

Apply tuning strategies in strict ascending order of runtime overhead:
1. **Tier 1 — Zero Inference Overhead (Calibration-Time Smoothing & Clipping)**:
   - **`iter_smooth` / `SmoothQuant` (`alpha=0.5..0.85`)**: Migrates activation outliers from $X$ into weight $W$ by fusing scale $s^{-1}$ into the preceding `RMSNorm` weight.
   - **`AWQ` / Weight Clipping (`w_method="autoround"` or `ssz` / `minmax` clip)**: Protects salient weight channels driven by high-magnitude activations.
   - **Calibration Dataset Alignment**: Replace generic calibration text with domain-matched prompts (`calib_set` matching reasoning/code/math distribution, `32..128` samples of length `2048..4096`).
2. **Tier 2 — Orthogonal Rotation (`QuaRot` / Hadamard Transform & `LAOS` for `W4A8`)**:
   - Applies Hadamard orthogonal rotation ($X H \cdot H^	op W^	op$) to disperse channel-wise outliers uniformly across all dimensions prior to 4-bit/8-bit quantization.
   - For `W4A8` MoE models, combine per-group `W4` (`group_size=64` or `128`) with `LAOS` (Low-rank / Learnable Affine Outlier Suppression) and dynamic per-token `A8`.
3. **Tier 3 — Selective Sensitive-Layer Float Rollback (`disable_names` / `rollback`)**:
   - Roll back the top `5%–15%` most sensitive layers identified by `msmodelslim analyze` (starting with `down_proj` in early/late layers and `o_proj`) to `BF16`/`FP16` or `W8A8_dynamic`.

## Verification Checklist

- Confirm `lm_head`, MoE `gate`/`router`, and `kv_a_proj` are excluded from low-bit quantization before launching expensive multi-hour `QuaRot` or `AutoRound` searches.
