---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-compute-fusion/operator-and-backend
evidence_strength: software_contract
id: unit-infer-core-operator-backend-selection-dit-rewrite-and-lora-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/op_backend_selection_ascendc_pypto_tilelang.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/fused_op_hardware_conditional_fallback.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/ascendc_custom_op_out_of_band_dtype.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/torchless_lowprec_dtype_acl_override.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dit_cnn_operator_algebraic_rewrite_npu.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/dit_mxfp_lowbit_quant_conv_as_linear.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/multi_lora_bgmv_ascendc_vector_kernel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/structural_isomorphic_model_reuse_stack.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-harmony/SKILL.md
  source_digest: 394e708cea357026073c630bc104dda12d08f93bd7fd5276c81888d71e59353a
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/op_backend_selection_ascendc_pypto_tilelang.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 1C (Operator Backends & Rewrites): AscendC
  priority, out-of-band FP8/HiF8 dtypes, DiT 1x1-conv algebraic rewrite, and Multi-LoRA
  BGMV'
triggers:
  error_signatures:
  - PyTorch custom op schema rejects HiFloat8 or sub-byte quantized tensor because
    torch.dtype has no native enum
  - DiT 1x1 Conv2d / Conv3d layers execute slowly on NPU vector/im2col path instead
    of Cube MatMul
  - Fused operator crashes when M or N dimension exceeds 65535 hardware axis ceiling
    without fallback
  keywords:
  - op_backend_selection_ascendc
  - ascendc_custom_op_out_of_band_dtype
  - torchless_lowprec_dtype_acl_override
  - dit_cnn_operator_algebraic_rewrite
  - multi_lora_bgmv_ascendc
  - structural_isomorphic_model_reuse
---
# Inference optimization rules Part 1C (Operator Backends & Rewrites): AscendC priority, out-of-band FP8/HiF8 dtypes, DiT 1x1-conv algebraic rewrite, and Multi-LoRA BGMV

## Applicability and symptom scope

Use this SOP when choosing between multiple custom-operator implementations, passing hardware-only low-precision formats (`HiFloat8` / custom FP8) across PyTorch C++ bindings, rewriting CNN/DiT layers for Cube utilization, or enabling Multi-LoRA BGMV on Ascend NPU.

## Procedure

1. **Prioritize AscendC over PyPTO/TileLang and gate hardware-conditional fallbacks (`op_backend_selection_ascendc_pypto_tilelang.md`, `fused_op_hardware_conditional_fallback.md`, `structural_isomorphic_model_reuse_stack.md`).**
   - When an operator provides `AscendC`, `PyPTO`, and `TileLang` implementations, default to **AscendC** in production because it supports Eager, `ACLGraph`, `TorchAir` GE Graph, and `SuperKernel` without backend lock-in.
   - Gate fused operators on explicit SoC whitelists and axis limits (e.g., falling back to chunked/unfused matmul when batch/token axis exceeds the `65535` hardware descriptor ceiling).
   - For structurally isomorphic models (e.g., a new MoE or MLA model sharing `DeepSeek-V3` or `Qwen3-MoE` block topology), reuse the verified model runner and fused-operator stack via config mapping rather than authoring a duplicate model class.

2. **Pass PyTorch-unrepresentable low-precision dtypes (`HiFloat8` / `MXFP4`) via out-of-band `int` dtype tags and `aclTensor` overrides (`ascendc_custom_op_out_of_band_dtype.md`, `torchless_lowprec_dtype_acl_override.md`).**
   - When PyTorch lacks a `torch.dtype` enum for an Ascend hardware format (such as `HiFloat8` / `ACL_HIFLOAT8`), store the raw bytes in a `torch.uint8` / `torch.int8` container tensor, pass an explicit `int64_t acl_dtype` parameter through the C++ operator signature, and override the `aclDataType` descriptor (`aclCreateTensor` / `TensorWrapper`) right at the `aclnn` / AscendC kernel launch boundary.

3. **Rewrite DiT `1x1` Convolutions into Cube `Linear` MatMuls and fold `LayerNorm + adaLN` (`dit_cnn_operator_algebraic_rewrite_npu.md`, `dit_mxfp_lowbit_quant_conv_as_linear.md`).**
   - A `Conv2d` / `Conv3d` with `kernel_size=1, stride=1, padding=0` is algebraically identical to a `Linear` projection over the channel axis after permuting channels to the last dimension (`[B, C, H, W] -> [B, H, W, C]`). Rewriting `1x1` convolutions into `nn.Linear` unlocks full Cube MatMul throughput and allows direct replacement with `MXFP8` (`a8w8`) / `MXFP4` quantized linear layers (with selective fallback to `a8w8` on layers where `a4w4` degrades FID/PSNR).
   - Fold `LayerNorm(x, elementwise_affine=False)` followed by `adaLN` scale/shift modulation into a single fused affine norm call.

4. **Route Multi-LoRA serving through the custom AscendC `bgmv` vector kernel (`multi_lora_bgmv_ascendc_vector_kernel.md`).**
   - On Atlas A2/A3 (non-310P), replace looped per-adapter PyTorch slicing with the AscendC `bgmv_shrink` / `bgmv_expand` vector kernels when `lora_rank in {8, 16, 32, 64}` and dtype is `float16` / `bfloat16`.
