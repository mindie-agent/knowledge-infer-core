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
  - Fused MoE GMM execution and W8A8/MXFP8 linear quantization across cann-recipes-infer
    models
domain_path: infer-core/acceleration-modules/moe-gmm-linear-and-mxquant
evidence_strength: reproduced_empirical
id: unit-cann-recipes-module-fuse-moe-gmm-linear-and-mxfp8-quantization
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/module/fuse_moe_gmm.py
  source_digest: 2e3a711c9571c6424cb7c19e532dcf3ddb5b6c69a79532095dc8647cdc4b063c
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:module/fuse_moe_gmm.py
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Implementation contract for `module/fuse_moe_gmm.py` and quantized linear
  modules covering `npu_moe_init_routing_v2`, `npu_grouped_matmul` (`split_item=2/3`),
  `npu_moe_finalize_routing`, and `W8A8`/`MXFP8` execution.
tier: L2
title: 'Reusable Acceleration Modules: Fused MoE GMM, Parallel Linear, and MXFP8/W8A8
  Quantization'
triggers:
  error_signatures:
  - module fuse_moe_gmm.py
  - npu grouped matmul split_item 2 3
  keywords:
  - module fuse_moe_gmm.py
  - npu grouped matmul split_item 2 3
  - npu moe init routing v2 finalize routing
  - mxfp8 dynamic quant matmul ascend 950
  - fuse-moe-gmm
  - npu-grouped-matmul
  - mxfp8
  - w8a8
  - split-item
  - moe-init-routing-v2
---
## 1. Canonical Fused MoE GMM Pipeline (`module/fuse_moe_gmm.py`)

`module/fuse_moe_gmm.py` replaces per-expert loops with a 5-stage deterministic NPU pipeline:
1. **Top-K Gating (`npu_moe_gating_top_k_softmax` or `npu_moe_gating_top_k`)**:
   - Computes `expert_weights` (`[T, K]`), `expert_idx` (`[T, K]`, `int32`), and `row_idx` directly from router logits `x @ gate_weight`.
2. **Token Permutation & Expert Histogram (`npu_moe_init_routing_v2`)**:
   - Calls `expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, _ = torch_npu.npu_moe_init_routing_v2(x, expert_idx, active_num=T*K, expert_num=num_local_experts, expert_tokens_num_type=1, expert_tokens_num_flag=True)`.
   - `expert_tokens_num_type=1` outputs per-expert token counts (`0` outputs prefix-sum cumsum), matching `group_list_type` in `npu_grouped_matmul`.
3. **First Grouped GEMM (`gate_up_proj`) + Fused `SwiGLU`**:
   - Weight shape: `[num_local_experts, hidden_size, 2 * intermediate_size]` (pre-transposed at load time so `GMM` reads contiguous inner dimensions without runtime transpose).
   - Executes `gate_up_out = torch_npu.npu_grouped_matmul([expanded_x], [w13], bias=None, group_list=expert_tokens, split_item=2, group_type=0, group_list_type=1)[0]`.
   - Applies `act_out = torch_npu.npu_swiglu(gate_up_out, dim=-1)`.
4. **Second Grouped GEMM (`down_proj`)**:
   - Weight shape: `[num_local_experts, intermediate_size, hidden_size]`.
   - Executes `down_out = torch_npu.npu_grouped_matmul([act_out], [w2], bias=None, group_list=expert_tokens, split_item=2, group_type=0, group_list_type=1)[0]`.
5. **Unpermutation & Weighted Reduction (`npu_moe_finalize_routing`)**:
   - Combines `K` expert outputs per token with optional shared-expert / residual add via `torch_npu.npu_moe_finalize_routing(down_out, skip1=shared_out, skip2=None, bias=None, scales=expert_weights, expanded_src_to_dst_row=expanded_row_idx, export_for_source_row=expert_idx, drop_pad_mode=2)`.

## 2. Quantized `W8A8` & `MXFP8` GroupedMatmul Execution

- **`W8A8` Dynamic Quantized MoE**:
  - Quantizes `expanded_x` via `npu_dynamic_quant`, runs `npu_grouped_matmul` with `scale=[w13_scale]` and `per_token_scale=[x_scale]`, or fuses `GMM1 + Dequant + SwiGLU + Quant` into `aclnnGroupedMatmulSwigluQuantV2` before `GMM2`.
- **`MXFP8` Microscaling (`Ascend 950PR/DT`)**:
  - Uses block-32 shared exponents (`E8M0` scale per 32 elements + `FLOAT8_E4M3FN` elements) supported natively by `aclnnGroupedMatmulSwigluQuantV2` on Ascend 950.

## Verification Checklist

- Pre-transpose expert weights `[E, Out, In] -> [E, In, Out]` and call `.contiguous()` once during `load_weights()`; runtime `.transpose(-1, -2)` inside `forward()` creates non-contiguous views or copies every step.
- Ensure `expert_tokens_num_type` in `npu_moe_init_routing_v2` matches `group_list_type` in `npu_grouped_matmul` (`1` = count, `0` = cumsum).
