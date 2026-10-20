---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-communication-and-parallel/hccl-and-mc2-domains
evidence_strength: software_contract
id: unit-infer-core-hccl-comm-groups-mc2-domain-and-collective-prep-sop
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/hccl_comm_group_construction_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/mc2_dedicated_comm_domain.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/tensor_prep_before_hccl_collective.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/moe_dispatch_combine_comm_quant.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/online_gloo_cpu_comm_domain_env_bringup.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/multi_instance_online_master_addr.md
  source_digest: 43a292297411b28be128d06d48645b83ceb86a7a5650b049ecf9d0d1cb8de052
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/hccl_comm_group_construction_pitfalls.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 3B (HCCL & MC2 Domains): Dedicated MC2 communication
  group, concurrent-stream group isolation, and pre-collective tensor prep'
triggers:
  error_signatures:
  - Reusing the standard EP or TP HCCL process group for MC2 npu_moe_distribute_dispatch_v2
    causes HCCL state corruption or hang
  - RowParallelLinear adds bias before AllReduce across TP ranks, multiplying the
    bias term by tp_size
  - Two concurrent NPU streams issuing collectives on the same HCCL group deadlock
    on hardware notify registers
  keywords:
  - hccl_comm_group_construction
  - mc2_dedicated_comm_domain
  - tensor_prep_before_hccl_collective
  - moe_dispatch_combine_comm_quant
  - get_ backend_group
  - hcomm_info
---
# Inference optimization rules Part 3B (HCCL & MC2 Domains): Dedicated MC2 communication group, concurrent-stream group isolation, and pre-collective tensor prep

## Applicability and symptom scope

Use this SOP when constructing `torch.distributed` HCCL process groups, wiring MC2 fused communication operators (`fullmesh_v2`, `dispatch_v2`/`combine_v2`), or preparing output buffers before collective calls on Ascend NPU.

## Procedure

1. **Allocate a dedicated HCCL communication domain for MC2 and per-stream concurrent collectives (`mc2_dedicated_comm_domain.md`, `hccl_comm_group_construction_pitfalls.md`).**
   - **MC2 Isolation Rule:** `npu_moe_distribute_dispatch_v2` / `combine_v2` (`fullmesh_v2`) manages internal HCCL RDMA/SDMA ring/mesh windows via its `group_ep` / `group_tp` `hcomm_info` handle. **Never share the same `ProcessGroupHCCL` instance between standard `dist.all_to_all_single` / `dist.all_reduce` calls and MC2 operators.** Create a dedicated `mc2_group = dist.new_group(ranks, backend="hccl")` and extract its `hcomm_info = mc2_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)`.
   - **Concurrent Stream Rule:** Two NPU streams executing collectives simultaneously on the *same* `ProcessGroupHCCL` serialize or deadlock on shared HCCL notify registers; instantiate separate `dist.new_group` handles for primary and secondary overlap streams.
   - **Singleton Group Sentinel:** When `group_size == 1`, short-circuit collective creation and return a lightweight no-op sentinel instead of initializing a 1-rank HCCL communicator.

2. **Avoid `RowParallelLinear` bias multiplication and redundant `torch.zeros` before collectives (`tensor_prep_before_hccl_collective.md`).**
   - In `RowParallelLinear` (`Y = sum_r (X_r W_r) + b`), never add the bias `b` inside the local matmul before `dist.all_reduce(Y, group=tp_group)` (which sums `b` across `tp_size` ranks, producing `tp_size * b`). Either add `b` *after* `all_reduce` or divide `b / tp_size` before `all_reduce`.
   - When allocating destination buffers for `all_gather_into_tensor` or `reduce_scatter_tensor` that completely overwrite the output tensor, allocate with `torch.empty(...)` instead of `torch.zeros(...)` to save an extra full-tensor Vector zero-fill kernel launch.

3. **Match MoE dispatch/combine communication quantization to activation and SoC capabilities (`moe_dispatch_combine_comm_quant.md`).**
   - Set `dispatch` quantization mode (`quant_mode`) to match the expert GMM input dtype (`INT8` dynamic quant in W8A8/W4A8 so `dispatch` transfers `1-byte` INT8 tokens + FP32 scale instead of `2-byte` BF16).
   - On Ascend 950 (`950PR`/`950DT`), enable `MXFP8` communication quantization on `combine` as well to halve return-path bandwidth.
