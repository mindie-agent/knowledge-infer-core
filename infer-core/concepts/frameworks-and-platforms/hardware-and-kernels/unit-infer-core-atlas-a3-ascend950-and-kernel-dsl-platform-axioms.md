---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/frameworks-and-platforms/hardware-and-kernels
evidence_strength: hardware_spec
id: unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
inherits_from: []
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/platforms/atlas_a3.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/platforms/ascend_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/ascendc.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/pypto.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/tilelang.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/torchair.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/frameworks/autofuse.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/quantization/fp8_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/quantization/mxfp8_a8w8.md
  source_digest: 4361535e38e21d6dd1225c9853fec4d8ec1fa7b079807f87f8a7b486b54b9c56
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/platforms/atlas_a3.md
  source_type: cannbot_platform_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend platform hardware contracts (Atlas A2, Atlas A3, Ascend 950) and kernel/compiler
  DSL boundaries (AscendC, PyPTO, TileLang, TorchAir, AutoFuse)
triggers:
  error_signatures:
  - FP8 or MXFP8/MXFP4 kernel throws RuntimeError on Ascend910B or Ascend910_93 because
    native FP8/microscaling requires Ascend 950
  - PyPTO operator fails in eager or ACLGraph mode because PyPTO only supports TorchAir
    GE Graph lowering
  - TileLang operator fails under TorchAir GE Graph because TileLang only supports
    eager and ACLGraph capture
  keywords:
  - Atlas_A3
  - Ascend910_93
  - Ascend_950
  - 950PR
  - 950DT
  - AscendC
  - PyPTO
  - TileLang
  - TorchAir
  - AutoFuse
  - MXFP8
  - HiFloat8
---
# Ascend platform hardware contracts (Atlas A2, Atlas A3, Ascend 950) and kernel/compiler DSL boundaries (AscendC, PyPTO, TileLang, TorchAir, AutoFuse)

## Condition

Applies when selecting quantization formats, collective topologies, custom-operator DSLs, and graph-compilation backends across Ascend NPU generations (`Atlas A2 / Ascend910B`, `Atlas A3 / Ascend910_93`, `Ascend 950 / 950PR / 950DT`, `Ascend 310P`).

## Constraint

1. **SoC Hardware Architecture & Precision Matrix (`atlas_a3.md`, `ascend_950.md`, `fp8_attention.md`, `mxfp8_a8w8.md`):**
   - **Atlas A2 (`Ascend910B`):** Single-die NPU with split AIC (Cube) and AIV (Vector) cores (`1:2` AIC:AIV ratio), `64 GiB` HBM, intra-server HCCS ring/mesh. Native compute types: `FP16`, `BF16`, `INT8`, `INT4` (`W4A8` / `W4A16` via Cube/Vector dequant). Does **not** support native `FP8` (`E4M3`/`E5M2`) or `MXFP8`/`MXFP4`.
   - **Atlas A3 (`Ascend910_93`):** Dual-die package (`2` logical NPU dies per physical card, `16` dies per 8-card node, `Phy-ID 0..15`) connected via high-bandwidth intra-package die-to-die and scale-out SuperPod high-radix HCCS fabric, enabling low-latency all-to-all EP (`MC2` / `SHMEM`). Native compute types match A2 (`FP16`, `BF16`, `INT8`, `INT4`, `C8`).
   - **Ascend 950 (`950PR` / `950DT`):** Next-generation architecture adding hardware **FP8 (`E4M3`/`E5M2`)**, **HiFloat8 (`HiF8`)**, and **Microscaling (`MXFP8` E4M3/E5M2 + E8M0 block scale, `MXFP4` E2M1 + E8M0 block scale)** Cube instructions (`npu_dynamic_mx_quant`, per-block FP8 FlashAttention, and MXFP8 MoE combine collectives).

2. **Custom Operator DSL & Graph Backend Compatibility Matrix (`ascendc.md`, `pypto.md`, `tilelang.md`, `torchair.md`, `autofuse.md`):**
   - **AscendC (`ops/ascendc`):** Native C++ AIC/AIV pipeline DSL (`CopyIn -> Compute -> CopyOut` over GM/L1/L0A/L0B/L0C/UB queues). Supports **all** execution modes: Eager, `ACLGraph` (`npugraph_ex`), and `TorchAir` GE Graph (when compiled with `--op_relocatable_kernel_binary=true`, also eligible for `SuperKernel` task-scheduler fusion).
   - **PyPTO (`Parallel Tensor Operator`):** Tile/tensor-level graph DSL that lowers directly into CANN GE IR. **Supports `TorchAir` GE Graph mode ONLY** (`mode="max-autotune"` / GE backend); cannot be invoked in Eager mode or `ACLGraph` capture.
   - **TileLang (`TileLang-Ascend`):** Pythonic tile-programming DSL lowering to AscendC source. **Supports Eager and `ACLGraph` (`npugraph_ex`) ONLY**; lacks GE IR converter registration for `TorchAir` GE Graph mode.
   - **AutoFuse (`TorchInductor-NPU`):** Subgraph compiler that fuses pointwise/reduction/norm chains via symbolic shape analysis (`torch.compile(backend="inductor")` with AutoFuse backend). Best suited for high-pointwise-density subgraphs with static or bounded symbolic shapes.
