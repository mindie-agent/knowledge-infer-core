---
category: workflow
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - DeepSeek-V4-Flash TileLang + Inductor inference
  - Single-NPU large MoE inference with host DRAM offload in SGLang
domain_path: infer-core/models/deepseek-recipes/v4-flash-tilelang-offload
evidence_strength: reproduced_empirical
id: unit-deepseek-v4-flash-tilelang-inductor-and-single-npu-moe-offload
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/deepseek_v4_flash_tilelang_and_inductor_af/README.md
  source_digest: 7e8659ef49578885aa3a46993ae649a9427251a633a3027df069617236f0c42e
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Workflow combining TileLang custom kernels with TorchInductor AutoFuse for
  DeepSeek-V4-Flash, plus the SGLang single-NPU MoE offload architecture using hot-expert
  HBM caching and async PCIe prefetching.
tier: L3
title: DeepSeek-V4-Flash TileLang + Inductor AutoFuse and SGLang Single-NPU MoE Offload
  Recipe
triggers:
  error_signatures:
  - deepseek v4 flash tilelang inductor autofuse
  - dsv4 flash single npu moe offload
  keywords:
  - deepseek v4 flash tilelang inductor autofuse
  - dsv4 flash single npu moe offload
  - sglang single card moe cpu offload
  - pinned memory expert weight prefetch
  - hybrid cpu npu moe execution
  - deepseek-v4-flash
  - tilelang
  - inductor
  - autofuse
  - sglang
  - moe-offload
  - single-npu
  - pcie-prefetch
---
## 1. DeepSeek-V4-Flash TileLang + Inductor AutoFuse Workflow

`models/deepseek_v4_flash_tilelang_and_inductor_af` demonstrates a dual-tier compilation strategy on Ascend NPU:
1. **Core Compute Operators in TileLang-Ascend**:
   - Implements compute-intensive Attention (`Sparse Flash Attention`), `mHC` manifold hyper-connections, `INT8 GEMM`, and dynamic activation quantization in TileLang using explicit `T.Parallel`, UB/L1 tiling, and automatic Cube/Vector (`CV`) pipeline splitting.
2. **Glue & Epilogue Subgraphs via TorchInductor AutoFuse (`inductor_af`)**:
   - Rather than writing hand-tuned AscendC kernels for every residual add, rotary slicing, and norm epilogue, `torch.compile` with the Inductor AutoFuse backend fuses surrounding PyTorch operators and stitches TileLang custom ops into a single captured execution graph.
   - Avoids graph breaks by registering fake tensor meta-functions and custom op schemas (`torch.library.custom_op`) for all TileLang kernels.

## 2. SGLang Single-NPU MoE Offload Architecture (`dsv4_flash_single_card_design.md`)

When deploying a large MoE model (`DeepSeek-V4-Flash`) on a **single Ascend NPU** (e.g., `64 GB` HBM on Atlas A2), non-expert weights (Embedding, Attention, Shared Experts, Norms, LM Head) plus KV Cache fit in HBM, while routed MoE expert weights exceed single-card memory.

### Memory Tiering Layout
- **NPU HBM Resident**:
  - All Attention weights (`W_qkv`, `W_o`), MLA projections, Shared Expert weights, Router gates, and Paged KV Cache.
  - **Hot-Expert Cache Slot Pool**: A configurable fraction of routed experts per layer cached in NPU HBM alongside a double-buffered staging buffer (`Buffer A` / `Buffer B`) for on-demand expert DMA transfers.
- **Host Pinned DRAM Resident**:
  - Full routed expert weights stored in page-locked (`pin_memory=True`), 64-byte aligned host memory bound to the local NUMA node of the NPU PCIe root complex.

### Execution & Overlap Pipeline
1. **Early Speculative Router Lookahead**:
   - Uses hidden states from layer $l-1$ (or early pre-attention hidden states of layer $l$) to predict active top-$k$ experts for layer $l$ ahead of time.
2. **Asynchronous Host-to-Device (`H2D`) PCIe Prefetch**:
   - On a dedicated `H2D` copy stream, issues `non_blocking=True` DMA transfers for any top-$k$ experts of layer $l$ that miss in the NPU Hot-Expert Cache while the NPU Cube/Vector units execute Layer $l$ Attention and Shared Expert projections.
3. **Cache Replacement & Quantized Transfer**:
   - Compresses offloaded expert weights to `INT8` / `INT4` (`W8A8` or `W4A16`) so PCIe transfer volume per activated expert is cut by 2x–4x, allowing PCIe Gen4/Gen5 bandwidth to keep pace with single-batch decode latency.
4. **Stream Synchronization Contract**:
   - Records an `npu.Event` on the `H2D` prefetch stream upon completion of missing-expert transfers and makes the compute stream wait on that event immediately before launching `npu_grouped_matmul`.

## Verification Checklist

- Confirm `numa_node` affinity binds the Python process and pinned host tensor allocator to the CPU socket directly attached to the target NPU card.
- Verify zero synchronous `dtoh` (`item()` / `.cpu()`) calls occur inside the expert routing loop by keeping hit/miss index resolution on-device or in pre-pinned async host buffers.
