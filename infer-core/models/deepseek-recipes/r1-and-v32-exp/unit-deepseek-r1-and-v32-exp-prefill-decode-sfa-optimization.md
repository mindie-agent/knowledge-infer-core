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
  - DeepSeek-R1 671B prefill/decode
  - DeepSeek-V3.2-Exp sparse attention inference
domain_path: infer-core/models/deepseek-recipes/r1-and-v32-exp
evidence_strength: reproduced_empirical
id: unit-deepseek-r1-and-v32-exp-prefill-decode-sfa-optimization
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_r1/deepseek_r1_decode_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_r1/deepseek_r1_prefill_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md
  source_digest: 598e304b95fb94d50fa4d379cbf3e57d5a9b0c69949f0fda14a3265760d3bea2
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_r1/deepseek_r1_decode_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: End-to-end inference recipe for DeepSeek-R1 and DeepSeek-V3.2-Exp covering
  MLA absorption, LightningIndexer + Sparse Flash Attention (SFA), dual-microbatch
  ping-pong communication hiding, Round-Robin sequence packing, and W8A8/C8 quantization.
tier: L2
title: DeepSeek-R1 and DeepSeek-V3.2-Exp Prefill/Decode MLA, SFA, and Multi-Stream
  Pipeline Playbook
triggers:
  error_signatures:
  - deepseek r1 prefill decode optimization
  - deepseek v3.2 exp sparse flash attention
  keywords:
  - deepseek r1 prefill decode optimization
  - deepseek v3.2 exp sparse flash attention
  - lightning indexer topk 2048 sfa
  - dual microbatch communication computation overlap
  - mla absorb page attention c8 kvcache
  - amla aic aiv parallel execution
  - deepseek-r1
  - deepseek-v3.2-exp
  - mla
  - sfa
  - lightning-indexer
  - prefill-decode
  - w8a8
  - c8
  - ep
  - mc2
---
## Overview & Core Optimization Architecture

DeepSeek-R1 (`671B` total parameters, `37B` active per token, `61` layers with `3` dense layers and `58` MoE layers, `256` routed experts + `1` shared expert, `top-8` routing) and DeepSeek-V3.2-Exp require distinct execution pipelines for Prefill (compute-bound, high token count) and Decode (memory-bandwidth and collective-latency bound).

## 1. DeepSeek-R1 Prefill Optimization Pipeline

1. **Hybrid Parallelism & Dynamic Sequence Load Balancing**:
   - **Attention**: Uses hybrid `DP + TP` with Sequence Parallelism (`SP`) on normalization and residual streams to eliminate redundant activation memory and reduce `AllReduce` volume into `ReduceScatter` / `AllGather`.
   - **MoE**: Uses hybrid `EP + TP` (`MC2` fused dispatch/combine or two-stage `AlltoAll` across nodes and intra-node rings).
   - **Round-Robin Pack Splitting**: For ultra-long contexts (`32K`–`128K`), causal masking creates a triangular compute load where trailing chunks perform 3x more work than leading chunks. Splitting a single request into `2 * CP` chunks and pairing chunk `i` with chunk `(2*CP - 1 - i)` equalizes FLOPs across all Context Parallel (`CP`) ranks.
2. **Dual-Microbatch Ping-Pong & Computation-Communication Overlap**:
   - Splits incoming prefill batches into two microbatches (`MB0`, `MB1`) on separate NPU execution streams (`Stream 0` compute, `Stream 1` communication).
   - Overlaps `MB0` MoE `AlltoAll` dispatch/combine with `MB1` Attention / Shared Expert computation.
   - Overlaps shared expert `MatMul` with routed expert dispatch, and folds `Cast`, `Add`, and `SwiGLU` into adjacent communication windows.
3. **Attention & MoE Operator Fusion (`FlashComm` & Quantized Kernels)**:
   - Replaces standard projections with `Add_RmsNorm_DynamicQuant`, `Matmul_AllReduce_AddRmsNorm`, and `GMM_Swiglu_Quant`.
   - Applies `W8A8` dynamic per-token quantization on linear/MoE layers and `C8` (`INT8`) quantization on KV cache blocks.

## 2. DeepSeek-R1 Decode Optimization Pipeline

1. **MLA Matrix Absorption & `AMLA` Cube/Vector Co-Execution**:
   - Absorbs $W^{UK}$ into $W^Q$ and $W^{UV}$ into $W^O$ during decode so the compressed latent vector $c_t^{KV}$ (`dim=512`) and RoPE key $k_t^R$ (`dim=64`) are read directly from paged KV cache without expanding full per-head K/V tensors.
   - **`AMLA` (Ascend MLA)**: Decouples AIC (Cube matrix multiplication) and AIV (Vector softmax/rescaling) pipelines using tile-level ping-pong buffers, raising single-step decode bandwidth utilization.
2. **Large-EP `MC2` Dispatch/Combine & Dynamic EPLB**:
   - Uses `torch_npu.npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` over HCCS/RDMA to fuse quantization, token permutation, and expert combine.
   - Enables Expert Parallel Load Balancing (`EPLB`) with redundant hot-expert placement to prevent straggler ranks from stalling `MC2` completion.
3. **MTP Speculative Decoding (`num_speculative_tokens=1..3`)**:
   - Expands decode query length per request from `1` to `1 + K` tokens, increasing arithmetic intensity in both `MLA` and `GroupedMatmul` (`GMM`) into the compute-efficient region while sharing KV cache reads across draft verification tokens.

## 3. DeepSeek-V3.2-Exp Sparse Attention (`LightningIndexer` + `SFA`)

1. **Two-Stage Sparse Attention Mechanism**:
   - **Stage 1 (`LightningIndexer`)**: Computes lightweight index scores between low-rank query/key projections (`index_n_heads=64`, `index_head_dim=128`) and selects `index_topk=2048` key token indices per query token via fused `npu_lightning_indexer`.
   - **Stage 2 (`Sparse Flash Attention` / `SFA`)**: Gathers or masks KV cache blocks using the `2048` selected indices and executes `npu_sparse_flash_attention`, reducing attention complexity from $O(S^2)$ to $O(S \cdot k)$ for `128K` context lengths.
2. **Ascend 950PR/DT vs. Atlas A2/A3 Execution**:
   - On Atlas A2/A3 (`8.5.0`), `32K`–`128K` inference with `DP2 + TP8 + EP16` and `MTP=1..3` achieves `4388 TPS` at `30 ms` TPOT (`4K+1K` workload) and `2397 TPS` at `50 ms` TPOT (`16K+1K` workload).
   - On Ascend 950PR/DT, `LightningIndexer` uses hardware histogram Top-K acceleration and `FP8` tensor core execution.

## Verification Checklist

- Confirm `kv_lora_rank=512`, `qk_rope_head_dim=64`, and `index_topk=2048` match model configuration before enabling `SFA`.
- Verify `torch_npu` graph capture (`aclgraph` / `torchair`) records fixed workspace addresses for `MC2` and `LightningIndexer` without dynamic shape recompilation.
