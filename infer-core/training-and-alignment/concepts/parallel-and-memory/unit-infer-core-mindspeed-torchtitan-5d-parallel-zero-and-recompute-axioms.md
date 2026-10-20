---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/training-and-alignment/concepts/parallel-and-memory
evidence_strength: software_contract
id: unit-infer-core-mindspeed-torchtitan-5d-parallel-zero-and-recompute-axioms
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/torchtitan_5d_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/dualpipev.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/fsdp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/context_parallel_train.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/custom_cp_dsa.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/quantization/hif8.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/recomputation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/optimizer_memory.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/chunk_loss.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/swap_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/moe_comm_overlap.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-train-oom-analysis/SKILL.md
  source_digest: 6b918115bffa0531bfceb639c830e2fe45ba4ffc56c68a04bb94a13218e47fda
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/training/concepts/parallelism/torchtitan_5d_parallel.md
  source_type: cannbot_training_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: 'Ascend LLM/MoE training invariants: MindSpeed & TorchTitan 5D Parallelism,
  DualPipeV zero-bubble, FSDP2/LayerZeRO, selective recomputation, and HiF8/MXFP8'
triggers:
  error_signatures:
  - Large-vocabulary LM Head cross-entropy materializes full [B, S, V] FP32 logits
    tensor and triggers training HBM OOM
  - Pipeline Parallel 1F1B schedule suffers high bubble ratio on large MoE pretraining
    without DualPipeV dW split
  - MoE AllToAll dispatch/combine blocks backward pass compute when cross-microbatch
    overlap is disabled
  keywords:
  - torchtitan_5d_parallel
  - MindSpeed
  - DualPipeV
  - FSDP2
  - LayerZeRO
  - Reuse-FP32-Param
  - Chunk_Loss
  - Swap_Attention
  - HiF8
  - SFAG
  - SLIG
  - SMLA
---
# Ascend LLM/MoE training invariants: MindSpeed & TorchTitan 5D Parallelism, DualPipeV zero-bubble, FSDP2/LayerZeRO, selective recomputation, and HiF8/MXFP8

## Condition

Applies to pretraining, continued pretraining, and SFT of large dense, MoE (`DeepSeek-V3/V3.2/V4`, `Qwen3-MoE`), hybrid linear-attention (`Gated DeltaNet`, `Mamba2`), and multimodal models using `MindSpeed`, `MindSpeed-MM`, or `torchtitan-npu` on Ascend NPU clusters.

## Constraint

1. **5D Hybrid Parallelism & `DualPipeV` Zero-Bubble Schedule (`torchtitan_5d_parallel.md`, `dualpipev.md`, `context_parallel_train.md`, `custom_cp_dsa.md`):**
   - `MindSpeed` and `torchtitan-npu` compose **5D Parallelism**: `[DP/FSDP2, TP/TP-2D, PP/VPP/DualPipeV, CP (Ulysses/Ring/Hybrid/Custom-DSA), EP]`.
   - **DualPipeV:** Folds a $2P$-stage bidirectional V-shaped pipeline onto $P$ physical ranks and splits backward into activation gradient ($dX$) and weight gradient ($dW$), scheduling compute-only $dW$ tiles to fill inter-stage communication bubbles and overlapping forward/backward MoE `AllToAll` across microbatches (`moe_comm_overlap.md`, `coc.md`).
   - **DSA/GDN Training Kernels (`sfag.md`, `slig.md`, `sasg.md`, `smla.md`, `gdn_chunk_kernels.md`):** Training DSA and DeepSeek-V4 requires dedicated backward kernels—`SFAG` (`SparseFlashAttentionGrad`), `SLIG` (`SparseLightningIndexerGradKLLoss` for KL-divergence alignment of the Lightning Indexer), and `SASG` (`SparseAttnSharedkvGrad`)—plus Triton/AscendC chunk-parallel WY kernels (`solve_tril`, `chunk_delta_h`, `chunk_o`) for Gated DeltaNet.

2. **Four-Pillar HBM Memory Optimization (`recomputation.md`, `optimizer_memory.md`, `chunk_loss.md`, `swap_attention.md`, `swap_optimizer.md`, `model-train-oom-analysis/SKILL.md`):**
   - **Granular Recomputation:** Instead of full-layer activation checkpointing, selectively recompute high-memory/low-FLOP sub-ops (`SwiGLU`, `RMSNorm`, attention softmax, MoE token permute) and assign per-PP-stage recompute layer counts proportional to stage memory pressure (`1F1B` early stages hold more in-flight microbatches than late stages).
   - **`Chunk Loss`:** Never materialize full `[B, S, Vocab]` logits in HBM for large vocabularies ($V > 128\text{K}$); slice hidden states along the token axis into chunks and fuse `LM_Head` projection + cross-entropy loss per chunk.
   - **Optimizer State Sharding & Reuse:** Use `FSDP2` (`DTensor` per-parameter sharding), `LayerZeRO`, `Reuse-FP32-Param` (aliasing BF16 param bytes into the upper half of FP32 master weights), and asynchronous `Swap Attention` / `Swap Optimizer` (`SmartSwap` to pinned Host RAM).

3. **HiFloat8 (`HiF8`) & `MXFP8` Low-Precision Pretraining (`hif8.md`, `deepseek_v3_low_precision_pretrain.md`):**
   - On Ascend 950 (`950DT`), low-precision pretraining uses `HiF8` / `MXFP8` for forward/backward GEMMs while keeping master weights, router logits, norms, and main-stream residual accumulations in `FP32`/`BF16`.
