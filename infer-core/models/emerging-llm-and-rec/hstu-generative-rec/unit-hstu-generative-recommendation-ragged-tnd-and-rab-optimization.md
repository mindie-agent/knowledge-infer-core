---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - HSTU (Hierarchical Sequential Transduction Units) generative recommendation inference
domain_path: infer-core/models/emerging-llm-and-rec/hstu-generative-rec
evidence_strength: reproduced_empirical
id: unit-hstu-generative-recommendation-ragged-tnd-and-rab-optimization
inherits_from:
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/hstu/hstu_optimization.md
  source_digest: f99dcf76cd5bb8f42b4331d4b921ce95cc112e12c82f493f6fd2ac5c93d2ba5e
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/hstu/hstu_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Optimization recipe for HSTU generative recommendation models on Ascend NPU
  covering ragged TND variable-length sequence packing, Relative Attention Bias (RAB)
  vectorization, and fused UVQK + SiLU pointwise attention.
tier: L2
title: 'HSTU Generative Recommendation NPU Optimization: Ragged TND Packing, RAB Bias,
  and Fused Ops'
triggers:
  error_signatures:
  - hstu generative recommendation optimization
  - hierarchical sequential transduction units ascend
  keywords:
  - hstu generative recommendation optimization
  - hierarchical sequential transduction units ascend
  - hstu ragged tnd sequence packing
  - relative attention bias rab time position
  - silu pointwise aggregated attention hstu
  - hstu
  - generative-recommendation
  - ragged-tensor
  - tnd-layout
  - rab
  - relative-attention-bias
  - pointwise-attention
---
## 1. HSTU Architectural Bottlenecks on NPU

Unlike standard LLM decoders that use softmax dot-product attention over uniform token blocks, **HSTU (Hierarchical Sequential Transduction Units)** for large-scale generative recommendation processes highly skewed user interaction histories (`seq_len` ranging from `10` to `4096` events per user in the same batch):
1. **Pointwise Aggregated Attention (SiLU-Attention instead of Softmax)**:
   - Projects input $x$ through a unified linear layer into $U, V, Q, K = 	ext{Split}(	ext{SiLU}(W_{uvqk} x))$, computes attention weights $	ext{SiLU}(Q K^	op + 	ext{RAB}^{p,t}) / N$, multiplies by $V$, and gates the result via $	ext{Norm}(A(x) V) \odot U$.
2. **Padding Waste in `BSND` Layout**:
   - Padding skewed user history lengths to `max_seq_len` wastes 60%–85% of Cube FLOPs on zero-padded tokens.

## 2. Core NPU Optimization Strategies (`hstu_optimization.md`)

1. **Ragged `TND` Variable-Length Sequence Packing (`jagged` / `varlen` Execution)**:
   - Concatenates all user interaction sequences along the token dimension into a dense 3D tensor `[T_total, N, D]` (`TND` layout) accompanied by cumulative sequence offset array `seq_offsets` (`cu_seqlens` of shape `[B + 1]`).
   - Eliminates padding tokens across both linear projections (`W_uvqk`, `W_o`) and attention kernels.
2. **Vectorized Relative Attention Bias (`RAB`) Pre-Computation**:
   - HSTU injects relative position and timestamp bucket biases ($	ext{RAB}^{p,t}$) into attention scores.
   - **Optimization**: Replace per-layer Python loops and dynamic bucket lookups with a single vectorized pre-computation pass per batch that constructs the bucketed relative time/position bias tensor once on NPU and reuses it across all HSTU transduction layers.
3. **Fused `UVQK` Projection & Dedicated HSTU Attention Kernel**:
   - Fuses the four separate projections (`u_proj`, `v_proj`, `q_proj`, `k_proj`) into a single `uvqk_linear` GEMM + fused `SiLU` + split.
   - Invokes the Ascend fused HSTU attention operator (`hstu_dense_forward` / custom AscendC pointwise attention kernel supporting `TND` jagged offsets, `silu` activation, causal + target-aware masking, and `RAB` bias addition) instead of materializing the $[B, H, S, S]$ score matrix in HBM.
4. **Fused Norm + Gating (`Norm(attn_out) * u`) + Static Bucket Graph Capture**:
   - Fuses LayerNorm/RMSNorm and elementwise gating with $U$ prior to the output projection.
   - Buckets `T_total` into fixed padding tiers (`2K, 4K, 8K, 16K`) so `aclgraph` / `torchair` can replay without dynamic shape recompilation.

## Verification Checklist

- Confirm `seq_offsets` (`int32` / `int64`) is monotonic with `seq_offsets[0] == 0` and `seq_offsets[-1] == T_total`.
- Verify target-aware masking (preventing candidate items from attending to other candidate items while allowing full causal attention over historical interactions) matches the reference HSTU mask specification.
