---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/models-and-modules/linear-ssm-and-mhc
evidence_strength: software_contract
id: unit-infer-core-gated-deltanet-mamba2-mhc-and-hybrid-moe-axioms
inherits_from: []
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/gated_deltanet.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/gated_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/mhc.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/recurrent_gated_delta_rule.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/mambav2_rmsnormgated.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/kernels/mhc_hcpre_hcpost.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/qwen3_next.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/qwen3_5_moe.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/longcat_flash.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/gemma4_26b_a4b.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/gpt_oss.md
  source_digest: 42fa927021ab81f5af9083f8a5527f87010e99b9f5c2eae7f3773c8a28b01f73
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/modules/gated_deltanet.md
  source_type: cannbot_concept_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Architectural invariants of Gated DeltaNet (GDN), Mamba2 SSM, Manifold-Constrained
  Hyper-Connections (mHC), and hybrid MoE models
triggers:
  error_signatures:
  - Hybrid Gated DeltaNet + Full Attention model corrupts recurrent state S_t when
    prefill chunking or decode padding mutates state in place
  - mHC manifold-constrained hyper-connection residual stream diverges when HcPre/HcPost
    mixing matrices lose doubly-stochastic normalization
  - LongCat-Flash zero-compute identity experts or GPT-OSS clamped SwiGLU fail on
    generic MoE fused operators
  keywords:
  - Gated DeltaNet
  - recurrent_gated_delta_rule
  - mambav2_rmsnormgated
  - HcPre
  - HcPost
  - Manifold-Constrained Hyper-Connections
  - Qwen3-Next
  - Qwen3.5-MoE
  - LongCat-Flash
  - Gemma-4-26B-A4B
  - GPT-OSS
---
# Architectural invariants of Gated DeltaNet (GDN), Mamba2 SSM, Manifold-Constrained Hyper-Connections (mHC), and hybrid MoE models

## Condition

Applies to hybrid linear-recurrent + full-attention architectures (`Qwen3-Next`, `Qwen3.5-MoE`), Manifold-Constrained Hyper-Connection (`mHC`) backbones (`DeepSeek-V4`), and specialized MoE variants (`LongCat-Flash`, `Gemma-4-26B-A4B`, `GPT-OSS`) on Ascend NPU.

## Constraint

1. **Gated DeltaNet (GDN) & Mamba2 Recurrent State Dynamics (`gated_deltanet.md`, `recurrent_gated_delta_rule.md`, `mambav2_rmsnormgated.md`):**
   - `Qwen3-Next` and `Qwen3.5-MoE` interleave $3:1$ linear **Gated DeltaNet (GDN)** layers with standard Full/Gated Attention layers.
   - Each GDN layer maintains two fixed-size per-request recurrent states independent of sequence length: a causal 1D depthwise convolution buffer (`conv_state`) and a matrix-valued associative memory state $S_t \in \mathbb{R}^{H_v \times d_k \times d_v}$ updated via the gated delta rule:
     $$S_t = S_{t-1}\left(\alpha_t (I - \beta_t k_t k_t^\top)\right) + \beta_t k_t v_t^\top, \quad o_t = S_t^\top q_t$$
   - **Prefill vs. Decode Execution:** Prefill uses hardware-friendly chunkwise parallel WY-representation kernels (`chunk_gated_delta_rule`), whereas decode uses the fused in-place single-step kernel `recurrent_gated_delta_rule` followed by `mambav2_rmsnormgated` ($\text{RMSNorm}(x \odot \text{SiLU}(z))$).
   - **State Isolation Invariant:** Unlike paged KV caches where dummy tokens can write to a scratch slot without affecting earlier positions, `recurrent_gated_delta_rule` updates $S_t$ in place. Dummy/padded decode requests (`seq_len == 0` or idle DP ranks) must be routed to a dedicated scratch state slot (`slot_idx = 0` / zero-padded workspace) and must never mutate a live sequence's state slot.

2. **Manifold-Constrained Hyper-Connections (`mHC` — `mhc.md`, `mhc_hcpre_hcpost.md`):**
   - `mHC` generalizes standard single-stream residual connections into an $n$-stream hyper-residual state $X \in \mathbb{R}^{B \times S \times n \times D}$ governed by dynamic mixing weights constrained onto the Birkhoff polytope (doubly stochastic manifold via Sinkhorn projection in training).
   - At inference time, `HcPre` fuses multi-stream weighted aggregation before the Attention/MoE block and `HcPost` fuses stream expansion and residual addition after the block, avoiding intermediate $n \times D$ HBM round-trips.

3. **Specialized MoE & Gated-Attention Variants (`gated_attention.md`, `longcat_flash.md`, `gemma4_26b_a4b.md`, `gpt_oss.md`):**
   - **Gated Attention (`Qwen3-Next`):** Applies a learned elementwise sigmoid/SiLU gate to the attention head output before `o_proj`.
   - **LongCat-Flash / Lite:** Combines shortcut Attention-FFN Disaggregation (AFD) overlap with `copy_expert_num` zero-compute identity experts.
   - **Gemma-4-26B-A4B & GPT-OSS:** `Gemma-4` uses `GEGLU` (GELU-gated MLP) and soft-capped attention logits rather than standard `SiLU`; `GPT-OSS` uses clamped SwiGLU (`clamp(up, -limit, limit)`) and sliding-window + attention-sink biases.
