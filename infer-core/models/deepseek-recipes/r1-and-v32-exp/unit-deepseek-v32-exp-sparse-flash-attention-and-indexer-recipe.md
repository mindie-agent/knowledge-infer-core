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
  - DeepSeek-V3.2-Exp long-context 32K-128K sparse attention inference
domain_path: infer-core/models/deepseek-recipes/r1-and-v32-exp
evidence_strength: reproduced_empirical
id: unit-deepseek-v32-exp-sparse-flash-attention-and-indexer-recipe
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md
  source_digest: 727f6e094d799cb4e674ee7959c34084d86e2b2a29b18c401874ee6538a2c216
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Dedicated DeepSeek-V3.2-Exp inference guide detailing the LightningIndexer
  + Sparse Flash Attention (SFA) two-stage execution graph, KV cache block indexing,
  MTP=1..3 speculative verification, and Atlas A2/A3/950 deployment benchmarks.
tier: L2
title: DeepSeek-V3.2-Exp Sparse Flash Attention (SFA), LightningIndexer, and MTP Serving
  Recipe
triggers:
  error_signatures:
  - deepseek v3.2 exp inference guide
  - npu_lightning_indexer npu_sparse_flash_attention
  keywords:
  - deepseek v3.2 exp inference guide
  - npu_lightning_indexer npu_sparse_flash_attention
  - deepseek v3.2 exp 32k 128k tpot
  - index_n_heads 64 index_head_dim 128 index_topk 2048
  - deepseek-v3.2-exp
  - sfa
  - lightning-indexer
  - mtp
  - sparse-attention
  - atlas-a3
  - ascend-950
---
## 1. DeepSeek-V3.2-Exp Sparse Attention Architecture (`deepseek_v3.2_exp_inference_guide.md`)

`DeepSeek-V3.2-Exp` augments Multi-head Latent Attention (`MLA`) with **Dynamic Sparse Attention (`DSA`)** composed of `LightningIndexer` + `Sparse Flash Attention (SFA)` to eliminate quadratic $O(S^2)$ prefill and linear $O(S)$ decode KV bandwidth bottlenecks at `32K`–`128K` context lengths:
1. **Indexer Projection Path**:
   - Projects hidden states into lightweight indexer queries $Q^{\text{idx}}$ (`index_n_heads=64`, `index_head_dim=128`), indexer keys $K^{\text{idx}}$ (`1` head, `dim=128`), and per-head importance weights $W^{\text{idx}}$ (`[T, 64]`).
   - Applies partial RoPE (`qk_rope_head_dim=64`) to the rotary slice of $Q^{\text{idx}}$ and $K^{\text{idx}}$, caching $K^{\text{idx}}$ in a dedicated paged `indexer_k_cache`.
2. **`LightningIndexer` Kernel (`torch_npu.npu_lightning_indexer`)**:
   - Evaluates weighted ReLU dot-product scores across all historical tokens:
     $$I(t, s) = \sum_{h=1}^{64} w_{t,h}^{\text{idx}} \cdot \text{ReLU}\big(\langle q_{t,h}^{\text{idx}}, k_s^{\text{idx}} \rangle\big)$$
   - Selects the exact `index_topk = 2048` highest-scoring token indices per query token (`sparse_indices` of shape `[T, 1, 2048]`, `int32`).
3. **`Sparse Flash Attention` (`SFA` / `npu_sparse_flash_attention`)**:
   - Gathers only the `2048` selected latent KV entries (`kv_c` `512`-dim + `k_pe` `64`-dim) from the paged MLA cache for each query token and computes exact scaled dot-product attention over those `2048` tokens.
   - For sequences with $S \le 2048$, bypasses sparse selection or pads valid indices up to actual sequence length `actual_seq_lengths_kv`.

## 2. Parallelism, Quantization & MTP Co-Design

1. **Parallelism & Quantization Stack**:
   - Deployed with `DP2 + TP8 + EP16` on 16-rank Atlas A2/A3 (`CANN 8.5.0`) using `W8A8` dynamic quantization on linear/MoE layers and `C8` KV cache quantization.
2. **Multi-Token Prediction (`MTP = 1..3`) with `SFA`**:
   - During decode with `MTP=K`, each sequence verifies `1 + K` query tokens simultaneously (`q_len = 1 + K`).
   - `LightningIndexer` computes top-`2048` indices for each of the `1 + K` draft tokens while respecting causal draft masking (`sparse_mode=3`), and `SFA` processes the `1 + K` queries in a single fused kernel launch.
3. **Measured Performance Benchmarks**:
   - `4K input + 1K output`: `4388 TPS` total throughput at `30 ms` TPOT.
   - `16K input + 1K output`: `2397 TPS` total throughput at `50 ms` TPOT.

## Verification Checklist

- Confirm `indexer_k_cache` page table is updated alongside the main MLA KV cache on every prefill and decode step.
- Ensure causal masking inside `LightningIndexer` prevents query token $t$ from selecting future key indices $s > t$ during chunked prefill and MTP verification.
