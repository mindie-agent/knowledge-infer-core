---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/frameworks-and-platforms/parallelism-topologies
evidence_strength: software_contract
id: unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/sp_tp_sp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/context_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/zigzag_cp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/ring_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/ulysses.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/kvp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/expert_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/embedding_tp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/cfg_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/vae_patch_parallel.md
  source_digest: 7f2fc6c0da1ef933205630f010d9326e957feaaaf98de1ac5d0256df29253a0d
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/parallelism/sp_tp_sp.md
  source_type: cannbot_concept_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: 'Parallelism topology invariants on Ascend NPU: TP, DP, EP, ZigZag/Ring/Ulysses
  CP, SP-TP-SP, KVP, and CFG/VAE Patch Parallelism'
triggers:
  error_signatures:
  - Naive contiguous sequence slicing in causal prefill CP leaves early ranks idle
    while the final rank computes full context
  - MLA AllReduce across TP ranks during long prefill saturates HCCS bandwidth when
    SP-TP-SP ReduceScatter/AllGather is omitted
  - Ulysses sequence parallel fails when attention head count is not divisible by
    ulysses_size
  keywords:
  - SP-TP-SP
  - ZigZag_CP
  - Ring_Attention
  - Ulysses
  - KV_Parallel
  - CFG_Parallel
  - VAE_Patch_Parallel
  - Embedding_TP
---
# Parallelism topology invariants on Ascend NPU: TP, DP, EP, ZigZag/Ring/Ulysses CP, SP-TP-SP, KVP, and CFG/VAE Patch Parallelism

## Condition

Applies when designing multi-card / multi-node parallelism topologies for LLM, MoE, and multimodal DiT inference on Ascend NPU clusters.

## Constraint

1. **Sequence & Context Parallelism Family (`context_parallel.md`, `zigzag_cp.md`, `ring_attention.md`, `ulysses.md`, `kvp.md`):**
   - **ZigZag-CP (Load-Balanced Causal Prefill):** Partitioning a causal sequence of length $S$ into $2P$ chunks and assigning chunk pair $(r,\; 2P - 1 - r)$ to rank $r \in [0, P-1]$ guarantees every rank computes an identical number of causal token pairs $S^2 / (2P)$.
   - **Ring Attention CP:** Rotates $K, V$ shards around a P2P ring overlapped with local chunk attention, merging partial outputs via online-softmax log-sum-exp (`LSE`):
     $$m_{\text{new}} = \max(m_1, m_2), \quad O_{\text{new}} = \frac{e^{m_1 - m_{\text{new}}} O_1 + e^{m_2 - m_{\text{new}}} O_2}{e^{m_1 - m_{\text{new}}} + e^{m_2 - m_{\text{new}}}}$$
   - **Ulysses SP (`ulysses.md`):** Transposes `[B, S/P, H, D]` to `[B, S, H/P, D]` via `AllToAll` before attention and inverse `AllToAll` after attention. Requires `H % P == 0` unless Unaligned Attention Head Adaptation (`UAA`) pads $H$.
   - **KV Parallel (`KVP` — `kvp.md`):** During long-context decode where $q_{\text{len}} = 1$ cannot be sequence-split, `KVP` shards the historical KV cache along the sequence axis across $P$ ranks; each rank computes partial attention + `LSE` over its local KV slice and merges across the KVP group.

2. **`SP-TP-SP` Hybrid Prefill Pattern for MLA (`sp_tp_sp.md`):**
   - In MLA prefill, keeping hidden states sequence-sliced (`[S/P, D]` — Sequence Parallel) across RMSNorm and latent compression (`q_a_proj`, `kv_a_proj`), performing `AllGather` only on the compact latent vectors (`1536` / `512` dims instead of `7168` hidden dims) before TP-sharded head attention, and ending `o_proj` with `ReduceScatter` (`SP -> TP -> SP`) cuts collective communication volume by $> 60\%$ compared to full-hidden-state `AllReduce`.

3. **CFG Parallel & VAE Patch Parallel for Diffusion Models (`cfg_parallel.md`, `vae_patch_parallel.md`):**
   - **CFG Parallel (`cfg_size = 2`):** Assigns positive (`cond`) and negative (`uncond`) classifier-free guidance branches to two parallel rank groups and exchanges predicted noise tensors via a single `AllGather` before computing $\epsilon = \epsilon_{\text{uncond}} + w (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$.
   - **VAE Patch Parallel:** Splits spatial `[H, W]` tiles across ranks with halo boundary exchange for Conv2d/Conv3d layers.
