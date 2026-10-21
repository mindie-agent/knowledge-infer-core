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
  - Qwen3.5-MoE, Qwen3-235B-A22B / 30B-A3B, Qwen3-Next-80B-A3B, and Qwen2.5/Qwen3
    Dense inference
domain_path: infer-core/models/kimi-and-qwen-recipes/qwen35-qwen3-moe-and-dense
evidence_strength: reproduced_empirical
id: unit-qwen35-moe-qwen3-moe-qwen3-next-and-dense-optimization-recipe
inherits_from:
- unit-infer-core-gated-deltanet-mamba2-mhc-and-hybrid-moe-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/qwen3_5/qwen3_5_moe_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/qwen3_moe/qwen3_moe_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/qwen/qwen_dense_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/integration/sglang/qwen3-next/qwen3_next_optimization.md
  source_digest: 9eb14fe7d621f5a9fd457364e778b3e41c2e00a1889c825e434f712f7d1a2842
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/qwen3_5/qwen3_5_moe_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: 'Unified optimization recipe for the Qwen family on Ascend NPU: Qwen Dense
  fused QKV/SwiGLU/aclgraph, Qwen3-MoE and Qwen3.5-MoE EP+GMM pipelines, and SGLang
  Qwen3-Next hybrid Gated DeltaNet + Full Attention execution.'
tier: L2
title: Qwen3.5-MoE, Qwen3-MoE, Qwen3-Next Hybrid Linear Attention, and Qwen Dense
  Playbook
triggers:
  error_signatures:
  - qwen3.5 moe optimization ascend
  - qwen3 moe optimization recipe
  keywords:
  - qwen3.5 moe optimization ascend
  - qwen3 moe optimization recipe
  - qwen3 next sglang gated deltanet
  - qwen dense optimization torch npu
  - hybrid linear attention mamba deltanet kv cache
  - qwen3.5-moe
  - qwen3-moe
  - qwen3-next
  - qwen-dense
  - gated-deltanet
  - hybrid-attention
  - sglang
  - gmm
---
## 1. Qwen Dense Optimization Baseline (`qwen_dense_optimization.md`)

1. **Weight Concatenation & Tensor Parallelism**:
   - Merges `q_proj`, `k_proj`, `v_proj` into a single `qkv_proj` column-parallel linear layer (`num_key_value_heads_per_rank = max(num_kv_heads // tp_size, 1)`).
   - Merges `gate_proj` and `up_proj` into `gate_up_proj` followed by `torch_npu.npu_swiglu(x, dim=-1)`.
2. **Core Operator Replacements**:
   - Per-head `q_norm` / `k_norm` (in Qwen3) + RoPE via `torch_npu.npu_rms_norm` and `torch_npu.npu_rotary_mul` (or `npu_apply_rotary_pos_emb`).
   - Residual + norm via `torch_npu.npu_add_rms_norm`.
   - Static Paged/Contiguous KV Cache updated in-place via `torch_npu.scatter_update_` and evaluated with `torch_npu.npu_fused_infer_attention_score` (`FIA`).

## 2. Qwen3-MoE & Qwen3.5-MoE Optimization (`qwen3_moe` & `qwen3_5_moe`)

1. **MoE Routing & GroupedMatmul (`GMM`) Pipeline**:
   - Replaces Python loops over `128` experts (`top-8`) with token sorting and grouped GEMM:
     - `npu_moe_gating_top_k_softmax` computes top-$k$ expert weights and indices.
     - `npu_moe_init_routing_v2` permutes input tokens into contiguous expert groups (`expanded_x`, `expert_tokens_count_or_cumsum`).
     - `npu_grouped_matmul` executes `gate_up` projection (`split_item=2`), followed by `npu_swiglu`, a second `npu_grouped_matmul` for `down_proj`, and `npu_moe_finalize_routing` to unpermute and accumulate weighted expert outputs.
2. **Hybrid `DP + TP + EP` Deployment**:
   - Small batch / low latency: `TP8` with `AllReduce` or `EP8` with `MC2` (`npu_moe_distribute_dispatch_v2` / `combine_v2`).
   - High throughput on Atlas A3 (`16` dies): `DP + EP16` with `W8A8` dynamic quantization (`npu_dynamic_quant` + quantized `GMM`).

## 3. Qwen3-Next Hybrid Architecture in SGLang (`qwen3_next_optimization.md`)

`Qwen3-Next` (`80B-A3B`) interleaves **Gated DeltaNet (Linear Attention / Recurrent State)** layers (`3` out of every `4` layers) with **Full GQA Attention** layers (`1` out of every `4` layers) plus fine-grained MoE (`512` experts) and `MTP`:
1. **Heterogeneous KV & Recurrent State Pool Management**:
   - Full Attention layers (1/4 of layers) use standard paged KV cache blocks (`page_size=128`).
   - Gated DeltaNet layers (3/4 of layers) maintain a fixed-size recurrent state tensor $(B, H, D_k, D_v)$ plus a short `conv1d` rolling state (`kernel_size=4`) per sequence slot instead of growing with sequence length $S$.
2. **Fused Linear Attention Kernels (`Flash KDA` / Chunked FLA)**:
   - Prefill uses chunkwise parallel Gated DeltaNet (`chunk_gated_delta_rule` / `flash_kda.py`) to maximize Cube utilization.
   - Decode fuses `causal_conv1d_update` and recurrent state update (`fused_sigmoid_gating_delta_rule_update`) with `MTP` multi-token draft verification so `1 + K` draft tokens update the recurrent state in a single kernel launch.

## Verification Checklist

- Confirm Qwen3-Next separates page-table allocation (only `num_full_attn_layers`) from recurrent state slot allocation (`num_deltanet_layers`) to avoid 4x KV cache over-allocation.
- Verify `q_norm` and `k_norm` are applied **before** `npu_rotary_mul` on Qwen3 / Qwen3.5 models.
