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
  - Parallelism selection, KV cache refactoring, W8A8/W4A8 quantization, and precision
    debugging on Ascend NPU
domain_path: infer-core/recipes-design/skills-parallel-kv-quant-debug
evidence_strength: software_contract
id: unit-agent-skills-parallel-analysis-kvcache-quantization-and-accuracy-debug
inherits_from:
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-parallel-analysis/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-kvcache/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-quantization/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-accuracy-debug/SKILL.md
  source_digest: ce6874c7c295198568ce7356947c2ea8deb706f674f0c14975f81b6f0e11d3b8
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:.agents/skills/model-infer-parallel-analysis/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Consolidated methodology from cann-recipes-infer skills for analytical TP/DP/EP/SP
  sizing, static KV cache refactoring, W8A8/W4A8/C8 quantization, and layer-wise accuracy
  bisection.
tier: L2
title: Agentic Playbook for Parallelism Sizing, KV Cache Staticization, Quantization,
  and Accuracy Debugging
triggers:
  error_signatures:
  - model infer parallel analysis skill
  - model infer kvcache skill
  keywords:
  - model infer parallel analysis skill
  - model infer kvcache skill
  - model infer quantization skill
  - model infer accuracy debug skill
  - layerwise cosine similarity bisect npu
  - model-infer-parallel-analysis
  - model-infer-kvcache
  - model-infer-quantization
  - model-infer-accuracy-debug
  - bisect-layer
---
## 1. Parallelism Analytical Sizing (`model-infer-parallel-analysis`)

1. **Per-Rank Memory Feasibility Check**:
   $$M_{\text{rank}} = \frac{M_{\text{attn}}}{\text{TP}_{\text{attn}}} + \frac{M_{\text{moe}}}{\text{EP} \times \text{TP}_{\text{moe}}} + M_{\text{KV}}(\text{batch}, S, \text{TP}_{\text{attn}}) + M_{\text{act}} + M_{\text{HCCL}} \le 0.85 \times M_{\text{HBM}}$$
2. **GQA Head Divisibility & KV Replication Rule**:
   - When `num_kv_heads < tp_size`, requires `tp_size % num_kv_heads == 0` and replicates each KV head across `tp_size // num_kv_heads` ranks.
   - For MLA (`num_kv_heads=1` compressed latent vector), prefer `DP` on Attention (`attn_tp=1` or `2`) so the latent KV cache is not replicated across 8 or 16 ranks.
3. **MoE `TP` vs. `EP` Selection**:
   - Single-node small model (`<= 140 GB` BF16) + small batch (`B <= 8`) without `MC2`: `TP8` (`2` `AllReduce`/layer) often beats naive `all_to_all` `EP8`.
   - Large MoE (`> 140 GB` or `>= 128` experts) or high concurrency (`B >= 16`) with `MC2`: use `EP16` / `EP32` (`moe_tp=1`, keeping `experts_per_rank <= 24`).

## 2. Static KV Cache Refactoring (`model-infer-kvcache`)

- **Step 1**: Pre-allocate `[num_layers, 2, batch, max_seq_len, kv_heads_per_rank, head_dim]` (or paged blocks) once before inference.
- **Step 2**: During Prefill (`seq_len = S_in`), write `k, v` into slice `[:, :S_in]` or via `scatter_update_`.
- **Step 3**: During Decode (`seq_len = 1`), update slot `cur_pos` in-place via `torch_npu.scatter_update_(k_cache, cur_pos_tensor, k_new, axis=-2)` and pass the full static buffer plus `actual_seq_lengths_kv = cur_pos + 1` to `FIA`.

## 3. Inference Quantization Selection (`model-infer-quantization`)

- **`W8A8` Dynamic Per-Token**: Use `torch_npu.npu_dynamic_quant(x)` -> `npu_quant_matmul(x_int8, w_int8, w_scale, pertoken_scale=x_scale, output_dtype=torch.bfloat16)` for compute-bound Prefill and large-batch MoE (`GMM`).
- **`W4A16` / `W8A16` Weight-Only**: Use `torch_npu.npu_weight_quant_batchmatmul` for memory-bandwidth-bound small-batch Decode (`B = 1..4`).
- **`C8` KV Cache**: Quantize K/V states to `INT8` before `scatter_update_` and pass `dequant_scale` to `FIA` for long-context decode.

## 4. Layer-Wise Accuracy Debugging (`model-infer-accuracy-debug`)

When optimized NPU output diverges from Golden:
1. **Binary Search Across Stages**: Disable Graph Mode -> Disable Fused MoE -> Disable FIA -> Disable Static KV -> Compare TP vs. 1-card to isolate the exact offending stage.
2. **Single-Step Layer Hook Comparison**: Register forward hooks on Layer `0..L-1` in step 0 (Prefill) and step 1 (first Decode token); locate the **first layer** where `cosine_similarity(out_opt, out_ref) < 0.999` or `max_abs_diff` spikes (`NaN`/`Inf`).
3. **Common NPU Pitfalls**: RoPE half-rotation vs. interleaved layout mismatch (`mode=0` vs `mode=1`), `TND` cumulative vs. non-cumulative `actual_seq_lengths`, missing `contiguous()` before `view()` on transposed attention outputs, or `FP16` overflow in un-clamped SwiGLU/RMSNorm.

## Verification Checklist

- Always check Step 0 (Prefill logits) and Step 1 (first Decode logits) separately: Step 0 failures point to weight slicing/RoPE/FIA prefill, whereas Step 0 pass + Step 1 fail points directly to KV Cache indexing or `actual_seq_lengths_kv`.
