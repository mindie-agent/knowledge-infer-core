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
  - Operator fusion, static graph capture, and multi-stream overlap in PyTorch NPU
    inference
domain_path: infer-core/recipes-design/skills-fusion-graph-and-stream
evidence_strength: software_contract
id: unit-agent-skills-fusion-graph-capture-and-stream-overlap-playbook
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-fusion/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-graph/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/.agents/skills/model-infer-stream-overlap/SKILL.md
  source_digest: 2b05edf8d1fba841902fc0e420eb3e1d0bf753924771906a490dd5f164b6e815
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:.agents/skills/model-infer-fusion/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Actionable playbook from cann-recipes-infer agent skills for high-ROI operator
  fusion patterns, eliminating graph breaks in npugraph_ex / TorchAir, and safe multi-stream
  event synchronization.
tier: L2
title: Agentic Playbook for Operator Fusion, ACLGraph/TorchAir Capture, and Multi-Stream
  Overlap
triggers:
  error_signatures:
  - model infer fusion skill
  - model infer graph aclgraph torchair
  keywords:
  - model infer fusion skill
  - model infer graph aclgraph torchair
  - model infer stream overlap skill
  - npugraph ex static kernel capture
  - fix graph break aclgraph torch npu
  - model-infer-fusion
  - model-infer-graph
  - model-infer-stream-overlap
  - aclgraph
  - npugraph-ex
  - torchair
  - static-kernel
---
## 1. High-ROI Operator Fusion Patterns (`model-infer-fusion`)

| Subgraph Pattern | `torch_npu` Fused Replacement | Key Invariant / Constraint |
|---|---|---|
| `x = x + residual; y = rms_norm(x)` | `y, _, x = torch_npu.npu_add_rms_norm(x, residual, weight, eps)` | Returns both normalized `y` and updated `x` in one HBM pass |
| `q_proj(x), k_proj(x), v_proj(x)` | Merged `qkv_proj` + `torch.split` | Concatenate weights offline at load time along output dim |
| `F.silu(gate_proj(x)) * up_proj(x)` | Merged `gate_up_proj` + `torch_npu.npu_swiglu(x, dim=-1)` | `gate` must occupy the first half and `up` the second half of last dim |
| Rotary Position Embedding (`cos`, `sin`) | `torch_npu.npu_rotary_mul(x, cos, sin)` or `npu_apply_rotary_pos_emb` | Pre-cache `cos`/`sin` tables on NPU for `max_seq_len` at init |
| Eager SDPA / FlashAttention | `torch_npu.npu_fused_infer_attention_score` (`FIA`) | Specify `input_layout` (`BSH`, `BNSD`, `BSND`, `TND`) and `actual_seq_lengths_kv` |
| Loop over $E$ MoE experts | `npu_moe_init_routing_v2` + `npu_grouped_matmul` (`split_item=2`) + `npu_moe_finalize_routing` | Group tokens by expert ID; avoid per-expert Python loop |

## 2. Graph Mode (`npugraph_ex` / `aclgraph` & `torchair`) Rules (`model-infer-graph`)

1. **Five Root Causes of Graph Capture Failure or Silent Fallback**:
   - **Host-Device Scalar Sync**: `.item()`, `.cpu()`, `print(tensor)`, `if tensor.any():`, or `nonzero()`.
   - **Dynamic Tensor Address in Replay**: Re-allocating input tensors instead of calling `.copy_()` into static addresses bound during capture.
   - **Host-List `actual_seq_lengths_kv`**: In `aclgraph` mode, if `actual_seq_lengths_kv` is passed as a Python `list[int]`, its value freezes at capture time! Use a device tensor or `torchair.scope.npu_Clear` / updated static tensor input supported by the operator signature.
   - **Unaligned `GMM` Group List**: Keep `group_list` (`expert_tokens`) as a device `int64` tensor (`group_list_type=0` or `1`) instead of transferring counts to CPU via `.tolist()`.
2. **Decode-Only vs. Full-Model Graph**:
   - Variable-length Prefill runs in Eager mode (compute-bound, minimal dispatch overhead) while fixed-token Decode (`seq_len=1` or `1+MTP`) runs inside `npugraph_ex` (`compile_type="aclgraph"`, `static_kernel=True`).

## 3. Multi-Stream Overlap Safety Rules (`model-infer-stream-overlap`)

1. **Dependency & Lifespan Protection**:
   - Wait on an `npu.Event` recorded on the main stream **before** launching secondary-stream work (`alt_stream.wait_event(start_event)`).
   - Call `tensor.record_stream(alt_stream)` on any tensor consumed by `alt_stream` so the caching allocator does not recycle its memory when `main_stream` advances.
   - Join `alt_stream` back onto `main_stream` (`main_stream.wait_event(end_event)`) before reading or summing the secondary output.

## Verification Checklist

- Confirm `torch.npu.synchronize()` is never called inside the captured graph region.
- Verify multi-step decode in graph mode produces bit-identical or `cos_sim > 0.999` outputs across steps `1..128` (detecting frozen `positions` or `actual_seq_lengths_kv`).
