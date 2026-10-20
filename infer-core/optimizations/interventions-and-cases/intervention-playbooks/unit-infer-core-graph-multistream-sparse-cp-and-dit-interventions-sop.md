---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/interventions-and-cases/intervention-playbooks
evidence_strength: software_contract
id: unit-infer-core-graph-multistream-sparse-cp-and-dit-interventions-sop
inherits_from:
- unit-infer-core-mla-dsa-lightning-indexer-and-sparse-attention-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/aclgraph_decode_steady_state_skip_guard_reinplace.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/aclgraph_print_bisect_silent_hang.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/sfa_dsa_cp_cross_group_kv_allgather.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/qwen3_next_gdn_context_parallel_zeco.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/kvp_longseq_decode.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/ring_attention_comm_compute_overlap_lse_merge.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/upstream_dit_video_monkeypatch_npu_adaptation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/video_dit_fp8_mxfp8_quantization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/guides/experience/interventions/vae_multicard_spatial_parallel.md
  source_digest: 76c746cb7a38840cb53d89afb66a82b95b49a831b9ecb8682e39364749a5f660
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/guides/experience/interventions/aclgraph_decode_steady_state_skip_guard_reinplace.md
  source_type: cannbot_intervention_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Distilled intervention playbooks: ACLGraph steady-state guard skip, print-bisect
  for silent graph hangs, SFA-CP cross-group KV AllGather, and KVP/ZeCO long-context'
triggers:
  error_signatures:
  - ACLGraph capture hangs silently without traceback or plog error code inside complex
    custom model forward
  - Combining DSA/SFA sparse attention with Context Parallelism (CP) fails because
    topk_indices reference global KV blocks absent on local CP rank
  - Steady-state ACLGraph decode incurs host CPU overhead from redundant Dynamo guard
    evaluation and reinplace passes
  keywords:
  - aclgraph_decode_steady_state_skip_guard
  - aclgraph_print_bisect_silent_hang
  - sfa_dsa_cp_cross_group_kv_allgather
  - qwen3_next_gdn_context_parallel_zeco
  - kvp_longseq_decode
  - upstream_dit_video_monkeypatch
  - video_dit_fp8_mxfp8
---
# Distilled intervention playbooks: ACLGraph steady-state guard skip, print-bisect for silent graph hangs, SFA-CP cross-group KV AllGather, and KVP/ZeCO long-context

## Applicability and symptom scope

Use this SOP when applying targeted architectural interventions from the 22 CANNBot intervention playbooks for ACLGraph capture, Sparse Attention + Context Parallelism, GDN linear-attention CP, and DiT non-intrusive monkey-patching.

## Procedure

1. **Bypass redundant steady-state Dynamo guards and bisect silent ACLGraph capture hangs (`aclgraph_decode_steady_state_skip_guard_reinplace.md`, `aclgraph_print_bisect_silent_hang.md`).**
   - Once all decode batch buckets are warmed up and captured into `ACLGraph` runners, dispatch steady-state decode directly to the bucket's captured graph handle keyed by `padded_batch_size` without re-running Python Dynamo guard checks or FX `reinplace` passes on every step.
   - When `ACLGraph` capture (`rtStreamBeginCapture`) hangs silently with zero traceback:
     1. Do **not** insert `torch.npu.synchronize()` inside the capture region (which is illegal during capture);
     2. Bisect by inserting host-side `print(..., flush=True)` markers between submodules during the *capture trace* and checking `rtStreamGetCaptureStatus` to pinpoint the exact operator whose C++ launch internally queries host-sync state or switches to an uncaptured default stream.

2. **Reconcile Sparse Attention (`SFA` / `DSA`) with Context Parallelism (`CP`) via cross-group KV `AllGather` (`sfa_dsa_cp_cross_group_kv_allgather.md`, `kvp_longseq_decode.md`, `qwen3_next_gdn_context_parallel_zeco.md`).**
   - Under Context Parallelism (`PCP`/`DCP`), each rank stores only its sequence shard of the KV cache, whereas `Lightning Indexer` + `SparseFlashAttention` selects arbitrary global `topk_indices` across the entire prompt.
   - **Intervention:** Before executing `Lightning Indexer` and `SFA` in CP prefill, `AllGather` the compressed latent KV (and indexer key) across the CP group into a temporary contiguous buffer (or gather only the selected KV blocks across CP ranks) so `topk_indices` address a complete KV view.
   - For `Qwen3-Next` GDN layers under Context Parallelism (`qwen3_next_gdn_context_parallel_zeco.md`), propagate the inter-rank recurrent state $S_t$ sequentially across CP chunks using the `ZeCO` dependency pipeline while overlapping chunk-local WY matmuls.

3. **Apply zero-intrusion runtime `apply_all()` monkey-patching and per-block FP8 attention for upstream Video DiT models (`upstream_dit_video_monkeypatch_npu_adaptation.md`, `video_dit_fp8_mxfp8_quantization.md`, `vae_multicard_spatial_parallel.md`).**
   - To adapt rapidly evolving upstream DiT repositories without maintaining a diverging fork, inject a single `apply_all()` monkey-patch entrypoint **before importing upstream model modules**, gating fused NPU replacements (`npu_rms_norm`, `npu_rotary_mul`, `FIA` / `Ulysses`) by runtime CANN capability checks.
   - On Ascend 950, prioritize **per-block dynamic FP8 Attention** (under Ulysses SP, which yields the primary end-to-end DiT speedup) over standalone `MXFP8` Linear quantization, and always exclude `adaLN` modulation projections from low-bit quantization.
