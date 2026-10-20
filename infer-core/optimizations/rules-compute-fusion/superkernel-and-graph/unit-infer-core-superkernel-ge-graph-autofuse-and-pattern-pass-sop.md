---
category: sop
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/optimizations/rules-compute-fusion/superkernel-and-graph
evidence_strength: software_contract
id: unit-infer-core-superkernel-ge-graph-autofuse-and-pattern-pass-sop
inherits_from:
- unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/superkernel_decode_ge_graph_fusion_scope.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/superkernel_scope_segmented_gated.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/superkernel_ascendc_relocatable_binary.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/graph_mode_decode_only_compile_backends.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/graph_mode_enable_decision_profiling.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/torchair_cache_compile_persisted_graph.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/graph_fusion_register_replacement_guardrails.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/inductor_autofuse_partial_compile_scope.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/staged_single_card_optimize_pipeline.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-superkernel/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-graph-mode/SKILL.md
  source_digest: 9c38ba1c67b72106b4caf789b9d3dc22ef99df109ad8e63992d3dd5d7f5909dd
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/optimizations/experience/rules/superkernel_decode_ge_graph_fusion_scope.md
  source_type: cannbot_optimization_rule_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Inference optimization rules Part 1B (Graph & SuperKernel): Segmented SuperKernel
  scopes, relocatable AscendC binaries, TorchAir pattern passes, and AutoFuse'
triggers:
  error_signatures:
  - Wrapping an entire Transformer layer in one SuperKernel scope fails or skips custom
    AscendC operators
  - Custom AscendC kernel inside SuperKernel scope is excluded from fusion because
    --op_relocatable_kernel_binary=true was omitted at build time
  - torchair register_replacement pattern pass matches across stream boundaries or
    wrong head_dim and corrupts graph execution
  keywords:
  - superkernel_decode_ge_graph
  - superkernel_scope_segmented
  - op_relocatable_kernel_binary
  - register_replacement
  - cache_compile
  - inductor_autofuse
  - dcci_disable
---
# Inference optimization rules Part 1B (Graph & SuperKernel): Segmented SuperKernel scopes, relocatable AscendC binaries, TorchAir pattern passes, and AutoFuse

## Applicability and symptom scope

Use this SOP when enabling NPU graph modes (`ge_graph`, `acl_graph`, `npugraph_ex`, `AutoFuse`), configuring `SuperKernel` task-level fusion in decode, or authoring `torchair.register_replacement` FX pattern passes.

## Procedure

1. **Follow the 3-stage optimization pipeline and profile-driven graph-mode decision gate (`staged_single_card_optimize_pipeline.md`, `graph_mode_enable_decision_profiling.md`, `graph_mode_decode_only_compile_backends.md`).**
   - Always execute single-card/multi-card optimization in strict stage order with accuracy + latency gates after each stage:
     1. Static/Paged KV Cache + Fused Attention (`FIA` / `PA`) replacement;
     2. Pointwise/Norm/RoPE/SwiGLU `torch_npu` fused operator replacement (`npu_rms_norm`, `npu_add_rms_norm`, `npu_rotary_mul`, `npu_swiglu`);
     3. Decode-only Graph compilation (`if not is_prefill:` -> `torchair` GE graph or `npugraph_ex` / `ACLGraph`), keeping variable-length prefill in eager mode.
   - Enable graph mode when profiling shows decode is host-bound (`Free` time / launch gap significant); persist compiled GE graphs to disk via `torchair.inference.cache_compile` (`torchair_cache_compile_persisted_graph.md`) keyed by CANN version, rank topology, and bucket shape.

2. **Segment `SuperKernel` scopes by functional sub-block and build AscendC ops with `--op_relocatable_kernel_binary=true` (`superkernel_decode_ge_graph_fusion_scope.md`, `superkernel_scope_segmented_gated.md`, `superkernel_ascendc_relocatable_binary.md`).**
   - `SuperKernel` merges consecutive device kernel launches in a `torchair` GE decode graph into a single continuous AICore instruction stream (eliminating inter-kernel task-scheduler gaps and DCCI cache flushes when paired with `dcci_disable()`).
   - **Segmentation Rule:** Never wrap an entire model or cross-collective / multi-stream region inside a single giant `super_kernel` scope; wrap **segmented functional sub-blocks** separately (one scope for the Attention sub-chain, one scope for the local MoE/MLP sub-chain) gated by a zero-overhead `nullcontext()` when disabled.
   - **Relocatable Binary Requirement:** Custom AscendC operators inside a `SuperKernel` scope are silently excluded from fusion unless compiled with `--op_relocatable_kernel_binary=true` in `CMakeLists.txt` / `build.sh`.

3. **Enforce the 3 guardrails on `torchair.register_replacement` FX subgraph fusion (`graph_fusion_register_replacement_guardrails.md`, `inductor_autofuse_partial_compile_scope.md`).**
   - When registering FX pattern replacements (`search_fn`, `replace_fn`, `extra_check`):
     1. **Hard-gate `head_dim` and shape alignments** inside `extra_check` (returning `False` on unsupported dimensions rather than failing in kernel tiling);
     2. **Pass pre-existing `scale` tensors directly** without inserting redundant cast/reciprocal nodes in `replace_fn`;
     3. **Reject cross-stream matches** in `extra_check` so nodes tagged with different `npu_stream_switch` scopes are never fused into one kernel.
   - For `Inductor` + `AutoFuse`, apply `torch.compile` selectively to pointwise-dense, static-structure sub-modules (e.g., norm + gating + residual epilogues) rather than full-model dynamic control flow.
