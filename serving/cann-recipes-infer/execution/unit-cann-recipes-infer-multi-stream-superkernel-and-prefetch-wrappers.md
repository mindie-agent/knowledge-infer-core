---
category: sop
conditions:
  repository: cann/cann-recipes-infer
  revision: 3c9266727c2a25bd17d503ad37d189a134ff364b
domain_path: serving/cann-recipes-infer/execution
evidence_strength: single_observation
id: unit-cann-recipes-infer-multi-stream-superkernel-and-prefetch-wrappers
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/cann-recipes-infer-reviewed-source.json
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/stream_utils.py#L25-L100
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L50-L120
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/graph_utils.py#L21-L122
  source_digest: ab761d3c2e830b80993d1dbda55723e92ed9e98db52ed9e59a99dc4c136113a6
  source_locator: docs/evidence/cann-recipes-infer-reviewed-source.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Dual-mode Eager and FX-Graph wrappers for NPU multi-stream switching, dependency events, SuperKernel scope, and L2 prefetch
triggers:
  error_signatures:
  - torch.fx.symbolic_trace fails on torch.npu.stream context manager or stream.wait_Stream
  - Tensor freed by primary stream caching allocator while secondary stream kernel is still executing
  - SuperKernel scope or DCCI disable pass fails to attach during FX graph compilation
  keywords:
  - cann-recipes-infer
  - stream_utils
  - npu_stream_switch
  - wait_tensor
  - record_stream
  - superkernel_scope
  - dcci_disable_on_kernel
  - npu_prefetch
---
At the pinned `cann/cann-recipes-infer` revision, `executor/utils/stream_utils.py`, `executor/utils/common_utils.py`, and `executor/utils/graph_utils.py` unify Eager (`torch.npu.stream`), `ACLGraph`, and `npugraph_ex` (`torch.fx.Interpreter` / `torchair.scope`) execution behind mode-aware primitives controlled by `GlobalConfig.enable_multi_stream` and `GlobalConfig.enable_graph_mode`.

## 1. Mode-Aware Stream Switching and Event Dependency (`stream_utils.py:25-100`)

1. **Named Stream Registry (`get_stream`, `lines 29-42`)**:
   - Lazily instantiates named secondary streams (`_STREAMS[stream_name] = torch.npu.Stream()`) alongside `"default"` (`torch.npu.current_stream()`).
2. **Unified `npu_stream_switch(stream_Tag, stream_priority=0)` (`lines 56-75`)**:
   - When `not GlobalConfig.enable_multi_stream`: yields as a no-op context manager.
   - When `GlobalConfig.enable_graph_mode`: delegates to `torchair.scope.npu_stream_switch(stream_Tag, stream_priority)` so the stream tag is recorded as graph scope metadata without breaking FX tracing.
   - When Eager (`not GlobalConfig.enable_graph_mode`): synchronizes `"default"` stream into the target stream on entry (`stream = get_stream(stream_Tag); stream.wait_stream(get_stream("default"))`) and enters `with torch.npu.stream(stream):`.
3. **Cross-Stream Ordering and Caching Allocator Lifetime (`lines 78-100`)**:
   - `wait_tensor(user, Wait_tensor, Wait_stream="")`:
     - In graph mode: calls `torchair.scope.npu_wait_tensor(user, Wait_tensor)` to insert a data-driven dependency edge between FX nodes.
     - In eager mode: calls `torch.npu.current_stream().wait_stream(get_stream(Wait_stream))` when `Wait_stream` is specified.
   - `record_stream(tensor, stream_Tag)`:
     - In eager multi-stream mode (`not GlobalConfig.enable_graph_mode`), calls `tensor.record_stream(get_stream(stream_Tag))` so the PyTorch NPU caching allocator does not reuse the HBM block of `tensor` until `stream_Tag` finishes consuming it.

## 2. SuperKernel Scope, DCCI Cache-Flush Suppression, and L2 Prefetch (`common_utils.py:50-120`, `graph_utils.py:21-122`)

1. **`superkernel_scope(scope_name, options=None)` and `dcci_disable_on_kernel()` (`common_utils.py:59-72`)**:
   - Both wrappers check `GlobalConfig.enable_super_kernel and GlobalConfig.enable_graph_mode`; only when both flags are true do they enter `torchair.scope.super_kernel(scope_name, options)` and `torchair.scope.dcci_disable()`, respectively.
   - In `graph_utils.py:58-67`, `GraphWrapper.forward` wraps the decode graph execution path (`if not self.is_prefill`) with `superkernel_scope(self.model.__class__.__name__)` and `limit_core_num(self.aicore_num, self.vector_num)`.
2. **Asynchronous Weight Prefetch (`npu_prefetch`, `common_utils.py:75-110`)**:
   - Signature: `npu_prefetch(weight, dependency, max_size=0, offset=0, enabled=True)`.
   - Skips execution unless `enabled and GlobalConfig.enable_prefetch`.
   - When `max_size <= 0`, computes `weight_size = weight.element_size() * weight.numel()` and caps `max_size` at `MAX_PREFETCH_SIZE` (`16 * 1024 * 1024` bytes = 16 MiB).
   - Dispatches `torch_npu.npu_prefetch(weight, dependency, max_size, offset)` to overlap HBM-to-L2 weight warm-up with the execution of the `dependency` operator.

This unit is a source-derived inspection procedure at the exact repository revision in `conditions`.

Pinned source ranges:

- [executor/utils/stream_utils.py:25-100](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/stream_utils.py#L25-L100)
- [executor/utils/common_utils.py:50-120](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/common_utils.py#L50-L120)
- [executor/utils/graph_utils.py:21-122](https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/executor/utils/graph_utils.py#L21-L122)
