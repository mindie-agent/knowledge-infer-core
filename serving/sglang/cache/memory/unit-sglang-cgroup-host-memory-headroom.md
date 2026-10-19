---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:54b516919de52d66ec37babdac160392bc2f4e037c1d9eb2081bc068f976ef0f
domain_path: serving/sglang/cache/memory
evidence_strength: single_observation
id: unit-sglang-cgroup-host-memory-headroom
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-cgroup-host-memory-headroom-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L24-L38
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L41-L133
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L136-L162
  source_digest: b91c2ae23a195e4526683c289db52f83813243d9dea4f07ebfc48e3286b84fc2
  source_locator: docs/evidence/unit-sglang-cgroup-host-memory-headroom-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Host-memory sizing through visible cgroup limits
triggers:
  error_signatures: []
  keywords:
  - sglang
  - host-memory
  - sizing
  - through
  - visible
  - cgroup
  - limits
---
available_host_memory_bytes starts from psutil's available RAM and, when _cgroup_memory_headroom returns a number, takes the smaller value. The helper resolves the process's cgroup membership against mounted cgroup filesystems and walks ancestors to the mount root. It considers memory.max and memory.high for cgroup v2 or memory.limit_in_bytes for v1, subtracts eligible file-page-cache credit from usage, clamps remaining capacity at zero and keeps the minimum discovered headroom. A nonzero or unreadable descendant memory.min disables that cache credit for v2; the reclaim target's own minimum is not checked by that descendant walk.

Missing procfs or no discovered numeric limit can return no bound normally. Unresolved memberships raise RuntimeError. The caller catches only OSError and RuntimeError: automatic sizing then raises an explicit-sizing error unless allow_cgroup_fallback is enabled, in which case it logs and returns host available RAM. ValueError from path or numeric parsing is outside that catch. These branches describe a sizing estimate; including memory.high does not establish a hard OOM boundary or aggregate safety across ranks.

Reviewed limits and corrections:

- Host available RAM can also be returned on the normal no-bound path; it is not exclusive to opt-in exception fallback.
- The fallback catches a specific exception set and is not universal.
- Do not turn a headroom estimate or memory.high inclusion into an OOM-prevention guarantee.

Omitted scope:

- IndexKeyCache, HiCache configurator staging and automatic ratio derivation are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/host_memory.py:24-38](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L24-L38)
- [python/sglang/srt/mem_cache/host_memory.py:41-133](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L41-L133)
- [python/sglang/srt/mem_cache/host_memory.py:136-162](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/host_memory.py#L136-L162)
