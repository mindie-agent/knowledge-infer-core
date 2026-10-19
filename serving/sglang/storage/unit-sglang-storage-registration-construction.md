---
category: sop
conditions:
  repository: sgl-project/sglang
  revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043
content_digest: sha256:6e7f58d6fb745f60866584091c5c22befc9218b8c476b66f05ed08359d5439e3
domain_path: serving/sglang/storage
evidence_strength: single_observation
id: unit-sglang-storage-registration-construction
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-sglang-storage-registration-construction-review.json
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L16-L63
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L65-L111
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L113-L150
  - https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L152-L200
  source_digest: 773a9b73577089641f9cc9be00954f818ea280601012f14b8741d803ece8943f
  source_locator: docs/evidence/unit-sglang-storage-registration-construction-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Storage backend registration and construction are separate gates
triggers:
  error_signatures: []
  keywords:
  - sglang
  - storage
  - backend
  - registration
  - and
  - construction
  - are
  - separate
  - gates
---
StorageBackendFactory.register_backend stores a loader closure and its module/class names without importing the module; duplicate names are logged and replaced. create_backend invokes that loader for a registered name. The loader imports the module, obtains the named attribute and requires a HiCacheStorage subclass. Construction then goes through _create_builtin_backend, which dispatches on literal names and uses different constructor signatures, including the HF3FS from_env_config path. An otherwise valid registered name with no recognized construction branch still raises Unknown built-in backend.

Dynamic loading is a separate route reached only when the requested name is exactly dynamic and extra_config is present. It requires backend_name, module_path and class_name, applies the same loader validation, and constructs the class with storage_config and the kwargs dictionary as the second positional argument. It does not pass mem_pool_host to that constructor. To extend this factory, inspect both the lazy-loading registration and the selected construction contract; adding a registry entry alone does not make an arbitrary name creatable.

Reviewed limits and corrections:

- Class validation occurs when the loader runs, not at registration.
- The dynamic branch is not a general fallback for any unknown name and does not forward the host pool as its second constructor argument.

Omitted scope:

- Individual backend implementations, AIBrix test execution and backend deployment support are outside this subset.

Adoption boundary: Use as an inspection procedure for the pinned SGLang source and the named path only. Source control flow and test oracles are not a reproduced execution result or measured performance benefit.

Evidence scope: Independently read the listed exact Git ranges and manually authored the corrected mechanism. Only this named subset is accepted; other original model claims and unread bytes are not accepted. No framework or hardware execution was performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [python/sglang/srt/mem_cache/storage/backend_factory.py:16-63](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L16-L63)
- [python/sglang/srt/mem_cache/storage/backend_factory.py:65-111](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L65-L111)
- [python/sglang/srt/mem_cache/storage/backend_factory.py:113-150](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L113-L150)
- [python/sglang/srt/mem_cache/storage/backend_factory.py:152-200](https://github.com/sgl-project/sglang/blob/8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043/python/sglang/srt/mem_cache/storage/backend_factory.py#L152-L200)
