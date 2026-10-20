# Core Inference Engine Knowledge (`infer-core`)

This independent local repository (`/Volumes/data/code/knowledge-infer-core`,
canonical domain `infer-core`, initial local commit
`da8844d02da74673320e7e5155dc5f742ec1fe20`) holds source-backed diagnostic
procedures and mechanisms for general inference engine request lifecycle,
scheduling, block/page reclamation, and KV-cache ownership. It contains no
migrated `vllm-ascend` corpus and no cross-domain operator units.

## Current Admitted Units & Exact Version Conditions

All three initial units are `L2` (`category: sop`, `evidence_strength:
single_observation`) admitted through the real C6 `prepare -> validate -> apply ->
export` pipeline from Root's independently source-verified diagnostic conclusions
(`/Volumes/data/temp/mindie-003-execution/wave3/research/editorial-review-packets/parent-acceptance.json`,
acceptance receipt `/Volumes/data/temp/mindie-003-execution/wave3/knowledge/infer-core-acceptance.json`).
Each unit is bound to a single engine implementation at an exact `repository` and
Git `revision`:

- `serving/vllm/unit-vllm-cancellation-and-block-reclamation.md`
  - `conditions`: `repository: vllm-project/vllm`,
    `revision: 752a3a504485790a2e8491cacbb35c137339ad34`
  - `evidence`: `docs/evidence/vllm-reviewed-source.json`
- `serving/sglang/unit-sglang-cancellation-and-cache-ownership.md`
  - `conditions`: `repository: sgl-project/sglang`,
    `revision: 8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043`
  - `evidence`: `docs/evidence/sglang-reviewed-source.json`
- `serving/tgi-v3/unit-tgi-v3-cancellation-and-batch-reclamation.md`
  - `conditions`: `repository: huggingface/text-generation-inference`,
    `revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed`
  - `evidence`: `docs/evidence/text-generation-inference-reviewed-source.json`

Default `Store` query matches a unit only when the caller supplies the exact
matching `repository` and `revision` (`default_matches = 1`, `explain` passed).
Queries with a mismatched revision are rejected (`wrong_revision_matches = 0`),
and queries omitting the revision are excluded from default results
(`missing_revision_matches = 0`) and returned only when `include_unknown=True` is
explicitly requested for inspection (`include_unknown_inspection = 1`).
Six-direction foreign reference checks across `operators`, `vllm-ascend`, and
`infer-core` reject cross-domain URI lookups on the wrong domain store.

## Evidence & Publication Boundaries

- **Source-Inspection Scope Only**: These `L2` SOPs trace request finish/abort
  handling, deferred block or radix-tree eviction fences, and bookkeeping vs.
  allocator pool return at the pinned source revisions. They do not prove
  physical device-memory deallocation, immediate termination of launched device
  work, a reproduced runtime fault, or any measured performance result.
- **Editorial Selection**: Only the source ranges verified and corrected in
  `parent-acceptance.json` were admitted; unreviewed raw cohort outputs were not
  adopted.
- **Local Draft Status**: This repository has no configured Git remote and no
  `publication-contract.json`. Strict `Feed` synchronization correctly blocks
  with `contract_mismatch` until an exact reviewed runtime commit implementing
  `mindie-publication-contract/2` is pinned. No remote publication, formal
  runtime pin cutover, or general retrieval benchmark acceptance is claimed.
- **Derived Indexes**: `index/routing-manifest.json` and
  `index/shard-topology.json` are generated from the Markdown units (`derived_index_validation: passed`).
