# Core Inference Engine Domain Boundaries (`domain: "infer-core"`)

This repository (`knowledge-infer-core`) contains core
inference engine knowledge and source-verification evidence under the canonical
domain name `infer-core`. Keep stable unit IDs independent of path moves. Parse
every unit as `mindie-graded-unit/1`, preserve its `evidence_strength` and
declared `conditions`, and rebuild `index/routing-manifest.json` and
`index/shard-topology.json` through the graded indexer after any unit edit.
Runtime code remains in `mindie-agent/knowledge`.

## Operating Rules

1. **Exact Repository & Revision Conditions**: Bind source-derived units to the
   exact `repository` and `revision` inspected (`conditions.repository` and
   `conditions.revision`). Keep one engine implementation per unit rather than
   blending `vLLM`, `SGLang`, and `TGI` state machines into unverified unified
   rules. Queries with mismatched revisions must be rejected, and queries
   omitting `revision` must remain excluded from default results (inspectable
   only via `include_unknown=True`).
2. **Evidence Honesty**: Keep request termination, ownership release, allocator
   pool return, and physical device execution distinct. Do not upgrade
   `single_observation` source-reading SOPs to `reproduced_empirical` or `L4`
   without real execution/ablation receipts, and never adopt unreviewed raw
   model outputs without line-level source verification.
3. **Domain Isolation**: Do not import Ascend operator kernels (`operators`),
   `vllm-ascend` adapter corpus files (`vllm-ascend`),
   or unclosed private histories into this repository. Cross-domain `explain`
   and `feedback` URIs belonging to other domains must be rejected.
4. **Publication Gate**: [`./publication-contract.json`](./publication-contract.json)
   (`mindie-publication-contract/2`) and maintenance entry points under
   `scripts/` define the validator pin and local checks; run
   `python3 scripts/build_routing_index.py --root . --check` after any unit or
   index edit.
