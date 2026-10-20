# Core Inference Engine Domain Boundaries (`domain: "infer-core"`)

This repository (`/Volumes/data/code/knowledge-infer-core`) contains core
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
   `vllm-ascend` adapter corpus files (`/Users/maoxx241/code/mindie-transcript-20261004/content`),
   or unclosed private histories into this repository. Cross-domain `explain`
   and `feedback` URIs belonging to other domains must be rejected.
4. **Local Draft & Publication Gate**: This repository (initial commit
   `da8844d02da74673320e7e5155dc5f742ec1fe20`) is a local draft without a remote
   or `publication-contract.json`. Do not fabricate a `publication-contract.json`
   using an older runtime base commit, and do not cut over installed Harness
   pins from this local draft before formal `mindie-publication-contract/2`
   release acceptance.
