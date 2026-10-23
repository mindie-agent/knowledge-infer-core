---
schema: mindie-graded-unit/1
id: unit-mindie-sd-vae-tae-sliced-state-decode-and-offload-sop
title: MindIE-SD VAE/TAE independence proof, latent-boundary sliced state decode, and async CPU offload SOP
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: infer-core/mindie-sd/vae-and-offload
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "all_gather_into_tensor writes to stack copy leaving original list uninitialized"
    - "nn.Upsample large batch silently leaves tail frames un-updated with stale HBM"
    - "deadlock when collective participation depends on rank < pieces"
  keywords:
    - vae-opt
    - sliced_state_decode
    - MemBlock
    - shard_equivalence_check
    - frame_health
    - enable_offload
    - share_memory
    - min_reserved_blocks_count
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
  source_digest: 351aff4228d45196180fe9a7fd227070f7cff06c7df846332af3d0cc8c43dcf6
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/cpu_offload.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/share_memory.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Step 0 — Stage Share Gate & Roofline Check**:
   - Measure `<Pipeline>.decode` (VAE/TAE) and host delivery (`mp4` encoding + worker-to-API transfer) as a fraction of end-to-end latency. Only invest in VAE/host optimization when the non-DiT stage share is $\ge 10\%$ (common on 4-step distilled video models). When VAE `cube_utilization` is already near 100% (`aicore_time == Task Duration`), kernel tuning has zero headroom—only **workload reduction** (multi-rank sharding or precision reduction) can reduce latency.

2. **Step 1 & 2 — Prove Cross-Axis Operator Independence Before Sharding**:
   - Audit every operator in the decoder for cross-axis coupling (`does output element y[t] depend on inputs outside slice boundary?`):
     - **Frame-Independent 2D Decoders (`tae-2d`)**: `Clamp`, `2D Conv 3x3`, spatial `Upsample`, and per-frame `quantile` normalization have zero temporal coupling $\implies$ shard freely along the frame ($T$) axis with **zero halo** and bit-exact equivalence.
     - **Temporal Recursive Decoders (`taehv-temporal`)**: `MemBlock(past)` recursively feeds the previous frame's activation into the next frame $\implies$ naive multi-rank $T$-axis sharding is **mathematically impossible** to make bit-exact without replaying prefix frames. (Never mistake stock `chunk_latents=5 / trim=3` boundary-artifact cropping for proof of temporal independence.)
     - **Conditional Equivalence on Audio VAE**: Short audio slices are bit-exact only when slice length $\ge \text{MINLEN}$ (determined by halo width / memory block geometry). Pad short slices with zeros to `MINLEN` and slice back after decoding (`ndiff = 0`).

3. **Step 3 — Slice Temporal Recursive Decoders at Latent-Frame Boundaries with Input State Carry (`sliced_state_decode`)**:
   - To halve peak NPU HBM and bypass CANN `nn.Upsample` large-batch tail-frame corruption (where oversized single-call batch sizes silently fail to write tail output frames, leaving stale HBM and flat/corrupted tail video frames with no error):
     1. Cut the latent sequence strictly at **latent-frame boundaries** (e.g., `time_upscale = 4` maps latent frame $j \leftrightarrow$ output frame $4j$, which is a frame boundary at every temporal upsampling stage).
     2. Carry `state[i]` across sequential slices, where `state[i]` is the **last-frame activation at the INPUT of `MemBlock[i]`** from slice $k-1$ (never the block output; initialize slice $0$ with zeros).
     3. Never replace `nn.Upsample` with `repeat_interleave` along spatial axes—while numerically equivalent, `repeat_interleave` allocates an extra full-size output tensor and triggers OOM.

4. **Step 4 & 5 — Enforce Rank-Independent Collective Guards, Exchange Budgets, and 4-Gate Verification**:
   - **Rank-Independent Collective Entry**: Derive sharding participation (`ok = pieces >= 2 and world >= 2 and shape_is_divisible(shape)`) purely from environment and tensor shape—never gate collectives on `if rank < pieces:`, which deadlocks when receivers need dynamic output shapes.
   - **Avoid `all_gather_into_tensor` Stack-Copy Silent Corruption**: Never pass `torch.stack(tensor_list)` as the destination of `dist.all_gather_into_tensor` (which writes into the temporary stack copy and leaves `tensor_list` uninitialized). Allocate a contiguous `torch.empty((world, *shape))` buffer first, gather into it, and slice.
   - **Pre-Calculate Exchange Byte Budget**: $\text{Exchange Bytes} = \text{Output Bytes} \times \frac{\text{pieces}-1}{\text{pieces}}$. Because FP32 video output reaches multiple GiB per rank, convert clamped `[0, 1]` output to `uint8` ($4\times$ smaller) before `all_gather`, or encode tiles locally.
   - **Four Mandatory Verification Gates**: (1) CPU full vs $N$-slice bitwise parity (`max|d| == 0` via `shard_equivalence_check.py`), (2) 4-request (1 cold + 3 hot) end-to-end `md5` parity, (3) bucketed high-frequency energy & inter-frame delta health (`frame_health.py` to catch tail-frame collapse), and (4) wall-clock + peak HBM budget check.

5. **Step 6 — Configure Async Block `enable_offload` & Multi-Process `share_memory`**:
   - For single-card HBM-constrained DiTs, call `mindiesd.enable_offload(model, blocks, min_reserved_blocks_count=...)` to overlap `h2d_stream` prefetch of block $i+1$ and `d2h_stream` eviction of block $i-1$ with compute on block $i$.
   - For multi-rank single-node inference with replicated read-only weights, use `mindiesd.init_share_memory` / `share_memory` to share physical NPU weight pages across worker processes.
