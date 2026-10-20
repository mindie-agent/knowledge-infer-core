---
category: axiom
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/concepts/models-and-modules/dit-vae-and-hstu
evidence_strength: software_contract
id: unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
inherits_from: []
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/dit.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/modules/vae.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/hunyuan_video.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/wan2_2_i2v.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/sana_video.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/hunyuan_image_3.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/models/hstu.md
  source_digest: aea8ed29e406ede50ce46bc948f4049f088521d7b3f6bd5b98b81559267fea70
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/inference/concepts/modules/dit.md
  source_type: cannbot_concept_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Architectural invariants of Diffusion Transformers (DiT), 3D Causal VAE, and
  HSTU generative recommenders on Ascend NPU
triggers:
  error_signatures:
  - Video DiT full 3D spatiotemporal attention OOMs or dominates denoising step time
    at 720p/1080p resolution
  - 3D Causal VAE spatial slice parallel produces visible seam artifacts at patch
    boundaries when conv halo context is omitted
  - HSTU generative recommender fails or OOMs on massive categorical embedding tables
    and jagged user-action sequences
  keywords:
  - Diffusion Transformer
  - adaLN-Zero
  - HunyuanVideo
  - Wan2.2-I2V
  - Sana-Video
  - HunyuanImage-3.0
  - HSTU
  - CausalVAE
---
# Architectural invariants of Diffusion Transformers (DiT), 3D Causal VAE, and HSTU generative recommenders on Ascend NPU

## Condition

Applies to multimodal video/image Diffusion Transformer (`DiT`) pipelines (`HunyuanVideo`, `Wan2.2-I2V`, `Sana-Video`, `HunyuanImage-3.0`), 3D Causal `VAE` decoders, and `HSTU` (Hierarchical Sequential Transduction Units) generative recommenders on Ascend NPU.

## Constraint

1. **Diffusion Transformer (`DiT`) Iterative Denoising & Conditioning (`dit.md`, `hunyuan_video.md`, `wan2_2_i2v.md`, `sana_video.md`, `hunyuan_image_3.md`):**
   - Unlike autoregressive LLM decode (memory-bandwidth bound with growing KV cache), `DiT` executes $T_{\text{steps}}$ (`20..50`) fixed-shape forward passes over a full 3D spatiotemporal latent token grid $N = (F/p_t) \times (H/p_h) \times (W/p_w)$ (`30K..120K` tokens), making every step heavily **Cube-compute bound** with bidirectional (non-causal) full attention.
   - Timestep and text/image conditioning are injected per block via **adaLN-Zero** (`scale, shift, gate = Linear(SiLU(t_emb))`), modulating normalized activations as $\text{LN}(x) \odot (1 + \text{scale}) + \text{shift}$.
   - `HunyuanVideo` and `Wan2.2-I2V` combine dual-stream (MM-DiT joint video+text) and single-stream blocks with 3D RoPE; `Sana-Video` replaces quadratic softmax attention with linear $O(N)$ ReLU/kernel attention and `1x1` Mix-FFN depthwise convolutions; `HunyuanImage-3.0` integrates autoregressive VLM planning with MoE-DiT diffusion heads.

2. **3D Causal VAE Spatial-Temporal Receptive Field & Halo Contract (`vae.md`):**
   - Video VAEs decode latent grids $[B, C, T', H', W']$ to pixel space $[B, 3, T, H, W]$ using 3D causal convolutions (causal padding along $T'$, symmetric padding along $H', W'$), GroupNorm, and upsampling stages.
   - Because $3 \times 3$ spatial convolutions depend on neighboring boundary pixels across a multi-layer receptive field, splitting high-resolution VAE tensors across multiple NPUs along $H'$ or $W'$ (`vae_patch_parallel`) requires exchanging boundary **halo/overlap strips** via P2P `Send`/`Recv` before each convolution and synchronizing global GroupNorm statistics via `AllReduce`.

3. **HSTU Generative Recommender Transduction (`hstu.md`):**
   - `HSTU` replaces standard softmax Transformer blocks with pointwise aggregated sequential transduction ($\text{SiLU}(Q K^\top + \text{rel\_pos\_bias}) V / N_{\text{seq}}$) gated by elementwise $U \odot \text{Norm}(\cdot)$ over variable-length jagged user interaction histories (`TND` packed format) coupled with multi-gigabyte vocabulary-parallel item embedding tables.
