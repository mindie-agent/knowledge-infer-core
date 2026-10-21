---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - DiT image/video diffusion quantization (Wan2.1/2.2, HunyuanVideo, Flux, Qwen-Image)
domain_path: infer-core/quantization-and-accuracy/dit-quantization-tuning
evidence_strength: software_contract
id: unit-msmodelslim-dit-diffusion-w8a8-dynamic-and-fa3-quantization-recipe
inherits_from:
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/msagent/blob/521f8ead0ebda5db5b282f0c3c4147006e01e016/skills/quantizer/quant-tuning-quantize-dit/SKILL.md
  source_digest: 490238f22134458ce1581e9475a3703d2198b0dd44db8292d708fdea73aaf828
  source_locator: Ascend/msagent@521f8ead0ebda5db5b282f0c3c4147006e01e016:skills/quantizer/quant-tuning-quantize-dit/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Specialized quantization SOP from Ascend/msagent (`quant-tuning-quantize-dit`)
  for DiT diffusion models covering data-free W8A8 dynamic quantization, timestep-sampled
  calibration, dual-expert DiT handling, and FA3 attention quantization.
tier: L2
title: DiT Diffusion Model W8A8 Dynamic Quantization, Timestep Calibration, and FA3
  Quant Recipe
triggers:
  error_signatures:
  - quant tuning quantize dit skill
  - dit diffusion w8a8 quantization
  keywords:
  - quant tuning quantize dit skill
  - dit diffusion w8a8 quantization
  - fa3 attention quantization dit
  - dit timestep aware calibration msmodelslim
  - dual expert dit quantization wan2.2
  - dit-quantization
  - quant-tuning-quantize-dit
  - w8a8-dynamic
  - fa3-quant
  - timestep-calibration
  - wan2.2
  - hunyuan-video
---
## 1. Why DiT Quantization Differs from Autoregressive LLMs

Diffusion Transformers (`Wan2.1/2.2`, `HunyuanVideo`, `Flux`, `SD3`) iterate over $T = 20..50$ denoising timesteps where activation distributions shift drastically from pure Gaussian noise at $t = T$ to structured latent features at $t = 1$:
- Static activation scales calibrated on a single timestep clip severely at other timesteps, causing color saturation shifts, grid artifacts, or temporal flickering.

## 2. Core DiT Quantization Strategies (`quant-tuning-quantize-dit`)

1. **Primary Recommendation — Data-Free `W8A8` Dynamic Per-Token Quantization**:
   - Quantizes DiT linear weights (`to_q`, `to_k`, `to_v`, `to_out`, `ff.net`) offline to symmetric `INT8` per-channel while computing activation scales **dynamically per token at every denoising timestep** via `npu_dynamic_quant` + `npu_quant_matmul`.
   - Requires **zero calibration dataset** (`data-free`) and naturally adapts to shifting activation ranges across all $T$ denoising steps.
2. **Timestep-Stratified Calibration (For Static `W8A8` / `SmoothQuant` / `FA3` Quant)**:
   - When static smoothing (`iter_smooth` / `SmoothQuant`) or `FA3` (`FP8`/`INT8` attention Q/K/V) quantization is enabled, the DiT adapter (`SetTimestep` hook) must sample activations uniformly across the full denoising trajectory (e.g., `8..16` Prompts $\times$ all $T$ timesteps) rather than calibrating on a single forward pass.
3. **Dual-Expert / Multi-Transformer DiT Handling (`Wan2.2` High-Noise / Low-Noise Experts)**:
   - Models like `Wan2.2` split denoising into two separate DiT backbones (`high_noise_model` for early steps and `low_noise_model` for late steps).
   - Quantize and export each sub-transformer into its own isolated subfolder with independent sensitivity rollback lists (`disable_names`).

## 3. Sensitive Module Rollback Rules for DiT (`apply_quantization` Filter)

Always keep the following DiT modules in `BF16` / `FP16` (`exclude_layers` / `disable_names`):
1. **Timestep & Conditioning Embedding MLP** (`time_embed`, `text_embed`, `guidance_embed`, `adaLN_modulation` / `scale_shift_table`): Quantizing modulation scale/shift parameters amplifies errors across the entire sequence $S$.
2. **Input Patchify (`patch_embed` / `x_embedder`) and Final Output Projection (`proj_out` / `final_layer`)**.
3. **Cross-Attention Text Projections (`add_q_proj`, `add_k_proj`, `add_v_proj` / `txt_attn`)** if text prompt adherence degrades.

## Verification Checklist

- Evaluate quantized DiT outputs against the `BF16` baseline using identical random seeds (`torch.npu.manual_seed`) on both quantitative metrics (`PSNR >= 28 dB`, `SSIM >= 0.88`, `LPPS`, `VBench`) and visual inspection for temporal flicker.
