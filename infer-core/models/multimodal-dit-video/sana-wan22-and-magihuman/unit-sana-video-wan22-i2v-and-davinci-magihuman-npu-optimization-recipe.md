---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - SANA-Video, Wan2.2-I2V, and daVinci-MagiHuman video diffusion and VAE inference
domain_path: infer-core/models/multimodal-dit-video/sana-wan22-and-magihuman
evidence_strength: reproduced_empirical
id: unit-sana-video-wan22-i2v-and-davinci-magihuman-npu-optimization-recipe
inherits_from:
- unit-infer-core-dit-vae-multimodal-and-hstu-architecture-axioms
- unit-infer-core-tp-dp-ep-cp-sp-ulysses-and-cfg-parallel-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/sana-video/SANA-Video_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/wan2.2-i2v/wan2.2-i2v_optimization.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/davinci-magihuman/davinci-magihuman_optimization.md
  source_digest: 1e000dff3a6f0b7d1902e7c96794dc5d45b5241b814434b79768ae55fe39c2e7
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/sana-video/SANA-Video_optimization.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Video generation optimization playbook for SANA-Video, Wan2.2-I2V, and daVinci-MagiHuman
  covering 1x1 Conv2d-to-Matmul Cube acceleration, GLUMBConv temporal axis swap, and
  FIA/RoPE/VAE adaptations.
tier: L2
title: SANA-Video, Wan2.2-I2V, and daVinci-MagiHuman Video Generation NPU Optimization
  Playbook
triggers:
  error_signatures:
  - sana video optimization ascend
  - wan2.2 i2v optimization npu
  keywords:
  - sana video optimization ascend
  - wan2.2 i2v optimization npu
  - davinci magihuman video generation ascend
  - 1x1 conv2d to linear matmul replacement
  - glumbconv temporal conv2d hw axis swap
  - sana-video
  - wan2.2-i2v
  - davinci-magihuman
  - conv2d-to-matmul
  - temporal-conv-swap
  - fia
  - rotary-mul
---
## 1. SANA-Video Convolutional & Linear Attention Optimizations (`SANA-Video_optimization.md`)

`SANA-Video` relies heavily on `GLUMBConv` (Inverted Residual / Pointwise + Depthwise + Temporal Convolution) blocks alongside linear/chunk causal attention:
1. **`1x1 Conv2d` (`stride=1, padding=0, groups=1`) to `F.linear` MatMul Transformation**:
   - On Ascend NPU, `1x1 Conv2d` in `NCHW` layout triggers implicit `Im2Col` / format conversion (`NC1HWC0`) overhead before reaching the Cube unit.
   - **Optimization**: Permute activation from `[B, C_in, H, W]` to `[B, H * W, C_in]`, reshape weight from `[C_out, C_in, 1, 1]` to `[C_out, C_in]`, execute `F.linear(x_2d, weight_2d, bias)` directly on the Cube engine, and permute back to `[B, C_out, H, W]`.
2. **Temporal `Conv2d` `HW`-Axis Swap in `GLUMBConvTemp` (`patch_temporal_conv_swap`)**:
   - A temporal convolution with kernel `(t_kernel_size, 1)` over tensor `[B, C, T, H*W]` strides across non-contiguous memory along the `T` axis.
   - **Optimization**: Replace `(t_kernel_size, 1)` with a swapped `nn.Conv2d` using `kernel_size=(1, t_kernel_size)` and `padding=(0, t_padding)`, permute the loaded checkpoint weight via `weight.permute(0, 1, 3, 2)`, and transpose the last two dimensions `(T, H*W) <-> (H*W, T)` around the call so the convolution filter runs along the contiguous innermost dimension.
3. **Fused Normalization & Attention**:
   - Replaces small norm ops with `torch_npu.npu_rms_norm` and `npu_fused_infer_attention_score`.

## 2. Wan2.2-I2V & daVinci-MagiHuman Optimization (`wan2.2-i2v` & `davinci-magihuman`)

1. **Global NPU Runtime Flags**:
   - Set `torch_npu.npu.set_compile_mode(jit_compile=False)` and `torch.npu.config.allow_internal_format = False` at entry (`generate.py` / `sample_video.py`) to prevent dynamic ACL compilation stalls and unintended `NZ` internal format conversions across DiT and 3D VAE layers.
2. **Variable-Length / Full-Sequence `npu_fused_infer_attention_score` (`FIA`)**:
   - In `Wan2.2-I2V` (`wan/modules/attention.py`) and `daVinci-MagiHuman` (`dit_module.py` and `vae2_2_module.py`), replace FlashAttention with `torch_npu.npu_fused_infer_attention_score` (`input_layout="BNSD"` or `"BSND"`, `pre_tokens=65535`, `next_tokens=65535 if not causal else 0`, `sparse_mode=0`, `inner_precise=0`, casting inputs to contiguous `bfloat16`).
3. **3D RoPE & Modulation Fusion (`npu_rotary_mul` & `npu_rms_norm`)**:
   - Replaces complex-valued `torch.view_as_complex` / `polar` rotary embeddings in video DiTs (`rope_apply`) with real-valued `torch_npu.npu_rotary_mul` (`cos`, `sin` tables cached on NPU) to eliminate complex dtype fallback to CPU or slow Vector emulation.

## Verification Checklist

- After patching `t_conv_swap` in `SANA-Video`, ensure `block.mlp.t_conv_swap.weight = nn.Parameter(block.mlp.t_conv.weight.permute(0, 1, 3, 2))` runs **after** `load_state_dict()`.
- Confirm `torch.view_as_complex` is completely removed from the DiT RoPE path so zero operators fall back to CPU.
