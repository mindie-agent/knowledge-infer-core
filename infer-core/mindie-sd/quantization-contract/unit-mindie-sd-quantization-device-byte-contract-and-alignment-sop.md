---
schema: mindie-graded-unit/1
id: unit-mindie-sd-quantization-device-byte-contract-and-alignment-sop
title: MindIE-SD quantization device byte contract reverse-engineering and Triton-Ascend bit-exact alignment SOP
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: infer-core/mindie-sd/quantization-contract
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-infer-core-atlas-a3-ascend950-and-kernel-dsl-platform-axioms
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "fused quantizer payload differs from aclnn reference on ~58% or ~0.1% of bytes"
    - "triton-ascend x.to(tl.bfloat16).to(tl.float32) silently eliminated as no-op"
    - "8-bit conversion wrap-around garbage when clamping is removed"
  keywords:
    - quantization-dev
    - MXFP8
    - e8m0
    - e4m3
    - RNE
    - ieee_rounding
    - bitcast
    - wrap-around
    - quant_four_over_six_a5
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/quantization-dev/SKILL.md
  source_digest: 907b981d252520f4e9d3e4b64e4ead918b9f2aff5efa2eeaa16093e358b24ba9
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/quantization-dev/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

When authoring or fusing a custom quantization kernel (`Triton-Ascend`, `AscendC`, or `Catlass`) that must match CANN `aclnn` quantizers byte-for-byte, reverse-engineer and verify the three binary degrees of freedom (**scale granularity, rounding mode, and overflow/degenerate behavior**) in order:

1. **Match the `MXFP8` (`e8m0` Scale + `e4m3` Payload) Device Byte Contract**:
   - **Exact Exponent Extraction (`byte = 119 + floor(log2(amax))`)**:
     - `amax` is normalized into $[256, 512)$, not $\le 448$ (`ceil(log2(amax / 448))` fails on values such as `450`, `1000`, `230`).
     - Extract the `e8m0` byte directly from the FP32 exponent bitfield without floating-point `log2`:
       ```python
       expf = (bits >> 23) & 0xFF
       byte = clamp(expf - 8, 0, 254)  # value(byte) = 2 ** (byte - 127)
       ```
   - **Scale Tensor Layout (`[T, ceil(K/32/2), 2]`)**: Despite the trailing `[..., 2]` shape annotation, the physical memory layout is **contiguous row-major identity mapping** (group `g` resides at flat offset `g`). Do **not** apply a deswizzle permutation.
   - **Payload Rounding & Degenerate Block Rules**:
     - Payload uses **Round-to-Nearest-Even (RNE)** (`RNE(bf16_val / 2**(byte - 127))`) and saturates finite overflow at $\pm 448$ (`byte = 126` / `254`).
     - If a 32-element block is all-zero or `amax < 2^-119` (all elements are BF16 subnormals), `scale byte = 0` and the entire block payload flushes to `0`.
     - If any element in the block is `±inf` or `nan`, `scale byte = 255` and the **entire block payload is zeroed**.
   - **Quantize Post-BF16 Values, Not FP32 Intermediates**: When fusing `Norm + Quant`, round the normalized activation to BF16 before computing `amax` and quantizing; using FP32 intermediates shifts `amax` across power-of-two boundaries on ~1% of blocks.

2. **Match the `INT8` Per-Block Dynamic Quantization Contract**:
   - **Align Scale Granularity First**: Verify whether the reference quantizer reduces per `(head, 64-token block)` ($64 \times 128 = 8192$ elements) vs per tile. Comparing rounding modes while scale granularity is mismatched yields ~58% byte differences across all rounding modes and masks the true rounding behavior.
   - **Integer RNE Magic-Number Rounding**:
     ```python
     magic = tl.full((1,), 12582912.0, tl.float32)  # 1.5 * 2**23
     q = tl.cast((v + magic) - magic, tl.int32)
     q = tl.minimum(tl.maximum(q, -127), 127)
     ```

3. **Enforce the Three Universal `Triton-Ascend` Bit-Exact Implementation Rules**:
   - **Rule A — Divide by a Memory-Loaded Scale (Never Multiply by Reciprocal)**:
     - Computing `amax / 448.0` inline inside a `triton-ascend` kernel is strength-reduced by the compiler into reciprocal multiplication (`~2 ulp` error), silently flipping `< 0.1%` of `e4m3` payload bytes. Load `scale` from memory (`x / scale`) or call `tl.fdiv(x, scale, ieee_rounding=True)`.
   - **Rule B — Use Integer Bitwise RNE for BF16 Rounding in Fused Kernels**:
     - In `triton-ascend`, `modulated.to(tl.bfloat16).to(tl.float32)` is **silently eliminated as a no-op** by the compiler (0% match on BF16 midpoints, corrupting several percent of quantized payload bytes). Always perform hardware-exact BF16 RNE via integer bit manipulation:
       ```python
       b = modulated.to(tl.int32, bitcast=True)
       m = ((b + 0x7FFF + ((b >> 16) & 1)) & -65536).to(tl.float32, bitcast=True)
       ```
   - **Rule C — Treat 8-Bit Narrow Casts as Wrap-Around, Not Hardware Saturation**:
     - Vector 8-bit narrowing instructions wrap around modulo 256 when inputs exceed $[-128, 127]$. Removing explicit `clamp` is only legal when mathematical normalization (`scale = amax / 127`) strictly guarantees $|x / \text{scale}| \le 127$, and this invariant must be documented in the kernel header.
