---
category: sop
conditions:
  cann_versions:
  - '>=8.0.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - msModelSlim LLM/VLM/MoE model adaptation and multi-card quantization calibration
domain_path: infer-core/quantization-and-accuracy/msmodelslim-adaptation
evidence_strength: software_contract
id: unit-msmodelslim-model-adapter-and-ep-layerwise-quantization-sop
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/msagent/blob/521f8ead0ebda5db5b282f0c3c4147006e01e016/skills/quantizer/msmodelslim-model-adapt/SKILL.md
  source_digest: b11cec6261e0788528adc99a66a581e7059cf49479fe683a7cfc6d790ac59eb2
  source_locator: Ascend/msagent@521f8ead0ebda5db5b282f0c3c4147006e01e016:skills/quantizer/msmodelslim-model-adapt/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Engineering SOP from Ascend/msagent for building msModelSlim model adapters
  (`generate_model_visit` / `generate_model_forward`), MoE linear unpacking, buffer
  preservation, layerwise lazy loading, and multi-card EP calibration.
tier: L2
title: msModelSlim Model Adapter Implementation, Layerwise Lazy Loading, and EP Parallel
  Quantization SOP
triggers:
  error_signatures:
  - msmodelslim model adapt skill
  - create msmodelslim model adapter
  keywords:
  - msmodelslim model adapt skill
  - create msmodelslim model adapter
  - generate_model_visit generate_model_forward
  - msmodelslim ep parallel adaptation
  - msmodelslim buffer to nn parameter
  - msmodelslim
  - model-adapter
  - layerwise-quantization
  - ep-parallel
  - four-step-verification
  - moe-unpack
---
## 1. msModelSlim Model Adapter Core Interface Contract (`msmodelslim-model-adapt`)

To adapt a new Decoder-only LLM or VLM text backbone into `msModelSlim`, implement the model adapter class and register it in `plugin.json` / entry points:
1. **Five Required Methods**:
   - `handle_dataset(dataset, ...)`: Tokenizes calibration prompts and prepares input dicts. **Critical Rule**: Always check `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token` before batch tokenization.
   - `init_model(...)`: Instantiates the model architecture and loads weights.
   - `generate_model_visit(model)`: Yields `(name, module)` pairs in topological layer order for quantization hooks.
   - `generate_model_forward(model, inputs)`: Yields per-block forward execution steps during calibration.
   - `enable_kv_cache(model, enabled: bool)`: Toggles KV cache during block-by-block calibration (`False` when replaying cached hidden states per decoder layer).

## 2. Four Mandatory Adaptation Invariants

1. **Strict `visit` <-> `forward` Topological Alignment**:
   - Every block yielded by `generate_model_visit` must correspond 1-to-1 in order and granularity with the block executed in `generate_model_forward`. Any mismatch causes calibration hooks to capture activations from the wrong layer.
2. **MoE Expert Unpacking to Standard `nn.Linear`**:
   - If a model stores fused 3D expert tensors (`nn.Parameter` of shape `[num_experts, hidden, 2 * inter]`), unpack them in the adapter into standard `nn.ModuleList` of `nn.Linear` (`gate_proj`, `up_proj`, `down_proj`) before calibration so `msModelSlim` quantizes each expert with standard linear observers and exports vLLM-Ascend / MindIE compatible safetensors.
3. **Buffer Weight Preservation (`register_buffer` -> `nn.Parameter`)**:
   - Any persistent model buffer required at inference time (e.g., `e_score_correction_bias`, fixed rotary frequencies, or scaling factors registered via `register_buffer`) **must** be converted to `nn.Parameter(..., requires_grad=False)` in the adapter; otherwise `msModelSlim` safetensors export silently drops non-parameter buffers.
4. **Optional Layerwise Lazy Loading & Multi-Card EP Calibration**:
   - Enable layerwise lazy loading (`msmodelslim-layerwise-quant-adaptation`) only when host CPU RAM cannot hold the full FP16/BF16 checkpoint (e.g., `671B` / `1T` models).
   - When `>= 2` NPU cards are provided for MoE calibration, enable `msmodelslim-ep-parallel-adaptation` so routed experts calibrate in parallel across NPU ranks without OOM.

## 3. Four-Step Adapter Verification Gate (`msmodelslim-adapter-verification`)

1. Adapter registration & CLI discovery check (`msmodelslim quant --help` / model registry lookup).
2. 1-layer or 2-sample dry-run quantization completion without hook mismatch errors.
3. Exported `quant_model_description.json` and `*.safetensors` key completeness check (verifying unpacked MoE weights and preserved buffer parameters exist).
4. End-to-end quantized inference smoke test in `vllm-ascend` or `cann-recipes-infer`.
