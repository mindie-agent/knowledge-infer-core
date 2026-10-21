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
  - RLHF / GRPO / PPO rollout-trainer precision alignment and Eagle/MTP speculator
    verification on NPU
domain_path: infer-core/quantization-and-accuracy/rl-spike-and-speculators
evidence_strength: software_contract
id: unit-rl-rollout-inference-engine-mismatch-and-spike-root-cause-sop
inherits_from:
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
- unit-infer-core-mindspeed-torchtitan-5d-parallel-zero-and-recompute-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/msagent/blob/521f8ead0ebda5db5b282f0c3c4147006e01e016/skills/accuracy/spike-root-cause-analysis/SKILL.md
  source_digest: 4bff7df4bdf95c184c98c81b26036e6ae0806dd09f50208ab85005818cb5711f
  source_locator: Ascend/msagent@521f8ead0ebda5db5b282f0c3c4147006e01e016:skills/accuracy/spike-root-cause-analysis/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Root-cause analysis SOP from Ascend/msagent (`spike-root-cause-analysis`)
  for diagnosing RL training loss/grad-norm spikes caused by inference-training logprob
  divergence, MoE expert routing flips, and speculator draft mismatches.
tier: L2
title: RL Rollout vs. Trainer Logprob Mismatch, Spike Root-Cause Analysis, and Speculator
  Alignment SOP
triggers:
  error_signatures:
  - spike root cause analysis skill msagent
  - rl rollout trainer logprob kl mismatch
  keywords:
  - spike root cause analysis skill msagent
  - rl rollout trainer logprob kl mismatch
  - moe router topk tie break rl spike
  - vllm sglang rollout mindspeed trainer precision mismatch
  - spike-root-cause
  - rl-rollout
  - kl-divergence
  - logprob-mismatch
  - routed-experts
  - speculators
---
## 1. Anatomy of RL Loss & Grad-Norm Spikes on NPU (`spike-root-cause-analysis`)

In GRPO / PPO / RLHF pipelines, an **Inference Engine** (`vLLM-Ascend` or `SGLang`) generates rollout trajectories and records behavior policy log-probabilities $\log \pi_{\text{infer}}(a_t | s_t)$, while a **Training Engine** (`MindSpeed` / `FSDP`) recomputes $\log \pi_{\text{train}}(a_t | s_t)$ on the same weights to form the importance sampling ratio:
$$r_t(\theta) = \exp\big(\log \pi_{\text{train}}(a_t | s_t) - \log \pi_{\text{infer}}(a_t | s_t)\big)$$
Even when weights are identical, numerical discrepancies between inference fused kernels and training kernels cause $|\log \pi_{\text{train}} - \log \pi_{\text{infer}}|$ spikes on rare tokens, exploding $r_t(\theta)$, `kl_loss`, and `grad_norm`.

## 2. Three Primary Sources of Inference-Training Divergence on Ascend NPU

1. **MoE Router Top-K Flip (`BF16` Router Logit Jitter)**:
   - When the $k$-th and $(k+1)$-th expert router scores differ by `< 1e-3`, slight differences between the inference fused `QKV/Attention` kernel and the training `FA` kernel flip which expert is selected at Layer $l$.
   - Once a single layer routes to a different expert, all downstream layers diverge macroscopically (`KL > 0.01`).
   - **Fix**:
     1. Cast router gate computation (`x.float() @ gate_weight.float()`) and softmax/sigmoid gating to **`FP32`** in both inference and training.
     2. Enable **Rollout Routed-Expert Replay (`routed_experts` pass-through)**: Record the exact `topk_ids` selected during inference rollout and force the training forward pass to route tokens to those exact expert IDs (computing gradients through the router weights of the selected experts).
2. **RoPE & RMSNorm Precision Mismatch**:
   - Inference engines often store `cos`/`sin` tables in `BF16` while training engines compute RoPE in `FP32` and cast back. Over `16K..32K` reasoning chains, phase error accumulates.
   - **Fix**: Align RoPE `cos`/`sin` computation dtype and `npu_rms_norm` accumulation (`FP32` variance calculation) between rollout and trainer.
3. **Matmul Accumulation (`HF32` vs. `BF16`) & Speculator Draft Verification**:
   - Disable `allow_hf32` mismatches and verify that Eagle3 / MTP speculative decoding in the rollout engine uses strict lossless rejection sampling (or greedy `temperature=0` verification) so accepted rollout tokens follow the exact target model distribution $\pi_{\text{infer}}$.

## Verification Checklist

- Measure Step-0 `max(|logp_train - logp_infer|)` and mean token-level `KL(pi_infer || pi_train)` on a fixed batch of rollout trajectories before starting RL updates; healthy alignment requires `mean_kl < 1e-4` (with expert replay) or `< 1e-3` (without expert replay).
