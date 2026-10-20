---
category: workflow
conditions:
  domain: ascend-inference-and-training
  npu_platform:
  - Atlas_A2
  - Atlas_A3
  - Ascend_950
  - Ascend_310P
domain_path: infer-core/training-and-alignment/concepts/rlhf-post-training
evidence_strength: software_contract
id: unit-infer-core-verl-grpo-dapo-rollout-switch-and-sam-speculative-workflow
inherits_from:
- unit-infer-core-mindspeed-torchtitan-5d-parallel-zero-and-recompute-axioms
- unit-infer-core-cann-executor-sglang-pd-and-kv-manager-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/train_rollout_switch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/grpo_dapo.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/rollout_rebalance.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/partial_rollout.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/sam_speculative.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/transferdock_data_scheduling.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/agent_rl_loop.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/techniques/length_aware_resampler.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/guides/recipes/deepseek_r1_rl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/guides/recipes/qwen3_rl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/guides/recipes/qwen3_code_toolcall_rl.md
  source_digest: 8a9681360ff19b2484b52c955f6f9c96ecc2f6e054e5460aa8a74b85107bde15
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/training/concepts/techniques/train_rollout_switch.md
  source_type: cannbot_training_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: 'Ascend RLHF/Post-Training workflow: veRL/MindSpeed-RL GRPO/DAPO, co-located
  Train/Rollout resharding, Partial Rollout, and SAM speculative decoding'
triggers:
  error_signatures:
  - Co-located veRL + MindSpeed + vLLM-Ascend GRPO training OOMs during Train-to-Rollout
    transition when optimizer states and rollout KV cache coexist in HBM
  - Long-tail reasoning rollout sequences (32K+ tokens) stall synchronous RL step
    barriers across DP workers
  - Multi-turn tool-calling Code RL blocks NPU workers on synchronous sandbox execution
  keywords:
  - train_rollout_switch
  - veRL
  - MindSpeed-RL
  - GRPO
  - DAPO
  - rollout_rebalance
  - partial_rollout
  - SAM_speculative
  - TransferDock
  - ScaleBox
---
# Ascend RLHF/Post-Training workflow: veRL/MindSpeed-RL GRPO/DAPO, co-located Train/Rollout resharding, Partial Rollout, and SAM speculative decoding

## Objective

Orchestrate high-throughput Reinforcement Learning (`GRPO` / `DAPO`) and multi-turn tool-use Agent RL for reasoning models (`DeepSeek-R1`, `Qwen3-235B-A22B`, `Qwen3-30B-A3B`) across `veRL`, `MindSpeed-RL`, `vLLM-Ascend`, and `ScaleBox` on Ascend NPU clusters.

## Workflow Steps

1. **Manage Co-Located `Train <-> Rollout Switch` HBM Lifecycle and Weight Resharding (`train_rollout_switch.md`, `deepseek_r1_rl.md`, `qwen3_rl.md`).**
   - In co-located RL (`HybridEngine`), the training engine (`MindSpeed` / `FSDP2` with high TP/PP/CP/EP) and the rollout inference engine (`vLLM-Ascend` with smaller TP and large DP) share the same physical NPUs:
     1. **Before Rollout:** Offload optimizer states and gradients to pinned Host DRAM, reshard actor weights from training `[PP, TP_train, EP_train]` layout to rollout `[TP_infer, EP_infer]` layout via direct device-to-device (`D2D`) collectives, free training activation buffers, and wake up `vLLM-Ascend`'s `CaMem` / KV cache sleep pool.
     2. **After Rollout:** Put `vLLM-Ascend` KV cache and weights into sleep/offload mode, release NPU allocator cache, and reload training shards and optimizer states from Host DRAM.

2. **Eliminate Long-Tail Rollout Stragglers via `Rollout Rebalance`, `Partial Rollout`, and `Length-Aware Resampler` (`rollout_rebalance.md`, `partial_rollout.md`, `length_aware_resampler.md`, `transferdock_data_scheduling.md`).**
   - Reasoning rollouts (`32K..64K` tokens) exhibit extreme sequence-length skew where a few long responses hold the entire cluster at the step barrier:
     - **Length-Aware Resampler:** Sort and bucket prompts by historical response length before rollout dispatch.
     - **Rollout Rebalance:** Dynamically migrate remaining active long-tail sequences from lightly loaded `vLLM-Ascend` DP engines into consolidated batches.
     - **Partial Rollout:** Pause unfinished ultra-long generations at a token budget ceiling, save their prefix tokens, and resume them in the next RL iteration.
     - **TransferDock:** Decouple rollout sample production from actor/ref log-prob consumption via an asynchronous distributed experience buffer.

3. **Accelerate Rollout Decoding with Lossless Suffix Automaton (`SAM`) Speculative Decoding and Async `ScaleBox` Tool Loops (`sam_speculative.md`, `agent_rl_loop.md`, `qwen3_code_toolcall_rl.md`).**
   - Because `GRPO` generates $G$ (`8..16`) trajectories per prompt and reasoning chains contain repetitive code/math sub-expressions, enable training-free **SAM (Suffix Automaton)** or Eagle-3 speculative decoding in `vLLM-Ascend` to draft multi-token continuations with zero draft-model HBM overhead.
   - For Code/Tool Agent RL (`verl-retool` + `ScaleBox`), drive rollout via `asyncLLM` + per-trajectory `agent_loop` coroutines so NPU decode batches continue serving active trajectories while paused trajectories await remote `ScaleBox` sandbox execution and `Rule Reward Verifier` scoring.
