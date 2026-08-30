#!/usr/bin/env bash

set -euo pipefail
set -x

# Single-GPU AgentOPSD run for physical GPU 1.
# The default HF rollout reuses the FSDP model and avoids loading a second
# inference engine. Set ENGINE=vllm only when the GPU has enough free memory.
export CUDA_VISIBLE_DEVICES=1
export ALFWORLD_DATA="/home/shuixia/users/houguoqiang/code/datasets/alfworld"

ENGINE="${ENGINE:-hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ASSET_DATA_DIR="/home/shuixia/users/houguoqiang/code/datasets"
ASSET_WEIGHTS_DIR="/home/shuixia/users/houguoqiang/code/weights"
MODEL_PATH="${ASSET_WEIGHTS_DIR}/Qwen2.5-7B-Instruct"
TRAIN_DATA="${ASSET_DATA_DIR}/verl-agent/text/train.parquet"
VAL_DATA="${ASSET_DATA_DIR}/verl-agent/text/test.parquet"

# Small batches keep the single-GPU smoke/training profile practical.
train_data_size="${TRAIN_DATA_SIZE:-8}"
val_data_size="${VAL_DATA_SIZE:-8}"
group_size="${GROUP_SIZE:-8}"
num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

# Full-parameter 7B Adam training is not suitable for one 48 GB GPU.
# LoRA keeps the base Qwen2.5-7B weights frozen while preserving the OPSD path.
lora_rank="${LORA_RANK:-16}"
lora_alpha="${LORA_ALPHA:-16}"
v0_prior="${V0_PRIOR:-0.5}"
mult_lambda="${MULT_LAMBDA:-0.5}"
granularity="${GRANULARITY:-turn}"
skill_all="${SKILL_ALL:-false}"

experiment_name="${AGENTOPSD_METHOD_NAME:-AgentOPSD}_alfworld_gpu1_${granularity}_lora${lora_rank}"

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
    "$PYTHON_BIN" -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "${ASSET_DATA_DIR}/verl-agent" \
        --train_data_size "$train_data_size" \
        --val_data_size "$val_data_size"
fi

"$PYTHON_BIN" -m verl.trainer.main_opsd \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    local_assets.model_path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="$lora_rank" \
    actor_rollout_ref.model.lora_alpha="$lora_alpha" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.actor.fsdp_config.single_gpu_lora=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +algorithm.opsd.v0_prior="$v0_prior" \
    +algorithm.opsd.belief_mult=true \
    +algorithm.opsd.mult_lambda="$mult_lambda" \
    +algorithm.opsd.granularity="$granularity" \
    +algorithm.opsd.signed=true \
    +algorithm.opsd.skills_dir=skills/alfworld \
    +algorithm.opsd.skill_all="$skill_all" \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    trainer.device=cuda \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name="$experiment_name" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "$@"
