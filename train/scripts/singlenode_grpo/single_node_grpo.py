#!/usr/bin/env python3
"""
Single Node GRPO (Group Relative Policy Optimization) for GSM8K
Based on existing PPO implementation in launch_training.py
"""

import os
import sys

def main():
    # Environment setup
    GPUS_PER_NODE = 8
    NNODES = 1
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ENTITY_NAME")
    WANDB_PROJECT_NAME = "competition_verl_grpo"
    WANDB_RUN_NAME = "llama3.2_GRPO_single_node"
    
    # GRPO specific arguments based on official VERL example
    args = [
        # GRPO algorithm specification
        "algorithm.adv_estimator=grpo",
        
        # Data configuration
        "data.train_files=/home/Competition2025/P04/shareP04/data/gsm8k/train.parquet",
        "data.val_files=/home/Competition2025/P04/shareP04/data/gsm8k/test.parquet",
        "data.train_batch_size=256",  # Adjusted for GSM8K
        "data.max_prompt_length=512",
        "data.max_response_length=512",  # Increased for mathematical reasoning
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        
        # Model configuration - Actor/Rollout/Reference
        "actor_rollout_ref.model.path=/home/Competition2025/P04/shareP04/models/Llama-3.2-1B-Instruct",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
        "actor_rollout_ref.rollout.n=5",  # Generate 5 responses per prompt for GRPO
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        
        # Algorithm configuration
        "algorithm.use_kl_in_reward=False",
        
        # Trainer configuration
        "trainer.critic_warmup=0",
        "trainer.logger=[\"console\",\"wandb\"]",
        f"trainer.project_name={WANDB_PROJECT_NAME}",
        f"trainer.experiment_name={WANDB_RUN_NAME}",
        f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
        f"trainer.nnodes={NNODES}",
        "trainer.save_freq=10",
        "trainer.test_freq=5",
        f"trainer.default_local_dir={os.environ['HOME']}/training/single_node/grpo/checkpoints",
        "trainer.total_epochs=15",
    ]
        
    # # Set Ray environment for single node
    # os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1" 
    # os.environ["RAY_ADDRESS"] = ""
    # os.environ["RAY_DEDUP_LOGS"] = "0"

    
    # Import and run GRPO using main_ppo with adv_estimator=grpo
    from verl.trainer import main_ppo
    sys.argv = ["verl.trainer.main_ppo"] + args
    main_ppo.main()

if __name__ == "__main__":
    main()