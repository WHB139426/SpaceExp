#!/bin/bash

# Distributed training configuration
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# DeepSpeed configuration
deepspeed_config=/data4/haibo/code/SpaceExp/sft/configs/zero3.json
export WANDB_PROJECT=SpaceExp

# Model configuration
model_name_or_path=/data4/haibo/weights/Qwen3.5-9B

# Data configuration
data_root=/data4/haibo/data/3D_Env

# trajectory_type=multi_step
# observation_source=render
# trajectory_json=/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_real_multi_step.json
# observation_root=/data4/haibo/data/3D_Env/train_traj_real_multi_step_renders

# trajectory_type=multi_step
# observation_source=frame
# trajectory_json=/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_real_multi_step.json
# observation_root=/data4/haibo/data/3D_Env

trajectory_type=single_step
observation_source=render
trajectory_json=/data4/haibo/code/SpaceExp/outputs/benchmark_train_traj_sim_single_step.json
observation_root=/data4/haibo/data/3D_Env/train_traj_sim_single_step_renders

# Training hyperparameters
batch_size=2
grad_accum_steps=4
learning_rate=2e-5
num_train_epochs=1

# Output configuration
run_name=Qwen3.5-9B-nav-${trajectory_type}-${observation_source}
output_dir=/data4/haibo/code/SpaceExp/outputs/sft/${run_name}

# Training arguments
entry_file=/data4/haibo/code/SpaceExp/sft/training/train.py
args="
    --deepspeed ${deepspeed_config}
    --model_name_or_path ${model_name_or_path}
    --data_root ${data_root}
    --trajectory_json ${trajectory_json}
    --observation_root ${observation_root}
    --trajectory_type ${trajectory_type}
    --observation_source ${observation_source}
    --bf16
    --gradient_checkpointing true
    --per_device_train_batch_size ${batch_size}
    --gradient_accumulation_steps ${grad_accum_steps}
    --learning_rate ${learning_rate}
    --num_train_epochs ${num_train_epochs}
    --lr_scheduler_type cosine
    --warmup_steps 0.03
    --max_grad_norm 1
    --dataloader_num_workers 4
    --output_dir ${output_dir}
    --run_name ${run_name}
    --save_strategy steps
    --save_steps 100
    --save_total_limit 1
    --logging_steps 1
    --report_to wandb
"

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \
                    --nnodes=1 \
                    --master_port=${MASTER_PORT} \
                    ${entry_file} ${args}




# CUDA_VISIBLE_DEVICES=0 \
# VLLM_ENGINE_READY_TIMEOUT_S=3600 \
# vllm serve /data4/haibo/code/SpaceExp/outputs/sft/Qwen3.5-9B-nav-multi_step-render \
#   --served-model-name qwen3.5-9b \
#   --host 127.0.0.1 \
#   --port 8001 \
#   --tensor-parallel-size 1 \
#   --max-model-len 262144 \
#   --max-num-seqs 1 \
#   --gpu-memory-utilization 0.50 \
#   --reasoning-parser qwen3 \
#   --enable-prefix-caching \
#   --limit-mm-per-prompt '{"image":100,"video":0}'
