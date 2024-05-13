#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=lora_32_3e-4
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 36 # number of cores (here 2 cores requested)
#SBATCH --time=2-00:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1,tmpfs:1000G # generic resource required (here requires 1 GPU)
#SBATCH --mem=40GB # specify memory required per node (here set to 8 GB)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yiping.ji@adelaide.edu.au
##SBATCH --gres=tmpfs:1000G
#SBATCH --signal B:USR2
#SBATCH -A strategic

source ~/.bashrc
conda activate llama7b
cd /gpfs/users/a1906566/2024/DoRA/commonsense_reasoning

python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' \
    --output_dir ./finetuned_result/dora_qv_r32_3e-4 \
    --batch_size 16  --micro_batch_size 8 --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "v_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing 

