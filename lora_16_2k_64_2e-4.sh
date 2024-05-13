#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=sine_lora_16_2k_64_2e-4
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
    --output_dir ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4 \
    --batch_size 16  --micro_batch_size 8 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "v_proj"]' \
    --lora_r 16 --lora_alpha 32 --use_gradient_checkpointing \
    --is_sine --freq 2000 --s 64 \

python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --is_sine --freq 2000 --s 64 \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/boolq.txt


python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --is_sine --freq 2000 --s 64 \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/piqa.txt 


python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --is_sine --freq 2000 --s 64 \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/social_i_qa.txt

python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --is_sine --freq 2000 --s 64 \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/hellaswag.txt 


python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --is_sine --freq 2000 --s 64 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/winogrande.txt \

python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --is_sine --freq 2000 --s 64 \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/ARC-Challenge.txt \

python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --is_sine --freq 2000 --s 64 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/ARC-Easy.txt \

python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --is_sine --freq 2000 --s 64 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4|tee -a ./finetuned_result/sine_lora_qv_r16_2k_64_2e-4/openbookqa.txt \
