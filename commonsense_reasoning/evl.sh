CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset boolq \
    --is_sine --s 64 --freq 6000 \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights ./finetuned_result/sine_lora_qv_r16_6k_64/checkpoint-31920|tee -a ./finetuned_result/sine_lora_qv_r16_6k_64/checkpoint-31920/boolq.txt
    