export CUDA_VISIBLE_DEVICES=0,1,2,3

# accelerate launch qlora.py

WORLD_SIZE=4 torchrun --nproc_per_node=4 qlora.py \
                      --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
                      --source_max_len 1024 \
                      --target_max_len 2048 \
                      --output_dir ./data/Verilog_code_generation/llm1_new_verilogeval_global_summary_to_block_summary_skip_single_block \
                      --dataset_dir /data/user_name_data/user_name/sft_dataset/llm1_new_verilogeval_global_summary_to_block_summary_skip_single_block \
                      --cache_dir /data/user_name_data/user_name/HF_cache \
                      --gradient_accumulation_steps 4 \
                      --save_steps 500

