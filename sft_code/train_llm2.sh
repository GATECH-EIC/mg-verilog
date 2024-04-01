export CUDA_VISIBLE_DEVICES=0,1,2,3

# accelerate launch qlora.py

WORLD_SIZE=4 torchrun --nproc_per_node=4 qlora.py \
                      --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
                      --source_max_len 2048 \
                      --target_max_len 1024 \
                      --output_dir ./data/Verilog_code_generation/llm2_block_summary_plus_new_verilogeval_global_summary_to_pure_code \
                      --dataset_dir /data/user_name_data/user_name/sft_dataset/llm2_block_summary_plus_new_verilogeval_global_summary_to_pure_code \
                      --cache_dir /data/user_name_data/user_name/HF_cache \
                      --gradient_accumulation_steps 4 \
                      --save_steps 500

