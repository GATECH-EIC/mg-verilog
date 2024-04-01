export CUDA_VISIBLE_DEVICES=0,1,2,3

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

accelerate launch --multi_gpu --num_processes 4 generate2.py \
    --checkpoint_dir /home/user_name/DAC_2024/checkpoint/llm1_high_level_summary_to_block_summary_dataset_skip_single_blocks_usage_summary_combined_better_formating_2/checkpoint-9000 \
    --model_type "qlora" \
    --base_model "meta-llama/Llama-2-7b-chat-hf" \
    --tokenizer_type "llama" \
    --cache_dir "/home/user_name/HF_cache/" \
    --hf_token "your_hf_token_if_you_want_to_use_it" \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.1 \
    --top_k 40 \
    --repetition_penalty 1.17 \
    --desc_file ../verilog_eval/descriptions/VerilogDescription_Machine.jsonl \
    --desc_key "detail_description" \
    --prompt_type "llm1" \
    --eval_file ../verilog_eval/data/VerilogEval_Machine.jsonl \
    --output_file ./data/gen.llm1.jsonl \
    --fp16 \
    --sample_k 10 \
    --result_name Test \
    --batch_size 2  \
    --skip_iverilog 