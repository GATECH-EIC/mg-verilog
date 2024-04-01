export CUDA_VISIBLE_DEVICES=0,1,2,3

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

accelerate launch --multi_gpu generate2.py \
    --checkpoint_dir /home/user_name/DAC_2024/checkpoint/llm2_new_block_summary_to_pure_code/checkpoint-9500 \
    --model_type "qlora" \
    --base_model "codellama/CodeLlama-7b-Instruct-hf" \
    --tokenizer_type "code_llama" \
    --cache_dir "/home/user_name/HF_cache/" \
    --hf_token "your_hf_token_if_you_want_to_use_it" \
    --max_new_tokens 1024 \
    --temperature 0.6 \
    --top_p 0.95 \
    --desc_file /home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/hdlbits_description_simple_description.jsonl \
    --desc_key "simple_description" \
    --prompt_type "baseline" \
    --eval_file ../verilog_eval/data/VerilogEval_Machine.jsonl \
    --output_file ./data/gen.llm2_new_block_summary_to_pure_code+simple_description.jsonl \
    --fp16 \
    --sample_k 10 \
    --result_name "llm2_new_block_summary_to_pure_code+simple_description" \
    --batch_size 2 