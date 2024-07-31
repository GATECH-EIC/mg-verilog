export CUDA_VISIBLE_DEVICES=0,1,2,3

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

accelerate launch --multi_gpu generate2.py \
    --checkpoint_dir ./result_ckpt \
    --model_type "qlora" \
    --base_model "codellama/CodeLlama-7b-Instruct-hf" \
    --tokenizer_type "code_llama" \
    --cache_dir "/home/user_name/HF_cache/" \
    --hf_token "your_hf_token_if_you_want_to_use_it" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --desc_file $OUTPUT_DIR/benchmark_packaged_dataset/hdlbits_for_llm2_eval.jsonl \
    --desc_key "block_to_code_description" \
    --prompt_type "llm2_block_to_code" \
    --eval_file ../verilog_eval/data/VerilogEval_Machine.jsonl \
    --output_file $OUTPUT_DIR/data/gen.jsonl \
    --fp16 \
    --sample_k 20 \
    --result_name Test \
    --batch_size 2 
