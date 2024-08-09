export CUDA_VISIBLE_DEVICES=0,1,2,3

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

accelerate launch --multi_gpu generate2.py \
    --checkpoint_dir ./result_ckpt \
    --model_type "qlora" \
    --base_model "codellama/CodeLlama-7b-Instruct-hf" \
    --tokenizer_type "code_llama" \
    --cache_dir "./HF_cache/" \
    --hf_token "" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --desc_file ../verilog_eval/descriptions/VerilogDescription_Machine.jsonl \
    --desc_key "detail_description" \
    --prompt_type "baseline" \
    --eval_file ../verilog_eval/data/VerilogEval_Machine.jsonl \
    --output_file ./data/gen.jsonl \
    --fp16 \
    --sample_k 20 \
    --result_name Test \
    --batch_size 1
