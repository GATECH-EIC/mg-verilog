export CUDA_VISIBLE_DEVICES=0,1,2,3

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'

# accelerate launch --num_processes 4 generate2.py \
# # python generate.py \
#     --model_name ./gpu5/output \
#     --model_type "qlora" \
#     --base_model "codellama/CodeLlama-7b-Instruct-hf" \
#     --fp16 \
#     --sample_k 20 \
#     --result_name Test \
#     --batch_size 2
#     # --bf16 \
#     # --desc_file ./verilog_eval/desc_mini.jsonl \
#     # --eval_file ./verilog_eval/eval_mini.jsonl \

accelerate launch --multi_gpu --num_processes 4 generate2_vanilla.py \
    --model_type "qlora" \
    --base_model "codellama/CodeLlama-7b-Instruct-hf" \
    --bf16 \
    --sample_k 10 \
    --result_name Test \
    --batch_size 1 \
    # --desc_file ./verilog_eval/desc_mini.jsonl \
    # --eval_file ./verilog_eval/eval_mini.jsonl \
    # --skip_gen \
    # --bf16 \
