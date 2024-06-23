TEST_DIR=$1
OUTPUT_DIR=$2
CURRENT_DIR=$(cd $(dirname $0); pwd)
export DATA4AIGCHIP_HOME=$(cd $CURRENT_DIR/..; pwd)
echo "DATA4AIGCHIP_HOME=$DATA4AIGCHIP_HOME"
echo "TEST_DIR=$TEST_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"



python ../auto_data_gen_val/preprocess_data/process_data/preprocess.py $OUTPUT_DIR/raw_code -customized_dataset_dir $TEST_DIR

python ../auto_data_gen_val/utils.py \
    --src_code_dir $OUTPUT_DIR/raw_code \
    --src_code_metadata_file $OUTPUT_DIR/module_inst.json \
    --output_dir $OUTPUT_DIR/partitioned_dataset_output_path/ \
    --shared_lib_dir $OUTPUT_DIR/directory_to_store_common_modules/ \
    --output_code_metadata_dir $OUTPUT_DIR/output_dir_for_code_metadata/ \
    --output_code_metadata_file codes.json \
    --module_to_task_id_map_file $OUTPUT_DIR/module_name_to_task_id_mapping.json


python ../auto_data_gen_val/line_by_line_comments_gen.py \
    --total_parts 1 \
    --output_dir $OUTPUT_DIR/documented_code \
    --src_code_dir $OUTPUT_DIR/partitioned_dataset_output_path/ \
    --code_metadata_dir $OUTPUT_DIR/output_dir_for_code_metadata/ \
    --code_lib_path $OUTPUT_DIR/directory_to_store_common_modules/ \
    --code_vec_store $OUTPUT_DIR/code_vec_store/test/ \
    --discard_original_comment


python ../auto_data_gen_val/gen_block_summaries.py 0 1 \
    --code_metadata_dir $OUTPUT_DIR/output_dir_for_code_metadata/ \
    --documented_code_dir $OUTPUT_DIR/documented_code \
    --block_line_length 10 \
    --model gpt-4-turbo



python ../auto_data_gen_val/gen_global_summary.py 0 1 \
    --code_metadata_dir $OUTPUT_DIR/output_dir_for_code_metadata/ \
    --documented_code_dir $OUTPUT_DIR/documented_code \
    --model gpt-4-turbo \
    --detailed


python ../auto_data_gen_val/gen_global_summary.py 0 1 \
    --code_metadata_dir $OUTPUT_DIR/output_dir_for_code_metadata/ \
    --documented_code_dir $OUTPUT_DIR/documented_code \
    --model gpt-4-turbo