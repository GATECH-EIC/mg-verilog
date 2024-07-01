import os
import sys
import shutil
import argparse
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))
from embedding_lookup_utils import CodeDataset
from langchain.callbacks import get_openai_callback #with get_openai_callback() as cb:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #positional arguments
    #start_id, total_code_parts
    parser.add_argument("start_id", help="start id of the code parts", type=int)
    parser.add_argument("total_code_parts", help="total number of code parts", type=int)

    #optional arguments
    parser.add_argument("--documented_code_dir", help="documented code directory", type=str, default="/home/user_name/DAC_2024/ckpts/")
    parser.add_argument("--code_metadata_dir", help="code metadata directory", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/")
    parser.add_argument("--model", help="model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--detailed", action="store_true", help="detailed summary")
    args = parser.parse_args()
    code_part_start_id = args.start_id
    total_code_parts = args.total_code_parts
    documented_code_dir = args.documented_code_dir
    code_metadata_dir = args.code_metadata_dir
    model = args.model
    detailed = args.detailed


    dataset_metadata_dir = os.path.join(documented_code_dir, "dataset_metadata")
    if not os.path.exists(dataset_metadata_dir):
        os.makedirs(dataset_metadata_dir)

    with get_openai_callback() as cb:
        for code_part_id in range(code_part_start_id, total_code_parts):
            if not os.path.exists("{}/part{}".format(dataset_metadata_dir, code_part_id)):
                os.makedirs("{}/part{}".format(dataset_metadata_dir, code_part_id))
            src_code_dir = os.path.join(documented_code_dir, "part{}".format(code_part_id))
            codedb = CodeDataset(
                                src_code_dir,
                                bookkeeping_dir="{}/part{}/bookkeeping/".format(dataset_metadata_dir,code_part_id),
                                vectorembedding_dir="{}/part{}/vectorembedding/".format(dataset_metadata_dir, code_part_id),
                                force_refresh=False,
                                cb=cb
                                )
            csv_code_dir = os.path.join(code_metadata_dir, "part{}".format(code_part_id), "assets", "verilog", "code_and_comment_src", "csv_src", "csv_code_src")
            csv_comment_dir = os.path.join(code_metadata_dir, "part{}".format(code_part_id), "assets", "verilog", "code_and_comment_src", "csv_src", "csv_new_comment_src")
            codedb.load_and_split_code(skip_small_doc=True, split_by_line=True, based_on_code_lines_only=True, 
                                        csv_code_dir=csv_code_dir,
                                        csv_comment_dir=csv_comment_dir
                                    )
            if detailed:
                codedb.init_vectorstore(global_summary_chain_from_verilog_eval=False,
                                        global_summary_model=model,
                                        global_summary_example_cstr_json = f"{os.environ.get('DATA4AIGCHIP_HOME')}/auto_data_gen_val/preprocess_data/example_code_strings_detailed_instructions.json",
                                        global_summary_example_code_description_file= f"{os.environ.get('DATA4AIGCHIP_HOME')}/verilog_eval/descriptions/VerilogDescription_Machine.jsonl"
                                        )
                codedb.supplement_summary(block_summary_placeholding=True,force_refresh_global_summary_detailed=True, global_summary_example_desc_key="detail_description")
                codedb.save_global_summary(
                                        "{}/part{}/global_detailed_summary.json".format(dataset_metadata_dir, code_part_id)
                                        )
            else:
                codedb.init_vectorstore(global_summary_chain_from_verilog_eval=False,
                                        detailed=False,
                                        global_summary_model=model,
                                        global_summary_example_cstr_json = f"{os.environ.get('DATA4AIGCHIP_HOME')}/auto_data_gen_val/preprocess_data/example_code_strings_simple_instructions.json",
                                        global_summary_example_code_description_file= f"{os.environ.get('DATA4AIGCHIP_HOME')}/verilog_eval/descriptions/VerilogDescription_Machine.jsonl"
                                        )
                codedb.supplement_summary(block_summary_placeholding=True,force_refresh_global_summary_high_level=True, global_summary_example_desc_key="simple_description")
                codedb.save_global_summary(
                                        "{}/part{}/global_high_level_summary.json".format(dataset_metadata_dir, code_part_id)
                                        )
