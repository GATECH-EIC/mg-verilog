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
    args = parser.parse_args()
    code_part_start_id = args.start_id
    total_code_parts = args.total_code_parts

    dataset_metadata_dir = "./dataset_metadata/"
    desc_key = "detail_description"
    if not os.path.exists(dataset_metadata_dir):
        os.makedirs(dataset_metadata_dir)

    with get_openai_callback() as cb:
        for code_part_id in range(code_part_start_id, total_code_parts):
            if not os.path.exists("{}/part{}".format(dataset_metadata_dir, code_part_id)):
                os.makedirs("{}/part{}".format(dataset_metadata_dir, code_part_id))
            codedb = CodeDataset(
                                "/home/user_name/DAC_2024/ckpts/test_10_30_{}_complete/documented_code/".format(code_part_id),
                                bookkeeping_dir="{}/part{}/bookkeeping/".format(dataset_metadata_dir,code_part_id),
                                vectorembedding_dir="{}/part{}/vectorembedding/".format(dataset_metadata_dir, code_part_id),
                                force_refresh=False,
                                cb=cb
                                )
            codedb.load_and_split_code(skip_small_doc=True, split_by_line=True, based_on_code_lines_only=True, 
                                        csv_code_dir="/home/user_name/DAC_2024/ckpts/test_10_30_{}_complete/assets/verilog/code_and_comment_src/csv_src/csv_code_src".format(code_part_id),
                                        csv_comment_dir="/home/user_name/DAC_2024/ckpts/test_10_30_{}_complete/assets/verilog/code_and_comment_src/csv_src/csv_new_comment_src".format(code_part_id)
                                    )
            codedb.init_vectorstore()
            codedb.supplement_summary(block_summary_placeholding=True,
                                      force_refresh_global_summary=True, 
                                      global_summary_example_desc_key=desc_key)

            codedb.save_global_summary(
                                    "{}/part{}/global_summary.json".format(dataset_metadata_dir, code_part_id)
                                    )
