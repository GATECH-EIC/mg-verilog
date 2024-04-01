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
    parser.add_argument("--block_line_length", help="block line length", type=int, default=10)
    parser.add_argument("--model", help="model", type=str, default="gpt-3.5-turbo-1106")
    args = parser.parse_args()
    code_part_start_id = args.start_id
    total_code_parts = args.total_code_parts
    documented_code_dir = args.documented_code_dir
    block_line_length = args.block_line_length
    model = args.model

    dataset_metadata_dir = os.path.join(documented_code_dir, "dataset_metadata")
    if not os.path.exists(dataset_metadata_dir):
        os.makedirs(dataset_metadata_dir)
        
    with get_openai_callback() as cb:
        for code_part_id in range(code_part_start_id, total_code_parts):
            if not os.path.exists("{}/part{}".format(dataset_metadata_dir, code_part_id)):
                os.makedirs("{}/part{}".format(dataset_metadata_dir, code_part_id))
            src_code_dir = os.path.join(documented_code_dir, "part_{}".format(code_part_id), "documented_code")
            codedb = CodeDataset(
                                src_code_dir,
                                bookkeeping_dir="{}/part{}/bookkeeping/".format(dataset_metadata_dir,code_part_id),
                                vectorembedding_dir="{}/part{}/vectorembedding/".format(dataset_metadata_dir, code_part_id),
                                force_refresh=False,
                                cb=cb
                                )
            csv_code_dir = os.path.join(documented_code_dir, "part_{}".format(code_part_id), "assets", "verilog", "code_and_comment_src", "csv_src", "csv_code_src")
            csv_comment_dir = os.path.join(documented_code_dir, "part_{}".format(code_part_id), "assets", "verilog", "code_and_comment_src", "csv_src", "csv_new_comment_src")
            codedb.load_and_split_code(skip_small_doc=True, split_by_line=True,
                                        line_length=block_line_length,
                                        based_on_code_lines_only=True, 
                                        csv_code_dir=csv_code_dir,
                                        csv_comment_dir=csv_comment_dir
                                    )
            codedb.init_vectorstore(block_summary_model=model)
            codedb.supplement_summary(block_summary_placeholding=False)
            codedb.save_block_summary(
                                        "{}/part{}/block_summary.json".format(dataset_metadata_dir, code_part_id),
                                        split_by_line = True
                                    )
