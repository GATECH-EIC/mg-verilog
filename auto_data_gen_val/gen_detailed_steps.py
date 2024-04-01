import os
import sys
import shutil
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))
from embedding_lookup_utils import CodeDataset
from langchain.callbacks import get_openai_callback #with get_openai_callback() as cb:

if __name__ == "__main__":
    dataset_metadata_dir = "./dataset_metadata/"
    if not os.path.exists(dataset_metadata_dir):
        os.makedirs(dataset_metadata_dir)

    total_code_parts = 6
    code_part_start_id = 2
    with get_openai_callback() as cb:
        for code_part_id in range(code_part_start_id, total_code_parts):
            if not os.path.exists("{}/part{}".format(dataset_metadata_dir, code_part_id)):
                os.makedirs("{}/part{}".format(dataset_metadata_dir, code_part_id))
            codedb = CodeDataset(
                                "/home/user_name/DAC_2024/ckpts_test/test_10_30_{}_complete/documented_code/".format(code_part_id),
                                bookkeeping_dir="{}/part{}/bookkeeping/".format(dataset_metadata_dir,code_part_id),
                                vectorembedding_dir="{}/part{}/vectorembedding/".format(dataset_metadata_dir, code_part_id),
                                force_refresh=False,
                                cb=cb
                                )
            codedb.load_and_split_code(skip_small_doc=True, split_by_line=True, based_on_code_lines_only=True, 
                                        csv_code_dir="/home/user_name/DAC_2024/ckpts_test/test_10_30_{}_complete/assets/verilog/code_and_comment_src/csv_src/csv_code_src".format(code_part_id),
                                        csv_comment_dir="/home/user_name/DAC_2024/ckpts_test/test_10_30_{}_complete/assets/verilog/code_and_comment_src/csv_src/csv_new_comment_src".format(code_part_id)
                                    )
            codedb.init_vectorstore()
            codedb.supplement_detailed_steps()

            codedb.save_detail_steps(
                                    "{}/part{}/detailed_steps.json".format(dataset_metadata_dir, code_part_id),
                                    split_by_line = True
                                    )