import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))
import openai
import requests
import json
import copy
import time
import datetime
import shutil
from embedding_lookup_utils import *
from utils import *
from completion_handler import *
from code_preprocesser import *
from code_repo_documentor import *

#documenting the first version with module instantiation
#one_shot 5 lines
#pure llama 2 70B
#around 12k samples

if __name__ == "__main__":
    #NOTE: run utils.py first to partition the code first
    code_part = 0
    code_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed/part{}".format(code_part)
    code_metadata_file = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/part{}/codes.json".format(code_part)
    code_lib_path =  "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib/"
    code_vec_store = "../code_vec_store/test_10_30/"
    language = os.environ.get("TARGET_LANG")
    if os.environ.get("TARGET_LANG") == "verilog":
        code_suffix = [".v", ".sv", ".vh"]
    elif os.environ.get("TARGET_LANG") == "xilinx_hls":
        code_suffix = [".c", ".cpp", ".h", ".hpp"]
    store_src_code_dir = os.environ.get("STORE_SRC_CODE_DIR")
    csv_code_dir = os.environ.get("CSV_CODE_DIR")
    csv_comment_dir = os.environ.get("CSV_COMMENT_DIR")
    csv_new_comment_dir = os.environ.get("CSV_NEW_COMMENT_DIR")
    csv_pure_gen_comment_dir = os.environ.get("CSV_PURE_GEN_COMMENT_DIR")
    code_summary_dir = os.environ.get("CODE_SUMMARY_DIR")
    documented_code_dir = os.environ.get("DOCUMENTED_CODE_DIR")


    with get_openai_callback() as cb:
        #This switch will discard 1. the comments in the raw code copy and 2. the comments will be converted to the raw code csv 
        discard_original_comment = True
        
        code_repo_documentor = CodeRepoDocumentor(code_dir, store_src_code_dir,
                                                    csv_code_dir, csv_comment_dir, csv_new_comment_dir, 
                                                    csv_pure_gen_comment_dir, code_summary_dir, documented_code_dir,
                                                    code_metadata_file=code_metadata_file,
                                                    code_suffix=code_suffix, language=language,
                                                    discard_original_comment=discard_original_comment,
                                                    code_lib_path=code_lib_path, code_vec_store=code_vec_store,
                                                    skip_rag_db=True,
                                                    cb = cb)
        code_repo_documentor.create_embedding()
        code_repo_documentor.code_preprocess()
        code_repo_documentor.package_documented_code("./documented_code")
