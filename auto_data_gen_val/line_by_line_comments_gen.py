import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
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
    import argparse

    parser = argparse.ArgumentParser(description='Line-by-line Code Documentor')
    parser.add_argument('--total_parts', type=int, default=10, help='total parts')
    parser.add_argument('--output_dir', type=str, default="./documented_code", help='output directory')
    parser.add_argument('--src_code_dir', type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed/", help='code directory')
    parser.add_argument('--code_metadata_dir', type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/", help='code metadata file')
    parser.add_argument('--code_lib_path', type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib/", help='code library path')
    parser.add_argument('--code_vec_store', type=str, default="../code_vec_store/test_10_30/", help='code vector store')
    parser.add_argument('--skip_preprocess', action='store_true', help='skip preprocessing')
    parser.add_argument('--skip_supplement_summary', action='store_true', help='skip supplementing summary')
    parser.add_argument('--discard_original_comment', action='store_true', help='discard original comment')

    args = parser.parse_args()
    total_parts = args.total_parts
    output_dir = args.output_dir
    src_code_dir = args.src_code_dir
    code_metadata_dir = args.code_metadata_dir
    code_lib_path = args.code_lib_path
    code_vec_store = args.code_vec_store
    skip_preprocess = args.skip_preprocess
    skip_supplement_summary = args.skip_supplement_summary
    discard_original_comment = args.discard_original_comment

    for code_part in range(total_parts):
        code_dir = os.path.join(src_code_dir, "part{}".format(code_part))
        code_metadata_file = os.path.join(code_metadata_dir, "part{}".format(code_part), "codes.json")
        # code_lib_path =  "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib/"
        # code_vec_store = "../code_vec_store/test_10_30/"

        
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
            # discard_original_comment = True
            # skip_preprocess = True
            # skip_supplement_summary = True

            code_repo_documentor = CodeRepoDocumentor(code_dir, store_src_code_dir,
                                                        csv_code_dir, csv_comment_dir, csv_new_comment_dir, 
                                                        csv_pure_gen_comment_dir, code_summary_dir, documented_code_dir,
                                                        code_metadata_file=code_metadata_file,
                                                        code_suffix=code_suffix, language=language,
                                                        discard_original_comment=discard_original_comment,
                                                        code_lib_path=code_lib_path, code_vec_store=code_vec_store,
                                                        skip_supplement_summary=skip_supplement_summary,
                                                        cb = cb)
            code_repo_documentor.create_embedding()
            code_repo_documentor.code_preprocess(skip_preprocess=skip_preprocess)
            code_repo_documentor.document_repo()

            output_dir_part = os.path.join(output_dir, "part{}".format(code_part))
            #check if output dir exists
            if not os.path.exists(output_dir_part):
                os.makedirs(output_dir_part)
            else:
                #ask for confirmation
                print("Output directory already exists. Do you want to overwrite? (y/n)")
                choice = input().lower()
                if choice == "y":
                    shutil.rmtree(output_dir_part)
                    os.makedirs(output_dir_part)
                else:
                    print("Exiting...")
                    continue
            code_repo_documentor.package_documented_code(output_dir_part)
            #copy assets to output dir
            shutil.copytree(os.environ.get("ASSET_DIR"), os.path.join(os.path.join(code_metadata_dir, "part{}".format(code_part)), "assets"))
            #copy vector store to output dir
            shutil.copytree(code_vec_store, os.path.join(os.path.join(code_metadata_dir, "part{}".format(code_part)), "code_vec_store"))
            

