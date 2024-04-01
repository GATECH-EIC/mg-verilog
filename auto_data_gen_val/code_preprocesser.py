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
from utils import *


def folder_create(folder_name):
    if not os.path.exists(folder_name):
        #recursively create the directory
        os.makedirs(folder_name)
    else:
        #ask the user if they want to delete the directory and create a new one
        print("The directory {} already exists. Do you want to delete it and create a new one?".format(folder_name))
        print("Type 'y' for yes and 'n' for no.")
        answer = input()
        if answer == "y":
            shutil.rmtree(folder_name)
            os.makedirs(folder_name)
        else:
            print("Leave the directory as it is.")


class CodePreprocesser:
    def __init__(self, code_dir, store_src_code_dir, 
                 csv_code_dir, csv_comment_dir, csv_new_comment_dir, 
                 csv_pure_gen_comment_dir, code_summary_dir, documented_code_dir,
                 code_suffix =[".v", ".sv", ".vh"], discard_original_comment = False):
        self.code_dir = code_dir
        self.code_suffix = code_suffix
        self.store_src_code_dir = store_src_code_dir
        self.csv_code_dir = csv_code_dir
        self.csv_comment_dir = csv_comment_dir
        self.csv_new_comment_dir = csv_new_comment_dir
        self.csv_pure_gen_comment_dir = csv_pure_gen_comment_dir
        self.code_summary_dir = code_summary_dir
        self.documented_code_dir = documented_code_dir
        self.discard_original_comment = discard_original_comment
        #check if the directory exists
        folder_create(self.store_src_code_dir)
        folder_create(self.csv_code_dir)
        folder_create(self.csv_comment_dir)
        folder_create(self.csv_new_comment_dir)
        folder_create(self.csv_pure_gen_comment_dir)
        folder_create(self.code_summary_dir)
        folder_create(self.documented_code_dir)
    
    def raw_code_copy(self, src_dir, dst_dir, skip_preprocess = False):
        #copy all the files with the suffix to the dst_dir
        self.code_files = []
        for file in os.listdir(src_dir):
            if file.endswith(tuple(self.code_suffix)):
                if not skip_preprocess:
                    shutil.copy(os.path.join(src_dir, file), dst_dir)
                self.code_files.append(file)
    
    def create_code_assets(self):
        #separate the comments and code and create corresponding csv files
        for code_file in tqdm(self.code_files, total=len(self.code_files), desc="Creating code assets"):
            src_code_file = os.path.join(self.store_src_code_dir, code_file)
            csv_code_file = os.path.join(self.csv_code_dir, code_file.split(".")[0] + ".csv")
            csv_comment_file = os.path.join(self.csv_comment_dir, code_file.split(".")[0] + ".csv")
            convert_raw_src_code_to_csv(src_code_file, csv_code_file, csv_comment_file, discard_original_comment = self.discard_original_comment)

    def pre_process_routines(self, dst_dir, discard_original_comment = True, rtl = True):
        for file in os.listdir(dst_dir):
            preprocess(os.path.join(dst_dir, file),discard_original_comment=discard_original_comment, rtl=rtl)

if __name__ == "__main__":
    code_dir = "../verilog/AccDNN/verilog"
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

    code_preprocesser = CodePreprocesser(code_dir, store_src_code_dir, 
                                         csv_code_dir, csv_comment_dir, 
                                         csv_new_comment_dir, csv_pure_gen_comment_dir, 
                                         code_summary_dir, documented_code_dir,
                                         code_suffix=code_suffix, discard_original_comment = False)
    code_preprocesser.raw_code_copy(code_dir, store_src_code_dir)
    code_preprocesser.create_code_assets()
