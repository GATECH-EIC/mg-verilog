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
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
from ast import literal_eval
import numpy as np
import re 
from tqdm import tqdm
import tiktoken
from datasets import load_from_disk
from datasets import Dataset
from utils import *

llama2_prompt_with_memory ="""
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {chat_history} {human_input} [/INST]
"""

llama2_prompt_without_memory ="""
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {human_input} [/INST]
"""

llama2_prompt_without_memory_without_sys ="""
<s>[INST] {human_input} [/INST]
"""

llama2_pompt_with_memory_without_sys ="""
<s>[INST] {chat_history} {human_input} [/INST]
"""

llama2_memory_prompt ="""{human_input} [/INST] {model_reply}</s><s>[INST]"""


def eval_file_to_part_data(eval_file, data_dir, meta_data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    if os.path.exists(meta_data_dir):
        shutil.rmtree(meta_data_dir)
    os.makedirs(meta_data_dir)
    code_pieces = []
    with open(eval_file, "r") as f:
        for line in f:
            code_pieces.append(json.loads(line))
    meta_data = {}
    for code_idx, code in tqdm(enumerate(code_pieces), total=len(code_pieces), desc="Preparing data"):
        code_name = code["task_id"] 
        module_header = code["prompt"].replace("top_module", code_name)
        code_content = module_header+code["canonical_solution"]

        #preprocess code
        output_str_list, module_name_list = part_verilog_module_string(code_content) 
        assert len(module_name_list) == 1
        assert code_name == module_name_list[0]
        code_content = output_str_list[0]

        #enter dummy meta data
        meta_data[code_name] = {"code_name": code_name, "module_inst_list": []}

        #save code to file 
        code_file = os.path.join(data_dir, code_name+".v")
        with open(code_file, "w") as f:
            f.write(code_content)
    #save meta data to file
    meta_data_file = os.path.join(meta_data_dir, "codes.json")
    with open(meta_data_file, "w") as f:
        json.dump(meta_data, f, indent=4)

    return data_dir




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="/home/user_name/DAC_2024/chatgpt4_auto_accel/verilog_eval/data/VerilogEval_Machine.jsonl")
    parser.add_argument("--data_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed/part10")
    parser.add_argument("--meta_data_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/part10")
    args = parser.parse_args()
    eval_file = args.eval_file
    data_dir = args.data_dir
    meta_data_dir = args.meta_data_dir
    eval_file_to_part_data(eval_file, data_dir, meta_data_dir)
                        