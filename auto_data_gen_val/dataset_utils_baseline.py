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
from datasets import load_from_disk
from datasets import Dataset
import jsonlines
from utils import extract_module_header, preprocess



def random_sample_dataset(datasetpath, sample_percent, savepath):
    dataset = load_from_disk(datasetpath)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset)*sample_percent)))
    dataset.save_to_disk(savepath)
    return dataset 

def find_fail_all_entries(result_file):
    all_entries = []
    with jsonlines.open(result_file) as reader:
        for obj in reader:
            all_entries.append(obj["task_id"])
    all_entries = set(all_entries)
    with jsonlines.open(result_file) as reader:
        for obj in reader:
            if obj["result"] == "passed":
                if obj["task_id"] in all_entries:
                    all_entries.remove(obj["task_id"])
    return all_entries

def form_new_prob_file(orig_prob_file, new_prob_file, fail_entries):
    with jsonlines.open(orig_prob_file) as reader:
        with jsonlines.open(new_prob_file, mode='w') as writer:
            for obj in reader:
                if obj["task_id"] in fail_entries:
                    writer.write(obj)
    return new_prob_file


if __name__ == "__main__":
    # datasetpath = "/home/user_name/DAC_2024/sft_dataset/vanilla_baseline"
    # sample_percent = 0.1
    # savepath = "/home/user_name/DAC_2024/sft_dataset/vanilla_baseline_sample_0_{}".format(int(sample_percent*10))
    # random_sample_dataset(datasetpath, sample_percent, savepath)
    result_file = "/home/user_name/DAC_2024/chatgpt4_auto_accel/model_eval_qlora_kevin/data/gen.merged_dataset+simple_description.jsonl_results.jsonl"
    new_prob_dir = "./special_set"
    hdlbits_hlvl = "hdlbits_description.jsonl"
    hdlbits_simple_desc = "hdlbits_description_simple_description.jsonl"
    hdlbits_detail_desc = "hdlbits_description_detail_description.jsonl"
    hdlbits_block_desc = "hdlbits_for_llm2_eval.jsonl"
    eval_file = "/home/user_name/DAC_2024/chatgpt4_auto_accel/verilog_eval/data/VerilogEval_Machine.jsonl"
    fail_entries = find_fail_all_entries(result_file)
    print(len(fail_entries))
    print(fail_entries)
    new_prob_file = form_new_prob_file(hdlbits_hlvl, os.path.join(new_prob_dir, hdlbits_hlvl), fail_entries)
    new_prob_file = form_new_prob_file(hdlbits_simple_desc, os.path.join(new_prob_dir, hdlbits_simple_desc), fail_entries)
    new_prob_file = form_new_prob_file(hdlbits_detail_desc, os.path.join(new_prob_dir, hdlbits_detail_desc), fail_entries)
    new_prob_file = form_new_prob_file(hdlbits_block_desc, os.path.join(new_prob_dir, hdlbits_block_desc), fail_entries)
    new_prob_file = form_new_prob_file(eval_file, os.path.join(new_prob_dir, "VerilogEval_Machine.jsonl"), fail_entries)
