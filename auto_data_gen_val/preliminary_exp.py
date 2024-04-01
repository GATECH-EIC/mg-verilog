import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))

sys.path.append("../verilog_eval/verilog_eval")
from evaluation import evaluate_functional_correctness

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
from utils import *
from tqdm import tqdm
import jsonlines


from chain_utils import gen_block_summary_chain, func_name_lookup_chain, VerilogEval, detail_steps_chain, openai_chat

from embedding_lookup_utils import openai_chat, validate_global_summary_openai

def reverse_code_gen_openai(desc_file, eval_file, result_file, repeat=10):
    desc_list = []
    with jsonlines.open(desc_file) as reader:
        for obj in reader:
            desc_list.append(obj)
    results = []
    for obj in desc_list:
        for r in range(repeat):
            print("task_id: {}".format(obj["task_id"]))
            task_id = obj["task_id"]
            # print("description: {}".format(desc_dict[task_id]))
            passed, code = validate_global_summary_openai(obj["detail_description"], task_id, eval_file, max_trials=1)
            results.append({"task_id":task_id, "completion": code, "passed":passed})
            print("passed: {}".format(passed))
    with jsonlines.open(result_file, "w") as writer:
        writer.write_all(results)
        

            
 



if __name__ == "__main__":
    verilogeval0 = VerilogEval(model="llama2")
    example_cstr_json = "/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/verilogeval_datagen/example_code_strings.json"
    with open(example_cstr_json, "r") as f:
        example_code_strings = json.load(f)
    example_code_description_file = "/home/user_name/DAC_2024/verilogeval/verilog-eval/descriptions/VerilogDescription_Machine.jsonl"
    eval_file = "/home/user_name/DAC_2024/verilogeval/verilog-eval/data/VerilogEval_Machine.jsonl"
    global_summary_chain = verilogeval0.verilog_eval_sft_data
    code_gen_chain = verilogeval0.code_gen

    tested_model = "llama2-70b"
    generated_description_file = "gen_{}.jsonl".format(tested_model)
    reverse_code_gen_openai_file = "reverse_code_gen_{}-openai-gpt4.jsonl".format(tested_model)
    reverse_code_gen_file = "reverse_code_gen_{}.jsonl".format(tested_model)
    repeat_times = 10 

    gen_description = False
    openai_code_gen = True


    reverse_code_gen = False

    if gen_description:
        #read the code_content from eval_file
        code_content = {}
        with jsonlines.open(eval_file) as reader:
            for obj in reader:
                code_content[obj["task_id"]] = obj["prompt"] + "\n"+obj["canonical_solution"]
        generated_description = []
        for task_id in code_content:
            print("generating description for task_id: {}".format(task_id))
            for i in range(repeat_times):
                print(i, end = " ")
                description = global_summary_chain(   code_content[task_id],
                                        example_code_description_file=example_code_description_file,
                                        example_code_strings=example_code_strings,
                                        desc_key="detail_description")
                #append task_id and description to generated_description
                generated_description.append({"task_id":task_id, "detail_description":description})
            print(len(generated_description))
            print()
        #store in a jsonl file
        with jsonlines.open(generated_description_file, "w") as writer:
            writer.write_all(generated_description)

    if openai_code_gen:
        generated_description_file = "/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/hdlbits_description.jsonl"
        #reverse code generation
        reverse_code_gen_openai(generated_description_file, eval_file, reverse_code_gen_openai_file)

    if reverse_code_gen:
        #code gen from tested model
        code_gen_chain(example_code_description_file, eval_file=eval_file, result_file=reverse_code_gen_file, repeat=repeat_times)



    
