import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))

sys.path.append("../verilog_eval/verilog_eval")
from evaluation import evaluate_functional_correctness

import requests
import json
import uuid
from io import StringIO  
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

from chain_utils import SimpleConverseChain
from pyverilog.vparser.parser import parse
from datasets import load_dataset, load_from_disk, Dataset


def compile_syntax_check(code_str):
    row = code_str
    file_id = str(uuid.uuid4()) 
    path = "tmp/file{}.v".format(file_id)
    asset_dir = "tmp/asset{}".format(file_id)
    
    #check if tmp dir exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp", exist_ok=True)

    #check and make asset dir
    if not os.path.exists(asset_dir):
        os.makedirs(os.path.dirname(asset_dir), exist_ok=True)

    with open(path, "w") as f:
        f.write(row)

    try:
        ast, directives = parse([path], debug=False, outputdir=asset_dir, preprocess_output="tmp/preprocess.output.{}".format(file_id))
        output = StringIO()
        ast.show(buf=output)
        for lineno, directive in directives:
            output.write('Line %d : %s' % (lineno, directive))
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        return True
    except Exception as e:
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        return False

def reverse_codegen(description, code_str, model="gpt-4-0613", max_trials=10):
    system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
    question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
    problem_description = "\n\n {description} \n\n Module header:\n\n {module_header}\n"
    #retrieve the module header
    module_header = extract_module_header(code_str, code_str=True)
    #generate the prompt
    user_prompt = question_prompt + problem_description.format(description=description, module_header=module_header)
    chain = SimpleConverseChain(system_prompt=system_prompt, model=model, temperature=0.7, max_tokens=512, top_p=0.95, have_memory=False, verbose=False)
    for trial in range(max_trials):
        completion = chain.chat(user_prompt, system_prompt=system_prompt)
        #check if the completion is valid
        if compile_syntax_check(completion):
            return True, completion
    return False, completion
    
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/user_name/DAC_2024/sft_dataset/detail_description_dataset")
    parser.add_argument("--output_dir", type=str, default="/home/user_name/DAC_2024/sft_dataset/detail_description_dataset_val")
    dataset_dir = parser.parse_args().dataset_dir
    output_dir = parser.parse_args().output_dir

    #load the dataset
    generated_dataset = load_from_disk(dataset_dir)

    new_dataset = {"code": [], "description": []}
    for i in range(len(generated_dataset)):
        code_str = generated_dataset[i]["code"]
        passed, completion = reverse_codegen(generated_dataset[i]["description"], code_str)
        if passed:
            new_dataset["code"].append(code_str)
            new_dataset["description"].append(generated_dataset[i]["description"])
    new_dataset = Dataset.from_dict(new_dataset)
    new_dataset.save_to_disk(output_dir)
        
    
        



