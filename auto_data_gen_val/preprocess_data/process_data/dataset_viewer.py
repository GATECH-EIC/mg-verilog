import sys
import os
sys.path.append(os.path.abspath("../../"))
from datasets import load_dataset, load_from_disk, Dataset
import uuid

# import sys
# sys.path.append("../finetuning/")
# from llama import Tokenizer
import tiktoken

from pyverilog.vparser.parser import parse
import pyverilog.vparser.ast as vast
from minhash import deduplicate_dataset

import os
import subprocess
import json
from io import StringIO  
from utils import *
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    dataset_path = "ckpt_separated_modules"
    dataset = load_from_disk(dataset_path)
    #input from user
    while True:
        index = input("Enter index of the module to view: ")
        if index == "exit":
            break
        index = int(index)
        print(dataset[index])
        print(dataset[index]["module_name"])
        print(dataset[index]["text"])
        print(dataset[index]["task_id"])
        print(dataset[index]["code_str_before_preprocessing"])

        #save code_str_before_preprocessing to a file
        with open("test.v", "w") as f:
            f.write(dataset[index]["code_str_before_preprocessing"])