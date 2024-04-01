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


def merge_metadata_dir(metadata_dir_1, metadata_dir_2, merged_metadata_dir, dataset_part):
    for part_num in range(dataset_part[0], dataset_part[1]):
        part_dir = os.path.join(merged_metadata_dir, "part{}".format(part_num))
        codes_json = os.path.join(part_dir, "bookkeeping", "codes.json")
        if os.path.exists(part_dir):
            shutil.rmtree(part_dir)
        os.makedirs(part_dir)
        os.makedirs(os.path.join(part_dir, "bookkeeping"))
        #copy vectorembedding from metadata_dir_2
        shutil.copytree(os.path.join(metadata_dir_2, "part{}".format(part_num), "vectorembedding"), os.path.join(part_dir, "vectorembedding"))
        #copy any other json files from metadata_dir_2 to part_dir
        for file in os.listdir(os.path.join(metadata_dir_2, "part{}".format(part_num))):
            if file.endswith(".json"):
                shutil.copy(os.path.join(metadata_dir_2, "part{}".format(part_num), file), os.path.join(part_dir, file))
        #copy any other json files from metadata_dir_1 to part_dir
        for file in os.listdir(os.path.join(metadata_dir_1, "part{}".format(part_num))):
            if file.endswith(".json"):
                shutil.copy(os.path.join(metadata_dir_1, "part{}".format(part_num), file), os.path.join(part_dir, file))
        #merge codes.json from metadata_dir_1 and metadata_dir_2,
        #if some entry of metadata_dir_2 is None, use the entry from metadata_dir_1
        codes_json_1 = json.load(open(os.path.join(metadata_dir_1, "part{}".format(part_num), "bookkeeping", "codes.json"), "r"))
        codes_json_2 = json.load(open(os.path.join(metadata_dir_2, "part{}".format(part_num), "bookkeeping", "codes.json"), "r"))
        merged_codes_json = {}
        for code in codes_json_2:
            merged_codes_json[code] = codes_json_2[code]
            for entry in codes_json_2[code]:
                if codes_json_2[code][entry] == None:
                    merged_codes_json[code][entry] = codes_json_1[code][entry]
                    continue
                if type(codes_json_2[code][entry]) == str and codes_json_2[code][entry] == "":
                    merged_codes_json[code][entry] = codes_json_1[code][entry]
                    continue
                if type(codes_json_2[code][entry]) == list and len(codes_json_2[code][entry]) == 0:
                    merged_codes_json[code][entry] = codes_json_1[code][entry]
                    continue
                if type(codes_json_2[code][entry]) == dict and len(codes_json_2[code][entry]) == 0:
                    merged_codes_json[code][entry] = codes_json_1[code][entry]
                    continue
                if type(codes_json_2[code][entry]) == list and  codes_json_2[code][entry][0] == None:
                    merged_codes_json[code][entry] = codes_json_1[code][entry]
                    continue
        with open(codes_json, "w") as f:
            json.dump(merged_codes_json, f)
        print("Merged part{} done".format(part_num))

def merge_dataset(dataset_list, merged_dataset_dir):
    if os.path.exists(merged_dataset_dir):
        shutil.rmtree(merged_dataset_dir)
    os.makedirs(merged_dataset_dir)
    new_merged_dataset_dict = {"code": [], "description": []}
    for dataset_dir in dataset_list:
        dataset = load_from_disk(dataset_dir)
        new_merged_dataset_dict["code"] += dataset["code"]
        new_merged_dataset_dict["description"] += dataset["description"]
    #randomly shuffle the dataset
    new_merged_dataset = Dataset.from_dict(new_merged_dataset_dict)
    new_merged_dataset = new_merged_dataset.shuffle()
    new_merged_dataset.save_to_disk(merged_dataset_dir)
    print("Merged dataset saved to {}".format(merged_dataset_dir))


def test_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    while True:
        #take user input for the index of the code to test
        idx = input("Enter the code index to test: ")
        if idx == "exit":
            break
        idx = int(idx)
        description = dataset[idx]["description"]
        code = dataset[idx]["code"]
        print("Description:\n {}".format(description))
        print("Code:\n {}".format(code))
        print("Code length: {}".format(len(tokenizer.encode(code))))
        print("Description length: {}".format(len(tokenizer.encode(description))))
        print("Total length: {}".format(len(tokenizer.encode(description)) + len(tokenizer.encode(code))))

class Data4AIGChipDataset:
    def __init__(self, dataset_metadata_dir: list, 
                 packaged_dir: str = "./packaged_dataset",
                 ) -> None:
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.dataset_metadata_dir = dataset_metadata_dir
        self.packaged_dir = packaged_dir
        self.bookkeeping_dir = "bookkeeping"
        self.metadata_json = "codes.json"
        self.load_code_entries()
        print("Loaded {} code entries".format(len(self.codes)))
        self.load_commented_code()
        self.load_pure_comment()
        self.load_pure_code()
        
    
    def load_code_entries(self, documented_src = "documented_code"):
        self.codes = {}
        task_id = 0
        for doc_dir in self.dataset_metadata_dir:
            individual_metadata_file = os.path.join(doc_dir, self.bookkeeping_dir, self.metadata_json)
            with open(individual_metadata_file, "r") as f:
                loaded_dict = json.load(f)
                print("Loaded {} code entries from {}".format(len(loaded_dict), individual_metadata_file))
                for code in loaded_dict:
                    self.codes[code+str(task_id)] = loaded_dict[code]
                    task_id += 1
    
    def load_commented_code(self):
        for code in self.codes:
            code_path = self.codes[code]["path"]
            with open(code_path, "r") as f:
                self.codes[code]["commented_code"] = f.read()
    
    
    def load_pure_comment(self):
        for code in self.codes:
            csv_comment = self.codes[code]["csv_comment"]
            df_comment = pd.read_csv(csv_comment)
            self.codes[code]["pure_comment"] = df_comment["content"].tolist()            

    def load_pure_code(self, code_path = "assets/verilog/code_and_comment_src/csv_src/csv_code_src"):
        for code in self.codes:
            csv_code = self.codes[code]["csv_code"]
            df_code = pd.read_csv(csv_code)
            self.codes[code]["pure_code"] = df_code["content"].tolist()

    def package_detailed_description_dataset(self):
        new_dataset_dict = {"code": [], "description": []}
        for code in self.codes:
            system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
            question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."

            assert type(self.codes[code]["global_summary_detailed"]) == str
            global_summary_str = self.codes[code]["global_summary_detailed"]

            # #handle cases where csv assets are not generated
            # if self.codes[code]["module_header"] == None:
            #     preprocess(self.codes[code]["path"])
            #     self.codes[code]["module_header"] = extract_module_header(self.codes[code]["path"])
            # if "pure_code" not in self.codes[code]:
            #     self.codes[code]["pure_code"] = open(self.codes[code]["path"], "r").readlines()

            problem_description = "\n\n" + global_summary_str + "\n\n Module header:\n\n"  + self.codes[code]["module_header"] + "\n"
            combined_prompt = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)
            assert self.codes[code]["module_header"] in " ".join(self.codes[code]["pure_code"])
            code_prompt = " ".join(self.codes[code]["pure_code"]).replace(self.codes[code]["module_header"], "")
            new_dataset_dict["code"].append(code_prompt)
            new_dataset_dict["description"].append(combined_prompt)
        new_dataset = Dataset.from_dict(new_dataset_dict)
        new_dataset.save_to_disk(os.path.join(self.packaged_dir, "detailed_description_dataset"))

    def package_simple_description_dataset(self):
        new_dataset_dict = {"code": [], "description": []}
        for code in self.codes:
            system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
            question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
            
            assert type(self.codes[code]["global_summary_high_level"]) == str
            global_summary_str = self.codes[code]["global_summary_high_level"]

            # #handle cases where csv assets are not generated
            # if self.codes[code]["module_header"] == None:
            #     preprocess(self.codes[code]["path"])
            #     self.codes[code]["module_header"] = extract_module_header(self.codes[code]["path"])
            # if "pure_code" not in self.codes[code]:
            #     self.codes[code]["pure_code"] = open(self.codes[code]["path"], "r").readlines()

            problem_description = "\n\n" + global_summary_str + "\n\n Module header:\n\n"  + self.codes[code]["module_header"] + "\n"
            combined_prompt = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)
            assert self.codes[code]["module_header"] in " ".join(self.codes[code]["pure_code"])
            code_prompt = " ".join(self.codes[code]["pure_code"]).replace(self.codes[code]["module_header"], "")
            new_dataset_dict["code"].append(code_prompt)
            new_dataset_dict["description"].append(combined_prompt)
        new_dataset = Dataset.from_dict(new_dataset_dict)
        new_dataset.save_to_disk(os.path.join(self.packaged_dir, "simple_description_dataset"))

    def package_llm1_high_level_summary_to_block_summary_dataset(self, skipping_single_block = False):
        new_dataset_dict = {"code": [], "description": []}
        for code in self.codes:
            system_prompt = "You only expand the high level summary of a Verilog code module design to block level summaries."
            question_prompt = "Come up with correct functional blocks for to implement the Verilog module in the high level summary, and expand the high level summary to block level summaries."

            assert type(self.codes[code]["global_summary_high_level"]) == str
            global_summary_str = self.codes[code]["global_summary_high_level"]

            problem_description = "\n Here is the high level summary:\n\n"  + global_summary_str + "\n\n"
            problem_description += "\n Here is the Verilog module header:\n\n"  + self.codes[code]["module_header"] + "\n\n" 
            combined_prompt = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)
            
            if skipping_single_block and len(self.codes[code]["block_summary"]) == 1:
                continue
            
            block_summary_dict = {}
            for block_idx, block in enumerate(self.codes[code]["block_summary"]):
                #better formatting
                if len(self.codes[code]["block_summary"]) == 1:
                    try:
                        block = json.loads(block)
                        block = block["usage"] + "; " + block["summary"]
                    except:
                        assert type(block) == str
                        block = block
                else:
                    split = block.split("summary:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'summary:'")
                    split = block.split("provided:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'provided:'")
                block_summary_dict["block_{}".format(block_idx)] = block
            code_prompt = "\n"
            for block in block_summary_dict:
                code_prompt += "{}: {}\n".format(block, block_summary_dict[block])
            new_dataset_dict["code"].append(code_prompt)
            new_dataset_dict["description"].append(combined_prompt)
        new_dataset = Dataset.from_dict(new_dataset_dict)
        new_dataset.save_to_disk(os.path.join(self.packaged_dir, "llm1_high_level_summary_to_block_summary_dataset"))
    
    def package_llm2_block_summary_to_pure_code_one_shot_dataset(self, skipping_single_block = False):
        new_dataset_dict = {"code": [], "description": []}
        for code in self.codes:
            block_summary_dict = {}

            if skipping_single_block and len(self.codes[code]["block_summary"]) == 1:
                continue

            for block_idx, block in enumerate(self.codes[code]["block_summary"]):
                #better formatting
                if len(self.codes[code]["block_summary"]) == 1:
                    try:
                        block = json.loads(block)
                        block = block["usage"] + "; " + block["summary"]
                    except:
                        assert type(block) == str
                        block = block
                else:
                    split = block.split("summary:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'summary:'")
                    split = block.split("provided:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'provided:'")
                block_summary_dict["block_{}".format(block_idx)] = block
            block_summary_str = ""
            for block in block_summary_dict:
                block_summary_str += "{}: {}\n".format(block, block_summary_dict[block])
            system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
            question_prompt = "Implement the Verilog module based on the following block level summaries. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
            problem_description = "\nHere are block level summaries:\n\n" + block_summary_str + "\n\n Module header:\n\n"  + self.codes[code]["module_header"] + "\n"
            combined_prompt = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)
            assert self.codes[code]["module_header"] in " ".join(self.codes[code]["pure_code"])
            code_prompt = " ".join(self.codes[code]["pure_code"]).replace(self.codes[code]["module_header"], "")
            new_dataset_dict["code"].append(code_prompt)
            new_dataset_dict["description"].append(combined_prompt)
        
        new_dataset = Dataset.from_dict(new_dataset_dict)
        new_dataset.save_to_disk(os.path.join(self.packaged_dir, "llm2_block_summary_to_pure_code_one_shot_dataset"))

    def package_hdlbits_for_llm2_eval(self, metadata_file):
        #load block summary file 
        metadata = json.load(open(metadata_file, "r"))
        task_entry_list =[]
        for code in metadata:
            task_entry = {}
            task_id = code.split(".v")[0]
            task_entry["task_id"] = task_id
            block_summary = metadata[code]["block_summary"]
            block_summary_dict = {}
            for block_idx, block in enumerate(block_summary):
                #better formatting
                if len(block_summary) == 1:
                    assert type(block) == str
                    block = block
                else:
                    split = block.split("summary:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'summary:'")
                    split = block.split("provided:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'provided:'")
                block_summary_dict["block_{}".format(block_idx)] = block

            block_summary_str = ""
            for block in block_summary_dict:
                block_summary_str += "{}: {}\n".format(block, block_summary_dict[block])


            assert type(metadata[code]["global_summary_high_level"]) == str
            global_summary_str = metadata[code]["global_summary_high_level"]

            task_entry["block_to_code_description"] = block_summary_str
            if len(block_summary_dict) > 1:
                task_entry["block_global_to_code_description"] = "\nHere are global summaries:\n\n"+ global_summary_str +"\n\n Here are block level summaries:\n\n" + block_summary_str
            else:
                task_entry["block_global_to_code_description"] = "\nHere are global summaries:\n\n" + global_summary_str

            task_entry_list.append(task_entry)
        desc_file = os.path.join(self.packaged_dir, "hdlbits_for_llm2_eval.jsonl")
        with jsonlines.open(desc_file, "w") as f:
            f.write_all(task_entry_list)

    def package_hdlbits_description_file(self, metadata_file, desc_key = "detail_description"):
        metadata_file = json.load(open(metadata_file, "r"))
        metadata_list = []
        for code in metadata_file:
            task_entry = {}
            task_id = code.split(".v")[0]
            task_entry["task_id"] = task_id
            if desc_key == "detail_description":
                global_summary_str = metadata_file[code]["global_summary_detailed"]
            elif desc_key == "simple_description":
                global_summary_str = metadata_file[code]["global_summary_high_level"]
            task_entry[desc_key] = global_summary_str
            metadata_list.append(task_entry)
        desc_file = os.path.join(self.packaged_dir, "hdlbits_description_{}.jsonl".format(desc_key))
        with jsonlines.open(desc_file, "w") as f:
            f.write_all(metadata_list)
                    

    def package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset(self, skipping_single_block = False):
        new_dataset_dict = {"code": [], "description": []}
        for code in self.codes:
            block_summary_dict = {}

            if skipping_single_block and len(self.codes[code]["block_summary"]) == 1:
                continue

            for block_idx, block in enumerate(self.codes[code]["block_summary"]):
                #better formatting
                if len(self.codes[code]["block_summary"]) == 1:
                    try:
                        block = json.loads(block)
                        block = block["usage"] + "; " + block["summary"]
                    except:
                        assert type(block) == str
                        block = block
                else:
                    split = block.split("summary:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'summary:'")
                    split = block.split("provided:")
                    if len(split) == 2:
                        block = split[1].strip()
                    elif len(split) == 1:
                        block = split[0].strip()
                    else:
                        raise("Error: block summary has more than one 'provided:'")
                block_summary_dict["block_{}".format(block_idx)] = block
            block_summary_str = ""
            for block in block_summary_dict:
                block_summary_str += "{}: {}\n".format(block, block_summary_dict[block])
            system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
            question_prompt = "Implement the Verilog module based on the following block level summaries. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
            if len(self.codes[code]["block_summary"]) > 1:
                assert type(self.codes[code]["global_summary_high_level"]) == str
                global_summary_str = self.codes[code]["global_summary_high_level"]
                problem_description = "\nHere are global summaries:\n\n"+ global_summary_str +"\n\n Here are block level summaries:\n\n" + block_summary_str + "\n\n Module header:\n\n"  + self.codes[code]["module_header"] + "\n"
            else:
                problem_description = "\nHere are block level summaries:\n\n" + block_summary_str + "\n\n Module header:\n\n"  + self.codes[code]["module_header"] + "\n"
            combined_prompt = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)
            assert self.codes[code]["module_header"] in " ".join(self.codes[code]["pure_code"])
            code_prompt = " ".join(self.codes[code]["pure_code"]).replace(self.codes[code]["module_header"], "")
            new_dataset_dict["code"].append(code_prompt)
            new_dataset_dict["description"].append(combined_prompt)

        new_dataset = Dataset.from_dict(new_dataset_dict)
        new_dataset.save_to_disk(os.path.join(self.packaged_dir, "llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset"))



    def token_length_profiler(self):
        for code in self.codes:
            self.codes[code]["token_length"] = {}
            self.codes[code]["token_length"]["commented_code"] = len(self.tokenizer.encode(self.codes[code]["commented_code"]))
            self.codes[code]["token_length"]["global_summary_detailed"] = len(self.tokenizer.encode(self.codes[code]["global_summary_detailed"]))
            self.codes[code]["token_length"]["block_summary"] = 0
            for block in self.codes[code]["block_summary"]:
                self.codes[code]["token_length"]["block_summary"] += len(self.tokenizer.encode(block))
            self.codes[code]["token_length"]["pure_comment"] = 0
            for comment in self.codes[code]["pure_comment"]:
                self.codes[code]["token_length"]["pure_comment"] += len(self.tokenizer.encode(comment))
            self.codes[code]["token_length"]["pure_code"] = 0
            for pure_code in self.codes[code]["pure_code"]:
                self.codes[code]["token_length"]["pure_code"] += len(self.tokenizer.encode(pure_code))

            #llm1 total token length global summary --> block summary
            self.codes[code]["token_length"]["llm1_total"] = self.codes[code]["token_length"]["global_summary_high_level"] + self.codes[code]["token_length"]["block_summary"]
            #llm2 total token length block summary --> commented code
            self.codes[code]["token_length"]["llm2_total"] = self.codes[code]["token_length"]["block_summary"] + self.codes[code]["token_length"]["commented_code"]
            #llm2-block_summary + global_summary --> commented code
            self.codes[code]["token_length"]["llm2_block_global_input"] = self.codes[code]["token_length"]["block_summary"] + self.codes[code]["token_length"]["global_summary_high_level"]
            #llm3 total token length block summary  --> pure comments
            self.codes[code]["token_length"]["llm3_total"] = self.codes[code]["token_length"]["block_summary"] + self.codes[code]["token_length"]["pure_comment"]
            #llm4 total token length pure comments--> pure code
            self.codes[code]["token_length"]["llm4_total"] = self.codes[code]["token_length"]["pure_comment"] + self.codes[code]["token_length"]["pure_code"]
            
            
            
    
    def plot_token_length(self):
        import matplotlib.pyplot as plt
        self.token_length_profiler()
        
        #for each token length entry, plot a separate bar chart distribution
        #x-axis: code idx
        #y-axis: token length
        
        #commented code 
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["commented_code"])
        plt.bar(x,y)
        plt.title("commented code token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("commented_code_token_length.png")
        plt.clf()

        #global summary
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["global_summary_detailed"])
        plt.bar(x,y)
        plt.title("global summary token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("global_summary_token_length.png")
        plt.clf()

        #block summary
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["block_summary"])
        plt.bar(x,y)
        plt.title("block summary token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("block_summary_token_length.png")
        

        #llm1 total
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["llm1_total"])
        plt.bar(x,y)
        plt.title("llm1 total token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("llm1_total_token_length.png")
        plt.clf()

        #llm2 total
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["llm2_total"])
        plt.bar(x,y)
        plt.title("llm2 total token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("llm2_total_token_length.png")
        plt.clf()

        #llm2_block_global_input
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["llm2_block_global_input"])
        plt.bar(x,y)
        plt.title("llm2_block_global_input token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("llm2_block_global_input_token_length.png")
        plt.clf()

        #llm3 total
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["llm3_total"])
        plt.bar(x,y)
        plt.title("llm3 total token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("llm3_total_token_length.png")
        plt.clf()

        #llm4 total
        x = []
        y = []
        for code in self.codes:
            x.append(code)
            y.append(self.codes[code]["token_length"]["llm4_total"])
        plt.bar(x,y)
        plt.title("llm4 total token length")
        #hide x ticks
        plt.xticks([])
        #save figure
        plt.savefig("llm4_total_token_length.png")
        plt.clf()

        



            


            
            
            


        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doced_dataset_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/")
    parser.add_argument("--total_part", type=int, default=10)
    parser.add_argument("--packaged_dir", type=str, default="./packaged_dataset")
    parser.add_argument("--package_detailed_description", action="store_true")
    parser.add_argument("--package_simple_description", action="store_true")
    parser.add_argument("--package_llm2_block_summary_to_pure_code_one_shot_dataset", action="store_true")
    parser.add_argument("--package_llm1_high_level_summary_to_block_summary_dataset", action="store_true")
    parser.add_argument("--package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset", action="store_true")
    parser.add_argument("--documented_benchmark_dataset", type=str, default="/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part10/")
    parser.add_argument("--package_hdlbits_global_summary_description_file", action="store_true")
    parser.add_argument("--package_hdlbits_block_summary_description_file", action="store_true")
    parser.add_argument("--package_merged_dataset", action="store_true")

    args = parser.parse_args()
    doced_dataset_dir = args.doced_dataset_dir
    total_part = args.total_part
    packaged_dir = args.packaged_dir
    package_detailed_description = args.package_detailed_description
    package_simple_description = args.package_simple_description
    package_llm2_block_summary_to_pure_code_one_shot_dataset = args.package_llm2_block_summary_to_pure_code_one_shot_dataset
    package_llm1_high_level_summary_to_block_summary_dataset = args.package_llm1_high_level_summary_to_block_summary_dataset
    package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset = args.package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset
    package_hdlbits_global_summary_description_file = args.package_hdlbits_global_summary_description_file
    package_hdlbits_block_summary_description_file = args.package_hdlbits_block_summary_description_file
    package_merged_dataset = args.package_merged_dataset

    documented_benchmark_dataset = args.documented_benchmark_dataset

    # dataset_dirs = ["/home/user_name/DAC_2024/sft_dataset/llm2_new_block_summary_plus_new_verilogeval_global_summary_to_pure_code_skip_single",
    #                 "/home/user_name/DAC_2024/sft_dataset/llm2_new_block_summary_to_pure_code",
    #                 "/home/user_name/DAC_2024/sft_dataset/new_baseline_verilogeval_global_summary"
    #                 "/home/user_name/DAC_2024/sft_dataset/simple_description_dataset",
    #                 ]
    # merge_dataset(dataset_dirs, "merged_dataset")
    # test_dataset("merged_dataset")
    # exit()
    
    # merge_metadata_dir("/home/user_name/DAC_2024/ckpt3_and_ckpts/11_13_23_18_31_with_block_summary_and_semi_global_summary/dataset_metadata",
    #                     "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_17_new_blk_summary/dataset_metadata",
    #                     "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_17_new_blk_summary/dataset_metadata_merged"
    #                     ,[0,10])
    
    # merge_metadata_dir("/home/user_name/DAC_2024/ckpt3_and_ckpts/11_14_23_11_39_with_hdlbits_separately_doced_in_part_10/dataset_metadata/",
    #                     "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_17_new_blk_summary/dataset_metadata",
    #                     "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_17_new_blk_summary/dataset_metadata_merged"
    #                     ,[10,11])


    # dataset_metadata_dir = [   "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part0",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part1",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part2",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part3",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part4",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part5",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part6",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part7",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part8",
    #                            "/home/user_name/DAC_2024/ckpt3_and_ckpts/11_18_simple_descriptions/dataset_metadata/part9"
    #                         ]

    dataset_metadata_dir = []
    metadata_dir = "./dataset_metadata"
    for part_num in range(total_part):
        dataset_metadata_dir.append(os.path.join(doced_dataset_dir, metadata_dir, "part{}".format(part_num)))
    
    dataset = Data4AIGChipDataset(dataset_metadata_dir, packaged_dir)
    if package_detailed_description:
        dataset.package_detailed_description_dataset()
    if package_simple_description:
        dataset.package_simple_description_dataset()
    if package_llm2_block_summary_to_pure_code_one_shot_dataset:
        dataset.package_llm2_block_summary_to_pure_code_one_shot_dataset(skipping_single_block=False)
    if package_llm1_high_level_summary_to_block_summary_dataset:
        dataset.package_llm1_high_level_summary_to_block_summary_dataset(skipping_single_block=False)
    if package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset:
        dataset.package_llm2_block_summary_plus_global_summary_to_pure_code_one_shot_dataset(skipping_single_block=True)
    
    if package_hdlbits_global_summary_description_file:
        benchmark_metadata = os.path.join(documented_benchmark_dataset, "bookkeeping", "codes.json")
        dataset.package_hdlbits_description_file(benchmark_metadata, desc_key = "detail_description")
        dataset.package_hdlbits_description_file(benchmark_metadata, desc_key = "simple_description")
    
    if package_hdlbits_block_summary_description_file:
        benchmark_metadata = os.path.join(documented_benchmark_dataset, "bookkeeping", "codes.json")
        dataset.package_hdlbits_for_llm2_eval(benchmark_metadata)

    if package_merged_dataset:
        dataset_dirs = [
                        "./llm2_block_summary_to_pure_code_one_shot_dataset",
                        "./detailed_description_dataset",
                        "./simple_description_dataset"
                        ]
        merge_dataset(dataset_dirs, os.path.join(packaged_dir, "merged_dataset"))
    # test_dataset("detailed_description_dataset")
    # dataset.plot_token_length()