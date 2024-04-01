import sys
import os
sys.path.append(os.path.abspath("../../"))
from datasets import load_dataset, load_from_disk, Dataset
import uuid
import shutil

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

#!NOTE: the Pyverilog has been customized to handle multi-processing: https://github.com/user_name-avionics/Pyverilog 


def store_dataset_entries_to_file(dataset, output_dir, output_suffix=".v"):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    #calculate the estimated file size in GB
    print("Calculating the estimated file size in GB ...")
    total_size = 0
    for data_id, data in tqdm(enumerate(dataset), total=dataset.shape[0]):
        module_str = data["text"]
        total_size += len(module_str)
    total_size = total_size / 1024 / 1024 / 1024
    print("Total size: " + str(total_size) + " GB")
    print("Storing the dataset ...")
    #store the dataset
    for data_id, data in tqdm(enumerate(dataset), total=dataset.shape[0]):
        module_str = data["text"]
        with open(os.path.join(output_dir, str(data["task_id"])+output_suffix), "w") as f:
            f.write(module_str)
    return None


def separate_modules_in_dataset(raw_dataset_name, raw_dataset_dir, output_dir="./separated_modules"):
    raw_dataset = load_dataset(raw_dataset_name, cache_dir=raw_dataset_dir)
    raw_dataset = raw_dataset["train"]
    module_num = []
    print(raw_dataset.shape)
    new_dataset_dict = {"text": [], "module_name": [], "task_id": [], "code_str_before_preprocessing": []}
    task_count = 0
    for data_id, data in tqdm(enumerate(raw_dataset), total=raw_dataset.shape[0]):
        module_str = data["text"]
        module_str_list, module_name_list = part_verilog_module_string(module_str)
        module_num.append(len(module_str_list))
        for idx, module_str in enumerate(module_str_list):
            new_dataset_dict["text"].append(module_str)
            new_dataset_dict["module_name"].append(module_name_list[idx])
            new_dataset_dict["task_id"].append(task_count)
            new_dataset_dict["code_str_before_preprocessing"].append(data["text"])
            task_count += 1
    new_dataset = Dataset.from_dict(new_dataset_dict)
    new_dataset.save_to_disk(output_dir)
    print('Original dataset shape: ' + str(raw_dataset.shape))
    print('New dataset shape: ' + str(new_dataset.shape))
    return new_dataset, module_num


def return_filtered_mask(dataset, filter_func):
    mask = []
    for data_id, data in tqdm(enumerate(dataset), total=dataset.shape[0]):
        mask.append(filter_func(data))
    return mask


def remove_comments(row):
    row_text = row["text"]
    lines = row_text.splitlines()
    #track or original comments
    comment_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip().startswith("//"):
            comment_lines.append(line_indx)
    
    comment_start = False
    for line_indx, line in enumerate(lines):
        if line.strip().startswith("/*"):
            comment_start = True
            comment_lines.append(line_indx)
            if line.strip().endswith("*/"):
                comment_start = False
        elif line.strip().endswith("*/"):
            comment_start = False
            comment_lines.append(line_indx)
        elif comment_start:
            comment_lines.append(line_indx)
    
    #remove the comments
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line_indx not in comment_lines:
            new_lines.append(line)
    row["text"] = "\n".join(new_lines)
    return row


def remove_premod_postmod_lines(row):
    row_text = row["text"]
    lines = row_text.splitlines()
    module_start = False
    module_end = False
    module_lines = []
    for line_id, line in enumerate(lines):
        if line.strip().startswith("module"):
            module_start = True
        elif line.strip().startswith("endmodule") or line.strip().endswith("endmodule"):
            module_end = True
        if module_start:
            #currently remove `include lines
            if not line.strip().startswith("`include"):
                module_lines.append(line_id)
        if module_start and module_end:
            break
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line_indx in module_lines:
            new_lines.append(line)
    row["text"] = "\n".join(new_lines)
    return row


# remove rows without "module" and "endmodule" keywords
def valid_content(row):

    tmp_module_inst_dir = "tmp_module_inst_valid_content/"
    module_inst_json = "module_inst{}.json".format(row["task_id"])
    module_inst_dict = {row["task_id"]: {"code_name": row["module_name"]}}
    with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
        json.dump(module_inst_dict, f, indent=4)

    row = row["text"]

    has_module = False
    has_endmodule = False
    lines = 0
    prelines = 0

    in_comment = False

    for line in row.splitlines():
        line = line.strip()

        if not in_comment and line.startswith("/*"):
            in_comment = True
        elif in_comment and line.endswith("*/"):
            in_comment = False
            continue

        if in_comment or line.startswith("//") or len(line) == 0: 
            continue

        if line.startswith("`") or line.startswith("(*"):
            prelines += 1
            continue
        else:
            lines += 1
        
        # must start with module
        if lines >= 2 and not has_module:
            return False
        
        # must end with endmodule and also self-contained, secondarily
        if has_endmodule:
            return False

        # must be self-contained
        if line.startswith("module"):
            # double modules
            if has_module:
                return False
            has_module = True
        if line.startswith("endmodule") or line.endswith("endmodule"):
            # shouldn't happen theoretically, but syntax error regardless
            if has_endmodule:
                return False
            has_endmodule = True

    # has_module = "module" in row and "endmodule" in row
    valid_module = has_module and has_endmodule

    has_keyword = "always" in row or "assign" in row or "always_ff" in row or "always_comb" in row or "always_latch" in row

    return valid_module and has_keyword and lines <= 200


# remove rows with more than 1024 tokens
def valid_len(row):
    # lines = len(row["text"].splitlines())
    # tokens = len(tokenizer.encode(row["text"], bos=True, eos=False))
    # tokens = len(row["text"]) / 4
    # tokenizer = Tokenizer(model_path="../finetuning/llama/tokenizer.model")

    tmp_module_inst_dir = "tmp_module_inst_valid_len/"
    module_inst_json = "module_inst{}.json".format(row["task_id"])
    module_inst_dict = {row["task_id"]: {"code_name": row["module_name"]}}
    with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
        json.dump(module_inst_dict, f, indent=4)

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = len(tokenizer.encode(row["text"]))
    return tokens <= 1024


def valid_syntax_with_module_inst(row):
    raw_row = row
    row = raw_row["text"]

    file_id = str(uuid.uuid4()) 
    path = "tmp/file{}.v".format(file_id)
    asset_dir = "tmp/asset{}".format(file_id)
    #check and make asset dir
    if not os.path.exists(asset_dir):
        os.makedirs(os.path.dirname(asset_dir), exist_ok=True)

    tmp_module_inst_dir = "tmp_module_inst_syntax/"
    module_inst_json = "module_inst{}.json".format(raw_row["task_id"])
    
    with open(path, "w") as f:
        f.write(row)

    try:
        ast, directives = parse([path], debug=False, outputdir=asset_dir, preprocess_output="tmp/preprocess.output.{}".format(file_id))
        output = StringIO()
        ast.show(buf=output)
        for lineno, directive in directives:
            output.write('Line %d : %s' % (lineno, directive))
        rslt = output.getvalue()
        module_inst_list = []
        for line in rslt.splitlines():
            if line.strip().startswith("InstanceList"):
                module_name = line.split(":")[1].strip().split(" ")[0]
                module_line_number = int(line.split(":")[1].strip().split(" ")[2].replace(")",""))
                module_inst_list.append(module_name)
            elif line.strip().startswith("ModuleDef"):
                module_def_name = line.split(":")[1].strip().split(" ")[0] 
                if module_def_name != raw_row["module_name"]:
                    raise Exception("Module def name parsed using pyverilog:({}) is not the same as the one in the dataset:({})".format(module_def_name, raw_row["module_name"]))
        module_inst_list = list(set(module_inst_list))
        module_inst_dict = {raw_row["task_id"]: {"code_name": raw_row["module_name"], "module_inst_list": module_inst_list}}
        with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
            json.dump(module_inst_dict, f, indent=4)
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        return True
    except Exception as e:
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        module_inst_dict = {raw_row["task_id"]: {"code_name": raw_row["module_name"], "code_str": row, "error": str(e)}}
        with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
            json.dump(module_inst_dict, f, indent=4)
        return False

def valid_syntax(row):
    raw_row = row
    row = raw_row["text"]

    file_id = str(uuid.uuid4())
    path = "tmp/file{}.v".format(file_id)
    asset_dir = "tmp/asset{}".format(file_id)
    #check and make asset dir
    if not os.path.exists(asset_dir):
        os.makedirs(os.path.dirname(asset_dir), exist_ok=True)

    tmp_module_inst_dir = "tmp_module_inst_syntax/"
    module_inst_json = "module_inst{}.json".format(raw_row["task_id"])

    with open(path, "w") as f:
        f.write(row)

    try:
        ast, directives = parse([path], debug=False, outputdir=asset_dir, preprocess_output="tmp/preprocess.output.{}".format(file_id))
        output = StringIO()
        ast.show(buf=output)
        for lineno, directive in directives:
            output.write('Line %d : %s' % (lineno, directive))
        rslt = output.getvalue()
        for line in rslt.splitlines():
            if line.strip().startswith("InstanceList"):
                # print("Module instantiation detected...")
                # print(line)
                module_inst_dict = {raw_row["task_id"]: {"code_name": raw_row["module_name"], "code_str": row, "error": "Module instantiation detected"}}
                with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
                    json.dump(module_inst_dict, f, indent=4)
                return False
            #no worries, ModuleDef is always before InstanceList
            elif line.strip().startswith("ModuleDef"):
                module_def_name = line.split(":")[1].strip().split(" ")[0] 
                if module_def_name != raw_row["module_name"]:
                    raise Exception("Module def name parsed using pyverilog:({}) is not the same as the one in the dataset:({})".format(module_def_name, raw_row["module_name"]))
                    raw_row["module_name"] = module_def_name
                    raw_row["module_name_change"] = "Module def name parsed using pyverilog:({}) is not the same as the one in the dataset:({})".format(module_def_name, raw_row["module_name"])
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        module_inst_dict = {raw_row["task_id"]: {"code_name": raw_row["module_name"]}}
        with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
            json.dump(module_inst_dict, f, indent=4)
        return True
    except Exception as e:
        #delete the file
        os.remove(path)
        shutil.rmtree(asset_dir)
        module_inst_dict = {raw_row["task_id"]: {"code_name": raw_row["module_name"], "code_str": row, "error": str(e)}}
        with open(os.path.join(tmp_module_inst_dir, module_inst_json), "w") as f:
            json.dump(module_inst_dict, f, indent=4)
        return False

def merge_module_insts(module_inst_dir="tmp_module_inst/", output_json_name = "module_inst.json"):
    module_inst_dict = {}
    for file_name in os.listdir(module_inst_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(module_inst_dir, file_name), "r") as f:
                module_inst_dict.update(json.load(f))
    with open(os.path.join(output_json_name), "w") as f:
        json.dump(module_inst_dict, f, indent=4)
    return module_inst_dict

def gen_module_name_to_task_id_mapping(module_inst_dict):
    module_name_to_task_id_mapping = {}
    for task_id in module_inst_dict.keys():
        module_name = module_inst_dict[task_id]["code_name"]
        if module_name not in module_name_to_task_id_mapping.keys():
            module_name_to_task_id_mapping[module_name] = []
        module_name_to_task_id_mapping[module_name].append(task_id)
    return module_name_to_task_id_mapping

class VerilogEval_Dataprocess:
    def __init__(self, raw_dataset_name="shailja/Verilog_GitHub",  raw_dataset_cache_dir="./cache", lfd=False):
        self.raw_dataset_name = raw_dataset_name
        self.raw_dataset_cache_dir = raw_dataset_cache_dir
        self.raw_dataset = None 
        self.lfd = lfd
        self.setup_env()
        
    def load_raw_dataset(self):
        if not self.lfd:
            self.raw_dataset = load_dataset(self.raw_dataset_name, cache_dir=self.raw_dataset_cache_dir)
            self.raw_dataset = self.raw_dataset["train"]
            print(self.raw_dataset.shape)
        else:
            self.raw_dataset = load_from_disk(self.raw_dataset_cache_dir)
            print(self.raw_dataset.shape)
        return self.raw_dataset

    
    def setup_env(self):
        os.environ['PATH'] = f'/home/user_name/iverilog/iverilog-11_0/iv_11_install/bin/:{os.environ["PATH"]}'
        
    def f_remove_comments(self, dataset):
        print("=="*20)
        print("Removing comments ...")
        data_content = dataset.map(remove_comments, num_proc=48)
        print(data_content.shape)
        print("=="*20)
        return data_content

    def f_remove_premod_postmod_lines(self, dataset):
        print("=="*20)
        print("Removing premod and postmod lines ...")
        data_content = dataset.map(remove_premod_postmod_lines, num_proc=48)
        print(data_content.shape)
        print("=="*20)
        return data_content
    
    def f_valid_content(self, dataset):
        print("=="*20)
        print("Content filtering ...")
        tmp_module_inst_dir = "tmp_module_inst_valid_content/"
        if not os.path.exists(tmp_module_inst_dir):
            os.makedirs(tmp_module_inst_dir)
        data_content = dataset.filter(valid_content, num_proc=48)
        print(data_content.shape)
        print("=="*20)
        return data_content
    
    def f_valid_len(self, dataset):
        print("=="*20)
        print("Token length filtering ...")
        tmp_module_inst_dir = "tmp_module_inst_valid_len/"
        if not os.path.exists(tmp_module_inst_dir):
            os.makedirs(tmp_module_inst_dir)
        data_content = dataset.filter(valid_len)
        print(data_content.shape)
        print("=="*20)
        return data_content
    
    def f_valid_syntax(self, dataset):
        print("=="*20)
        print("Syntax filtering ...")
        tmp_module_inst_dir = "tmp_module_inst_syntax/"        
        if not os.path.exists(tmp_module_inst_dir):
            os.makedirs(tmp_module_inst_dir)
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        self.setup_env()
        data_content = dataset.filter(valid_syntax, num_proc=32)
        print(data_content.shape)
        print("=="*20)
        return data_content
    
    def f_valid_syntax_with_module_inst(self, dataset):
        print("=="*20)
        print("Syntax filtering with modules...")
        tmp_module_inst_dir = "tmp_module_inst_syntax/"        
        if not os.path.exists(tmp_module_inst_dir):
            os.makedirs(tmp_module_inst_dir)
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        self.setup_env()
        data_content = dataset.filter(valid_syntax_with_module_inst, num_proc=32)
        print(data_content.shape)
        print("=="*20)
        return data_content
    
    def f_de_duplicate(self, dataset, threshold=0.8):
        print("=="*20)
        print("De-duplicating ...")
        data_content, duplicate_clusters = deduplicate_dataset(dataset, jaccard_threshold=threshold)
        #dump the duplicate clusters
        with open("duplicate_clusters.json", "w") as f:
            json.dump(duplicate_clusters, f, indent=4)
        print("=="*20)
        return data_content

    #TODO: use simpler deduplication method: https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html 
    
    
if __name__ == "__main__":
    #take arg of the output raw dataset dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset_output_dir", help="output dir of the raw dataset", type=str)
    args = parser.parse_args()
    raw_dataset_output_dir = args.raw_dataset_output_dir

    _ , module_num = separate_modules_in_dataset("shailja/Verilog_GitHub", "./cache", output_dir="./ckpt_separated_modules")
    # exit()

    VerilogEval_Dataprocess0 = VerilogEval_Dataprocess(raw_dataset_name="shailja/Verilog_GitHub",  raw_dataset_cache_dir="./ckpt_separated_modules", lfd=True)
    VerilogEval_Dataprocess0.load_raw_dataset()
    dataset = VerilogEval_Dataprocess0.f_remove_comments(VerilogEval_Dataprocess0.raw_dataset)
    dataset.save_to_disk("./ckpt1_user_name_remove_comments")
    dataset = VerilogEval_Dataprocess0.f_remove_premod_postmod_lines(dataset)
    dataset.save_to_disk("./ckpt2_user_name_remove_premod_postmod_lines")

    dataset = VerilogEval_Dataprocess0.f_valid_content(dataset)
    dataset.save_to_disk("./ckpt3_user_name_valid_content")
    dataset = VerilogEval_Dataprocess0.f_valid_len(dataset)
    dataset.save_to_disk("./ckpt3_user_name_before_syntax")
    
    # dataset = VerilogEval_Dataprocess0.f_valid_syntax(dataset)
    # dataset.save_to_disk("./ckpt3_user_name_no_prelines")
    dataset = VerilogEval_Dataprocess0.f_valid_syntax_with_module_inst(dataset)
    dataset.save_to_disk("./ckpt3_user_name_no_prelines_with_module_inst")
    dataset = VerilogEval_Dataprocess0.f_de_duplicate(dataset, threshold=0.8)
    print(dataset.shape)
    dataset.save_to_disk("./ckpt4_user_name_de_dup_with_module_inst")

    module_inst_dict = merge_module_insts(module_inst_dir = "tmp_module_inst_syntax/")
    module_name_to_task_id_mapping = gen_module_name_to_task_id_mapping(module_inst_dict)
    print(len(module_inst_dict.keys()))
    print(len(module_name_to_task_id_mapping.keys()))
    #dump the mapping as deduplication is not decisive, use the version before deduplication
    with open("module_name_to_task_id_mapping.json", "w") as f:
        json.dump(module_name_to_task_id_mapping, f, indent=4)
    # exit()

    #if want to use the module inst to recover the raw dataset generation ckpt3_user_name_before_syntax is needed, as the idx of the json is based on ckpt3_user_name_before_syntax 
    dataset_to_store = load_from_disk("./ckpt4_user_name_de_dup_with_module_inst")
    # raw_dataset_output_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content"
    store_dataset_entries_to_file(dataset_to_store, raw_dataset_output_dir)
    # exit()