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
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
from ast import literal_eval
import numpy as np
import re 
from tqdm import tqdm


#############################################################################################################
# Verilog / HLS code & comment csv asset generator
#############################################################################################################



def create_empty_csv(fields, path):
    df = pd.DataFrame(columns=fields)
    df.to_csv(path, index=False)

def remove_empty_lines(fname):
    #remove lines with empty content (only contains "\n")
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip() != "":
            new_lines.append(line.replace("\n", ""))
    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))

def merge_lines(fname):
    #merge entries with the same line_number
    df = pd.read_csv(fname)
    # #find the line where content is not string and its line_number
    # for i in range(len(df)):
    #     if not isinstance(df.iloc[i]["content"], str):
    #         print("line_number: ", df.iloc[i]["line_number"])
    #         print("content: ", df.iloc[i]["content"])
    df['content'] = df['content'].astype(str)
    df = df.groupby("line_number").agg({"content": lambda x: ' '.join(x)})
    df.to_csv(fname, index=True)

def convert_raw_src_code_to_csv(src_code_file, csv_code_file, csv_comment_file, discard_original_comment = False):
    remove_empty_lines(src_code_file)
    #read the file line by line
    with open(src_code_file, "r") as f:
        lines = f.readlines()
    #create a csv file with the fields
    create_empty_csv(["content", "line_number"], csv_code_file)
    create_empty_csv(["content", "line_number"], csv_comment_file)
    #create an entry for each line
    code_line_number = 0
    comment_line_number = 0
    prev_line_is_comment = False
    for line_id, line in enumerate(lines):
        #check if the line is a comment
        #potentially have "tab" or "space" before the comment
        if line.strip().startswith("//"):
            if not prev_line_is_comment:
                df = pd.DataFrame({"content": [line], "line_number": [comment_line_number]})
                df.to_csv(csv_comment_file, mode='a', header=False, index=False)
                comment_line_number += 1
                prev_line_is_comment = True
            else:
                #append to the previous comment (comment_line_number-1)
                df = pd.DataFrame({"content": [line], "line_number": [comment_line_number-1]})
                df.to_csv(csv_comment_file, mode='a', header=False, index=False)
        else:
            df = pd.DataFrame({"content": [line], "line_number": [code_line_number]})
            df.to_csv(csv_code_file, mode='a', header=False, index=False)
            code_line_number += 1
            if not prev_line_is_comment:
                df = pd.DataFrame({"content": ['no_comment'], "line_number": [comment_line_number]})
                df.to_csv(csv_comment_file, mode='a', header=False, index=False)
                comment_line_number += 1
            prev_line_is_comment = False
    #merge entries with the same line_number
    merge_lines(csv_comment_file)
    #if discard_original_comment, overwrite the csv_comment_file with empty "no_comment" for all lines
    if discard_original_comment:
        df = pd.read_csv(csv_comment_file)
        df["content"] = "no_comment"
        df.to_csv(csv_comment_file, index=False)

def load_code_lines(df, line_start, line_end):
    #[line_start, line_end)
    #csv file, with entry "content" and "line_number"
    #can also be used to load comment file
    code_lines = []
    #need to cover cases where line_start < 0 or line_end > max_line_number
    if line_start < 0:
        line_start = 0
    if line_end > len(df):
        line_end = len(df)
    for line_id in range(line_start, line_end):
        #get the content based on line_number entry in the csv file
        code_line = df[df["line_number"] == line_id]["content"].values[0]
        code_lines.append(code_line)
    return code_lines

def load_code_lines_scatter(df, line_ids: list = []):
    code_lines = []
    for line_id in line_ids:
        #get the content based on line_number entry in the csv file
        code_line = df[df["line_number"] == line_id]["content"].values[0]
        code_lines.append(code_line)
    return code_lines

def merge_code_and_comment(df_code_file, df_comment_file):
    output_str = ""
    df_code = pd.read_csv(df_code_file)
    df_comment = pd.read_csv(df_comment_file)
    for line_id in range(len(df_code)):
        code_line = df_code[df_code["line_number"] == line_id]["content"].values[0]
        comment_line = df_comment[df_comment["line_number"] == line_id]["content"].values[0]
        if comment_line != "no_comment":
            #separate the comment_line into multiple lines
            inner_lines = comment_line.split("\n")[:-1]
            for inner_line in inner_lines:
                if inner_line.strip().startswith("//"):
                    output_str += (inner_line + "\n")
                else:
                    output_str += ("// "+inner_line + "\n")
        code_line = code_line.replace("\n", "")
        output_str += code_line + "\n"
    return output_str


def merge_brackets_str(file_content):
    #merge lines with only "{" or "}" with the previous line
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip() == "{" or line.strip() == "}":
            new_lines[-1] = new_lines[-1].strip() + line.strip()
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def remove_empty_lines_str(file_content):
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip() != "":
            new_lines.append(line.replace("\n", ""))
    return "\n".join(new_lines)

def flatten_paranthesis_str(file_content):
    lines = file_content.split("\n")
    new_lines = []
    unclosed_paranthesis = 0
    for line_indx, line in enumerate(lines):
        if unclosed_paranthesis > 0:
            new_lines[-1] = new_lines[-1].strip() + line.strip()
        else:
            new_lines.append(line)
        unclosed_paranthesis += len(re.findall(r"\(", line))
        unclosed_paranthesis -= len(re.findall(r"\)", line))
    return "\n".join(new_lines)

def flatten_semicolon(file_content):
    #if a line start with a semicolon, merge it with the previous line
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip().startswith(";"):
            new_lines[-1] = new_lines[-1].strip() + line.strip()
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def flatten_module(file_content):
    #if the previous line only have "module", merge it with the current line
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if len(new_lines) > 0 and new_lines[-1].strip() == "module" :
            new_lines[-1] = new_lines[-1].strip()+ " " + line.strip()
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def remove_in_line_comments(file_content):
    #remove in line comments
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if "//" in line:
            line = line.split("//")[0]
        new_lines.append(line)

    #remove /* */ comments 
    new_lines_str = "\n".join(new_lines)
    new_lines_str = re.sub(r"/\*.*?\*/", "", new_lines_str, flags=re.DOTALL)
    return new_lines_str

def force_separate_module_args(file_content):
    lines = file_content.split("\n")
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip().startswith("module") and (re.search(r'\)\s*;', line) is not None):
            #split based on "\)\s*;"
            new_lines.append(re.sub(r'\)\s*;', r');\n', line))
        elif line.strip().startswith("module") and (";" in line):
            #split based on ";"
            new_lines.append(re.sub(r';', r';\n', line))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)
#############################################################################################################
# prompt generator
#############################################################################################################
def form_function_retrival_prompt(df, comment_file, line_ids: list = []):
    #load current line
    curr_line = load_code_lines_scatter(df, line_ids)
    prompt = ""
    for relative_line_idx in range(len(curr_line)):
        prompt += curr_line[relative_line_idx]
        prompt += "\n"
    return prompt

def form_commenting_prompt(df, comment_file, line_ids, 
                           trace_back_lines = 15, look_forward_lines = 15, 
                           helper_function_names = [], helper_function_summarys = [],
                           format_instruction = None):
    #sort line ids
    line_ids.sort() 
    prompt = ""
    #load previous lines
    prev_lines = load_code_lines(df, line_ids[0]-trace_back_lines, line_ids[0])
    #load current line
    curr_lines = load_code_lines_scatter(df, line_ids)
    #load next lines
    next_lines = load_code_lines(df, line_ids[-1]+1, line_ids[-1]+1+look_forward_lines)

    #load previous comments
    prev_comments = load_code_lines(comment_file, line_ids[0]-trace_back_lines, line_ids[0])
    #load current comment
    curr_comment = load_code_lines_scatter(comment_file, line_ids)
    #load next comments
    next_comments = load_code_lines(comment_file, line_ids[-1]+1, line_ids[-1]+1+look_forward_lines)
    
    #form prompt
    if len(prev_lines) > 0:
        prompt += "For your reference only, here are the previous {} lines of code with comments:\n".format(trace_back_lines)
        prompt += 'Previous code and comments:\n'
        prompt += '```\n'
        for relative_line_idx in range(len(prev_lines)):
            if prev_comments[relative_line_idx] != "no_comment":
                prompt += str(line_ids[0]-trace_back_lines+relative_line_idx) + ": "+ prev_comments[relative_line_idx]
            prompt += str(line_ids[0]-trace_back_lines+relative_line_idx) + ": " +prev_lines[relative_line_idx]
        prompt += '```\n'

    if len(next_lines) > 0:
        prompt += "For your reference only, here are the next {} lines of code with comments:\n".format(look_forward_lines)
        prompt += "Code:\n"
        prompt += '```\n'
        for relative_line_idx in range(len(next_lines)):
            if next_comments[relative_line_idx] != "no_comment":
                prompt += str(line_ids[-1]+1+relative_line_idx) + ": "+ next_comments[relative_line_idx]
            prompt += str(line_ids[-1]+1+relative_line_idx) + ": " +next_lines[relative_line_idx]
        prompt += '```\n'

    if helper_function_names is not None:
        for helper_function_name, helper_function_summary in zip(helper_function_names, helper_function_summarys):
            prompt += "Helper function summary for {} for your reference:\n".format(helper_function_name)
            prompt += '```\n'
            prompt += helper_function_summary + "\n"  
            prompt += '```\n'

    prompt += "Based on the reference, add comments for the following code lines, with line numbers {}:\n".format(line_ids)
    # prompt += "Current codes and comments:\n"
    prompt += '```\n'
    for relative_line_idx in range(len(curr_lines)):
        if curr_comment[relative_line_idx] != "no_comment":
            prompt += str(line_ids[relative_line_idx]) + ": "+ curr_comment[relative_line_idx]
        prompt += str(line_ids[relative_line_idx]) + ": " +curr_lines[relative_line_idx]
    prompt += '```\n'
    prompt += "Add concrete and precise comments; only add comments when the code is not clear; do not include code in the comments.\n"
    # prompt += "Do you think it's necessary to add and only add additional comments to each line of code in line {}?\n".format(line_ids)
    if format_instruction is not None:
        prompt += format_instruction + "\n"
    else:
        prompt +=  """Format your answer in json format, with entries of "comment_exist", "comment", and "line_number"; \n "comment_exist" is a List of boolean value denating if comment exist for each code line.\n "comment" is list of string comments, each of which is the comment for the corresponding code line; do not include original code here; empty string if comment does not exist. \n "line_number" is {line_numbers}. \n Here is the response format: {"comment_exist": [bool, bool], "comment": [str comment, str comment], "line_number": {line_numbers}}\n Only include the json response! Do not include anything else!\n""".replace("{line_numbers}", str(line_ids))
        # prompt += 'Format your answer in json format, with entries of "comment_exist", "comment", and "line_number"; value for each of the entry is a list, with elements corresponding to each line of the code; for example, {"comment_exist": [True, False], "comment": ["This is a comment.", ""], "line_number": [10,11]}\n'
    return prompt


#############################################################################################################
# HLS code partitioner
#############################################################################################################


def is_function_definition(prev_line):
    combined = prev_line.strip()
    if combined.startswith("//"):
        return False
    if combined.startswith("/*"):
        return False
    if combined.startswith("*"):
        return False
    if combined.startswith("*/"):
        return False
    return re.match(r'\s*(\S+)\s+(\S+)\s*\(', combined) is not None or \
           re.match(r"template\s*<.*>\s*(\S+)\s+(\S+)\s*\(", combined) is not None or \
           re.match(r'\s*(\S+)\s+(\S+)\s+(\S+)\s*\(', combined) is not None or \
           re.match(r"template\s*<.*>\s*(\S+)\s+(\S+)\s+(\S+)\s*\(", combined) is not None or\
           re.match(r"template\s*" , combined) is not None


def is_array_init(curr_line):
    combined = curr_line.strip()
    if combined.startswith("//"):
        return False
    if combined.startswith("/*"):
        return False
    if combined.startswith("*"):
        return False
    if combined.startswith("*/"):
        return False
    return re.match(r"\s*\S+\s+\S+\s*.*\s*=", combined) is not None or \
           re.match(r"\s*\S+\s+\S+\s+\S+\s*.*\s*=", combined) is not None  


def extract_functions(file_content):
    functions = []
    in_function = False
    args_passed = False
    function_ended = True
    function_buffer = ""
    unclosed_brackets_func = 0
    function_num = 0
    func_seg = {}
    file_content = file_content.replace("\t", "    ")
    file_content = remove_in_line_comments(file_content)
    file_content = flatten_semicolon(file_content)
    file_content = remove_empty_lines_str(file_content)
    file_content = merge_brackets_str(file_content)
    file_content = flatten_paranthesis_str(file_content)
    file_content = remove_in_line_comments(file_content)
    for line_idx, curr_line in enumerate(file_content.split("\n")):
        if is_function_definition(curr_line) and function_ended:
            in_function = True
            function_ended = False
            #remove in line comment 
            curr_line = re.sub(r"//.*", "", curr_line)
            function_buffer += curr_line + " "
            #find all "{"
            unclosed_brackets_func += len(re.findall(r"\{", curr_line))
            if unclosed_brackets_func > 0:
                args_passed = True
            #find all "}"
            unclosed_brackets_func -= len(re.findall(r"\}", curr_line))
            
            #substitute non_letter and non_digit with "_"
            func_seg[function_num] = {}
            func_seg[function_num]["func_num"] = function_num
            func_seg[function_num]["lines"] = []
            func_seg[function_num]["lines"].append(line_idx)
            if unclosed_brackets_func == 0:
                function_ended = True
                in_function = False
                args_passed = False
                functions.append(function_buffer)
                func_name = re.findall(r"\s+\S+\s*\(", function_buffer.replace("\n", ""))[0].split("(")[0].strip()
                func_seg[function_num]["func_name"] = func_name
                function_buffer = ""
                function_num += 1
            
        elif in_function and not function_ended:
            unclosed_brackets_func += len(re.findall(r"\{", curr_line))
            if unclosed_brackets_func > 0:
                args_passed = True
            unclosed_brackets_func -= len(re.findall(r"\}", curr_line))
            if unclosed_brackets_func > 0 or not args_passed:
                if not args_passed:
                    function_buffer = function_buffer.replace("\n", " ")
                    #replace multiple spaces with single space
                    function_buffer = re.sub(r"\s+", " ", function_buffer)
                    #replace tab with space
                    function_buffer = function_buffer.replace("\t", " ")
                    function_buffer += re.sub(r"//.*", "", curr_line) + " " 
                else:
                    function_buffer += curr_line + "\n"
            else:
                in_function = False
                args_passed = False
                function_ended = True
                function_buffer += curr_line + "\n"
                functions.append(function_buffer)
                func_name = re.findall(r"\s+\S+\s*\(", function_buffer.replace("\n", ""))[0].split("(")[0].strip()
                func_seg[function_num]["func_name"] = func_name
                function_buffer = ""
                func_seg[function_num]["lines"].append(line_idx)
                function_num += 1
    return functions, func_seg

def exclude_function_lines(file_content, func_seg):
    #find all code lines that are not in any function
    raw_code_lines = file_content.split("\n")
    new_code_lines = []
    new_code_start_line = 0
    new_code_end_line = 0
    for func_num in func_seg:
        func_start_line = func_seg[func_num]["lines"][0]
        for line_idx in range(new_code_start_line, func_start_line):
            new_code_lines.append(raw_code_lines[line_idx])
        new_code_start_line = func_seg[func_num]["lines"][-1] + 1
    for line_idx in range(new_code_start_line, len(raw_code_lines)):
        new_code_lines.append(raw_code_lines[line_idx])
    new_code_end_line = len(new_code_lines)
    new_code = "\n".join(new_code_lines)
    return new_code

def extract_headers(file_content, func_seg):
    headers = []
    file_content = exclude_function_lines(file_content, func_seg)
    for curr_line in file_content.split("\n"):
        if curr_line.startswith("//"):
            continue
        if curr_line.startswith("/*"):
            continue
        if curr_line.startswith("*"):
            continue
        if curr_line.startswith("*/"):
            continue
        if re.match(r"#include|#define|typedef", curr_line):
            headers.append(curr_line)
    return headers

def extract_variables(file_content, func_seg):
    variables = []
    file_content=exclude_function_lines(file_content, func_seg)
    for curr_line in file_content.split("\n"):
        if curr_line.startswith("//"):
            continue
        if curr_line.startswith("/*"):
            continue
        if curr_line.startswith("*"):
            continue
        if curr_line.startswith("*/"):
            continue
        if ";" in curr_line and "}" not in curr_line and "{" not in curr_line:
            variables.append(curr_line)
    return variables


def extract_arrays(file_content, func_seg):
    arrays = []
    in_array = False
    array_ended = True
    array_buffer = ""
    unclosed_brackets_array = 0

    file_content=exclude_function_lines(file_content, func_seg)

    for curr_line in file_content.split("\n"):
        if is_array_init(curr_line) and array_ended:
            in_array = True
            array_ended = False
            array_buffer += curr_line + "\n"
            #find all "{"
            unclosed_brackets_array += len(re.findall(r"\{", curr_line))
            #find all "}"
            unclosed_brackets_array -= len(re.findall(r"\}", curr_line))
        elif in_array and not array_ended:
            unclosed_brackets_array += len(re.findall(r"\{", curr_line))
            unclosed_brackets_array -= len(re.findall(r"\}", curr_line))
            if unclosed_brackets_array > 0:
                array_buffer += curr_line + "\n"
            else:
                in_array = False
                array_ended = True
                array_buffer += curr_line + "\n"
                arrays.append(array_buffer)
                array_buffer = ""
    return arrays




def extract_headers_functions_and_variables(file_content):
    #replace tab with 4 spaces
    file_content = file_content.replace("\t", "    ")
    functions, func_seg= extract_functions(file_content)
    arrays = extract_arrays(file_content, func_seg)
    headers = extract_headers(file_content, func_seg)
    variables = extract_variables(file_content, func_seg)
    return functions, arrays, headers, variables, func_seg

#TODO: better to also use anything_other_than_function mode for easier compilation?
def write_partitioned_files(functions, arrays, headers, variables, func_seg, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx, function in enumerate(functions):
        function_name = func_seg[idx]["func_name"]
        used_variables = []
        used_arrays = []
        
        for var in variables:
            var_name = re.sub(r"\[.*", "", var).strip()
            var_name = re.sub(r"=.*", "", var_name).strip()
            var_name = var_name.split()[-1]
            var_name = var_name.replace("*", "")
            var_name = var_name.replace("&", "")
            var_name = var_name.replace(";", "")
            if var_name in function:
                used_variables.append(var)
        
        for arr in arrays:
            arr_name = arr.split("\n")[0].strip()
            arr_name = re.sub(r"\[.*", "", arr_name).strip()
            arr_name = re.sub(r"=.*", "", arr_name).strip()
            arr_name = arr_name.split()[-1]
            arr_name = arr_name.replace("*", "")
            arr_name = arr_name.replace("&", "")
            arr_name = arr_name.replace(";", "")  
            if arr_name in function:
                used_arrays.append(arr)
        
        with open(f"{output_dir}/{function_name}.cpp", "w") as f:
            for header in headers:
                f.write(header + "\n")
            f.write("\n")
            for var in used_variables:
                f.write(var + "\n")
            for arr in used_arrays:
                f.write(arr + "\n")
            f.write("\n")
            f.write(function)


def part_hls_function(fname, output_dir):
    #partion an hls code file into multiple files each containing one function definition
    #keep the include files in the original file
    #assume the file is in the format of:
    #include files
    #function definition 
    #fname: <function_name>.<original_file_extension>
    #return: a list of file names
    #e.g., part_hls_function("sub_module.h")
    #return: ["<sub_module1>.h", "<sub_module2>.h", ...]

    with open(fname, "r") as f:
        file_content = f.read()

    functions, arrays, headers, variables, func_seg = extract_headers_functions_and_variables(file_content)
    if len(functions) == 0:
        print("No function found in the file.")
        return
    write_partitioned_files(functions, arrays, headers, variables, func_seg, output_dir)
    

            
#############################################################################################################
# Verilog code partitioner
#############################################################################################################

def extract_module_header(fname, code_str=False):
    if code_str:
        code_lines = fname.split("\n")
    else:
        code_lines = open(fname, "r").readlines()
    module_header_end_line = 0
    #find the module header end line
    for i, line in enumerate(code_lines):
        if ";" in line:
            module_header_end_line = i
            break
    module_header = "\n".join(code_lines[:module_header_end_line+1])
    module_header = module_header.split(";", 1)[0]+";"
    return module_header

def remove_newlines_in_module_header(fname):
    module_header = extract_module_header(fname)
    new_module_header = module_header.replace("\n", " ")
    new_module_header = re.sub(r"\t+", " ", new_module_header)
    new_module_header = re.sub(r"\s+", " ", new_module_header)
    code_content = open(fname, "r").read()
    code_content = new_module_header + code_content.split(";", 1)[1]
    with open(fname, "w") as f:
        f.write(code_content)



def pre_process_orig_comments(fname):
    #track or original comments
    comment_lines = []
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
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
            if "//" in line:
                line = line.split("//")[0]
            new_lines.append(line)

    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))


def merge_brackets(fname):
    #merge lines with only "{" or "}" with the previous line
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
    new_lines = []
    for line_indx, line in enumerate(lines):
        if line.strip() == "{" or line.strip() == "}":
            new_lines[-1] = new_lines[-1].strip() + line.strip()
        else:
            new_lines.append(line)
    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))


def flatten_paranthesis(fname):
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
    new_lines = []
    unclosed_paranthesis = 0
    for line_indx, line in enumerate(lines):
        if unclosed_paranthesis > 0:
            new_lines[-1] = new_lines[-1].strip() + line.strip()
        else:
            new_lines.append(line)
        unclosed_paranthesis += len(re.findall(r"\(", line))
        unclosed_paranthesis -= len(re.findall(r"\)", line))
    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))


def merge_module_param_and_args(fname):
    #merge the module parameters and arguments in the same line
    #NOTE: only works for verilog; only used after remove_empty_lines and flatten_paranthesis
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
    new_lines = []
    output_str_list, module_name_list = part_verilog_module_string("".join(lines))
    unclosed_paranthesis = 0
    
    for line_indx, line in enumerate(lines):
        if line_indx > 0:
            if "#" in lines[line_indx - 1] and ";" not in lines[line_indx - 1] and module_name_list[0] in lines[line_indx - 1]:
                new_lines[-1] = new_lines[-1].strip() + " " + line.strip()
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))


def sub_multi_space_tab_with_single_space(fname):
    with open(os.path.join(fname), "r") as f:
        lines = f.readlines()
    new_lines = []
    for line_indx, line in enumerate(lines):
        new_line = re.sub(r"\s+", " ", line)
        new_line = re.sub(r"\t+", " ", new_line)
        new_lines.append(new_line)
    with open(os.path.join(fname), "w") as f:
        f.write("\n".join(new_lines))


def preprocess(fname, discard_original_comment = True, rtl = True):
    if discard_original_comment:
        pre_process_orig_comments(fname)
    merge_brackets(fname)
    flatten_paranthesis(fname)
    remove_empty_lines(fname)
    if rtl:
        merge_module_param_and_args(fname)
    remove_empty_lines(fname)
    remove_newlines_in_module_header(fname)
    sub_multi_space_tab_with_single_space(fname)
    


def is_module_definition(prev_line):
    combined = prev_line.strip()
    if combined.startswith("//"):
        return False
    if combined.startswith("/*"):
        return False
    if combined.startswith("*"):
        return False
    if combined.startswith("*/"):
        return False
    return re.match(r'module\s+\S', combined) is not None 

def extract_modules(file_content):
    modules = []
    in_module = False
    args_passed = False
    module_ended = True
    module_buffer = ""
    unclosed_brackets_mod = 0
    mod_num = 0
    mod_seg = {}
    #replace tab with 4 spaces
    file_content = file_content.replace("\t", "    ")
    file_content = remove_in_line_comments(file_content)
    file_content = flatten_semicolon(file_content)
    file_content = remove_empty_lines_str(file_content)
    file_content = merge_brackets_str(file_content)
    file_content = flatten_paranthesis_str(file_content)
    file_content = flatten_module(file_content)
    file_content = force_separate_module_args(file_content)
    file_content = remove_in_line_comments(file_content)
    for line_idx, curr_line in enumerate(file_content.split("\n")):
        if is_module_definition(curr_line) and module_ended:
            in_module = True
            module_ended = False
            #remove in line comment 
            curr_line = re.sub(r"//.*", "", curr_line)
            module_buffer += curr_line + " "
            if re.search(r'\)\s*;', curr_line) is not None or (";" in curr_line and "module" in curr_line):
                args_passed = True
            #substitute non_letter and non_digit with "_"
            mod_seg[mod_num] = {}
            mod_seg[mod_num]["mod_num"] = mod_num
            mod_seg[mod_num]["lines"] = []
            mod_seg[mod_num]["lines"].append(line_idx)
            if "endmodule" in curr_line:
                module_ended = True
                in_module = False
                args_passed = False
                modules.append(module_buffer)
                try:
                    mod_name = re.findall(r"module\s+\S+\s*(?:import)?[\(\#;`]|module\s+\S+\s*import", module_buffer.replace("\n", ""))[0]
                    mod_name = re.split("\s+", mod_name)[1].strip()
                    mod_name = mod_name.split("(")[0].strip().split("#")[0].strip().split(";")[0].strip().split("import")[0].strip()
                except:
                    mod_name = "no_name_found"
                mod_seg[mod_num]["mod_name"] = mod_name
                module_buffer = ""
                mod_num += 1
        elif in_module and not module_ended:
            if re.search(r'\)\s*;', curr_line) is not None or (";" in curr_line and "module" in curr_line):
                args_passed = True
            if "endmodule" in curr_line:
                module_ended = True
            if not module_ended or not args_passed:
                if not args_passed:
                    module_buffer = module_buffer.replace("\n", " ")
                    #replace multiple spaces with single space
                    module_buffer = re.sub(r"\s+", " ", module_buffer)
                    #replace tab with space
                    module_buffer = module_buffer.replace("\t", " ")
                    module_buffer += re.sub(r"//.*", "", curr_line) + " " 
                else:
                    module_buffer += curr_line + "\n"
            else:
                in_module = False
                args_passed = False
                module_ended = True
                module_buffer += curr_line + "\n"
                modules.append(module_buffer)
                try:
                    mod_name = re.findall(r"module\s+\S+\s*(?:import)?[\(\#;`]|module\s+\S+\s*import", module_buffer.replace("\n", ""))[0]
                    mod_name = re.split("\s+", mod_name)[1].strip()
                    mod_name = mod_name.split("(")[0].strip().split("#")[0].strip().split(";")[0].strip().split("import")[0].strip()
                except:
                    mod_name = "no_name_found"
                mod_seg[mod_num]["mod_name"] = mod_name
                module_buffer = ""
                mod_seg[mod_num]["lines"].append(line_idx)
                mod_num += 1
    return modules, mod_seg, file_content

def exclude_module_lines(file_content, mod_seg):
    #find all code lines that are not in any function
    raw_code_lines = file_content.split("\n")
    new_code_lines = []
    new_code_start_line = 0
    new_code_end_line = 0
    for mod_num in mod_seg:
        mod_start_line = mod_seg[mod_num]["lines"][0]
        for line_idx in range(new_code_start_line, mod_start_line):
            new_code_lines.append(raw_code_lines[line_idx])
        new_code_start_line = mod_seg[mod_num]["lines"][-1] + 1
    for line_idx in range(new_code_start_line, len(raw_code_lines)):
        new_code_lines.append(raw_code_lines[line_idx])
    new_code_end_line = len(new_code_lines)
    new_code = "\n".join(new_code_lines)
    return new_code

def write_partitioned_verilog_files(modules, anything_other_than_mod, mod_seg, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, module in enumerate(modules):
        module_name = mod_seg[idx]["mod_name"]
        with open(f"{output_dir}/{module_name}.v", "w") as f:
            f.write(anything_other_than_mod)
            f.write("\n")
            f.write(module)

def write_partitioned_verilog_string(modules, anything_other_than_mod, mod_seg):
    #return a string
    output_str_list = []
    module_name_list = []
    for idx, module in enumerate(modules):
        output_str = ""
        module_name_list.append(mod_seg[idx]["mod_name"])
        output_str += anything_other_than_mod
        output_str += "\n"
        output_str += module
        output_str_list.append(output_str)
    return output_str_list, module_name_list

def extract_modules_and_others(file_content):
    modules, mod_seg, new_file_content = extract_modules(file_content)
    anything_other_than_mod = exclude_module_lines(new_file_content, mod_seg)
    return modules, anything_other_than_mod, mod_seg

def part_verilog_module(fname, output_dir):
    #partion a verilog code file into multiple files each containing one module definition
    #keep the include files in the original file
    #assume the file is in the format of:
    #include files
    #module definition 
    #fname: <module_name>.<original_file_extension>
    #return: a list of file names
    #e.g., part_verilog_module("sub_module.v")
    #return: ["<sub_module1>.v", "<sub_module2>.v", ...]

    with open(fname, "r") as f:
        file_content = f.read()

    try:
        modules, anything_other_than_mod, mod_seg = extract_modules_and_others(file_content)
    except Exception as e:
        print(e)
        print("Error in file {}".format(fname))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        code_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, code_name, exc_tb.tb_lineno)   
        return
    if len(modules) == 0:
        print("No module found in the file {}.".format(fname))
        exit()
        return
    write_partitioned_verilog_files(modules, anything_other_than_mod, mod_seg, output_dir)

def part_verilog_module_string(module_str):
    modules, anything_other_than_mod, mod_seg = extract_modules_and_others(module_str)
    if len(modules) == 0:
        # print("No module found in the file.")
        # print(module_str)
        return [], []
    output_str_list, module_name_list = write_partitioned_verilog_string(modules, anything_other_than_mod, mod_seg)
    return output_str_list, module_name_list

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_code_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content/", help="source code directory")
    parser.add_argument("--src_code_metadata_file", type=str, default="/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/verilogeval_datagen/process_data/module_inst.json", help="source code metadata file")
    parser.add_argument("--output_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed/", help="output directory")
    parser.add_argument("--shared_lib_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib/", help="shared library directory")
    parser.add_argument("--output_code_metadata_dir", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/", help="output code metadata directory")
    parser.add_argument("--output_code_metadata_file", type=str, default="codes.json", help="output code metadata file")
    parser.add_argument("--skip_shared_lib_gen", action="store_true", help="skip shared library generation")
    parser.add_argument("--generating_from_existing_file_structure", action="store_true", help="generating from existing file structure")
    parser.add_argument("--existing_folder_to_refer", type=str, default="/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed/", help="existing folder to refer")
    parser.add_argument("--module_to_task_id_map_file", type=str, default="/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/verilogeval_datagen/process_data/module_name_to_task_id_mapping.json", help="module to task id map file")
    args = parser.parse_args()
    
    src_code_dir = args.src_code_dir
    src_code_metadata_file = args.src_code_metadata_file
    output_dir = args.output_dir
    shared_lib_dir = args.shared_lib_dir
    output_code_metadata_dir = args.output_code_metadata_dir
    output_code_metadata_file = args.output_code_metadata_file
    file_list = os.listdir(src_code_dir)

    skip_shared_lib_gen = args.skip_shared_lib_gen

    generating_from_existing_file_structure = args.generating_from_existing_file_structure
    existing_folder_to_refer = args.existing_folder_to_refer
    module_to_task_id_map_file = args.module_to_task_id_map_file

    src_code_metadata = json.load(open(src_code_metadata_file, "r"))

    if not skip_shared_lib_gen:
        #create shared library
        print("creating shared library")
        if os.path.exists(shared_lib_dir):
            shutil.rmtree(shared_lib_dir)
        os.makedirs(shared_lib_dir)
        lib_list = {}
        for file in tqdm(file_list, total=len(file_list)):
            if file.endswith(".v"):
                src_code_file = os.path.join(src_code_dir, file)
                code_content = open(src_code_file, "r").read()
                output_str_list, module_name_list = part_verilog_module_string(code_content)
                task_id = file.split(".v")[0]
                for output_str, module_name in zip(output_str_list, module_name_list):
                    lib_list[module_name] = task_id
                    
        #only if the module is instantiated in the other code, then put it into the shared library
        shared_lib = {}
        module_to_be_removed = copy.deepcopy(list(lib_list.keys()))

        for module in tqdm(lib_list, total=len(lib_list)):
            task_id = lib_list[module]
            module_instances = src_code_metadata[task_id]["module_inst_list"]
            if "error" in src_code_metadata[task_id].keys():
                raise Exception("This code id {} (module name {}) has error; and should not be selected.".format(task_id, module))
            if len(module_instances) == 0:
                continue
            for other_module in module_to_be_removed:
                if other_module == module:
                    continue
                #check if other_module is instantiated in module
                if other_module in module_instances:
                    shared_lib[other_module] = open(os.path.join(src_code_dir, lib_list[other_module]+".v"), "r").read()
                    module_to_be_removed.remove(other_module)

        #store the shared library
        for module in tqdm(shared_lib, total=len(shared_lib)):
            with open(os.path.join(shared_lib_dir, module+".v"), "w") as f:
                f.write(shared_lib[module])    
        print("Shared lib length: {}".format(len(shared_lib)))
    
    total_clusters =  (len(file_list) // 1000) + 1
    if not generating_from_existing_file_structure:
        print("Generating src to be documented...")
        #evenly separate the files into 10 parts
        file_clusters  = [[] for _ in range(total_clusters)]
        for idx, file in enumerate(file_list):
            file_clusters[idx%total_clusters].append(file)
        
        for idx, file_cluster in enumerate(file_clusters):    
            print("partitioning part {}".format(idx))
            if not os.path.exists(os.path.join(output_code_metadata_dir, "part"+str(idx))):
                os.makedirs(os.path.join(output_code_metadata_dir, "part"+str(idx)))
            code_metadata = {}
            for file in tqdm(file_cluster, total=len(file_cluster)):
                if file.endswith(".v"):
                    #copy the code metadata
                    task_id = file.split(".v")[0]
                    module_name = src_code_metadata[task_id]["code_name"]
                    code_metadata[module_name] = src_code_metadata[task_id]
                    if "error" in src_code_metadata[task_id].keys():
                        raise Exception("This code id {} (module name {}) has error; and should not be selected.".format(task_id, module_name))
                    src_code_file = os.path.join(src_code_dir, file)
                    part_verilog_module(src_code_file, os.path.join(output_dir, "part"+str(idx)))
            #store the code metadata
            with open(os.path.join(output_code_metadata_dir, "part"+str(idx), output_code_metadata_file), "w") as f:
                json.dump(code_metadata, f, indent=4)
            out_files = os.listdir(os.path.join(output_dir, "part"+str(idx)))
            print(len(out_files))
    else:
        print("Generating src to be documented...")
        module_to_task_id_map = json.load(open(module_to_task_id_map_file, "r"))
        #evenly separate the files into 10 parts
        for idx in range(total_clusters):
            print("Part {}".format(idx))
            file_list_in_ref_folder = os.listdir(os.path.join(existing_folder_to_refer, "part"+str(idx)))
            #creating code metadata dir
            if not os.path.exists(os.path.join(output_code_metadata_dir, "part"+str(idx))):
                os.makedirs(os.path.join(output_code_metadata_dir, "part"+str(idx)))
            code_metadata = {}
            for file in file_list_in_ref_folder:
                if file.endswith(".v"):
                    module_name = file.split(".v")[0]
                    potential_task_ids = module_to_task_id_map[module_name]
                    #find the one matches the file in the ref folder
                    ref_code_file = os.path.join(existing_folder_to_refer, "part"+str(idx), file)
                    ref_code_str = open(ref_code_file, "r").read()
                    matched_task_id = None
                    for task_id in potential_task_ids:
                        src_code_file = os.path.join(src_code_dir, task_id+".v")
                        src_code_str = open(src_code_file, "r").read()
                        src_code_str_list, module_name_list = part_verilog_module_string(src_code_str)
                        src_code_str = src_code_str_list[0]
                        if src_code_str == ref_code_str:
                            matched_task_id = task_id
                            if not "error" in src_code_metadata[matched_task_id].keys():
                                break
                    if matched_task_id is None:
                        raise Exception("No matched task id found for module {}.".format(module_name))
                    #copy the file to the new folder
                    src_code_file = os.path.join(src_code_dir, matched_task_id+".v")
                    part_verilog_module(src_code_file, os.path.join(output_dir, "part"+str(idx)))
                    #copy the code metadata
                    code_metadata[module_name] = src_code_metadata[matched_task_id]
                    if "error" in src_code_metadata[matched_task_id].keys():
                        raise Exception("This code id {} (module name {}) has error; and should not be selected.".format(matched_task_id, module_name))
            #store the code metadata
            with open(os.path.join(output_code_metadata_dir, "part"+str(idx), output_code_metadata_file), "w") as f:
                json.dump(code_metadata, f, indent=4)
            out_files = os.listdir(os.path.join(output_dir, "part"+str(idx)))
            print(len(out_files))
        
    if generating_from_existing_file_structure:
        #check if two folders contain the same files
        for idx in range(10):
            src_code_dir = "{}/part{}/".format(existing_folder_to_refer, idx)
            target_dir = "{}/part{}/".format(output_dir, idx)
            file_list = os.listdir(src_code_dir)
            out_files = os.listdir(target_dir)
            for file in file_list:
                if file not in out_files:
                    print("{} present in {} but not in {}".format(file, src_code_dir, target_dir))
            for file in out_files:
                if file not in file_list:
                    print("{} present in {} but not in {}".format(file, target_dir, src_code_dir))

    exit()

    #############################################################################################################
    # legacy part and preprocess code; recommend to use the above code
    #############################################################################################################
    src_code_file = "/home/user_name/Edge-MoE/src/moe.cpp"
    # src_code_file = "./test.h"
    part_hls_function(src_code_file, "./Edge-MoE/")
    exit()


    #############################################################################################################
    # legacy prompt testing
    #############################################################################################################

    src_code_file = "code_and_comment_src/raw_src/raw_code_src/sub_module.h"
    csv_code_file = "code_and_comment_src/csv_src/csv_code_src/sub_module.csv"
    csv_comment_file = "code_and_comment_src/csv_src/csv_comment_src/sub_module.csv"
    convert_raw_src_code_to_csv(src_code_file, csv_code_file, csv_comment_file)
    df_code = pd.read_csv(csv_code_file)
    df_comment = pd.read_csv(csv_comment_file)

    while True:
        line_id = int(input("Please enter the line number: "))
        if line_id == -1:
            break
        prompt = form_commenting_prompt(df_code, df_comment, line_id)
        print(prompt)

    






