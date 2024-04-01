import os
import sys
import shutil
import pandas as pd
import json
import jsonlines


if __name__ == "__main__":
    task_ids = [
         "shift18",
         "rule110",
         "lemmings1",
         "fsm3onehot"
    ]
    

    example_code_strings_name = "example_code_strings_detailed_instructions.json"
    eval_file = "../../verilog_eval/data/VerilogEval_Machine.jsonl"
    eval_dict = {}
    with jsonlines.open(eval_file) as reader:
        for obj in reader:
            eval_dict[obj["task_id"]] = {}
            eval_dict[obj["task_id"]]["code"] = obj["prompt"] + obj["canonical_solution"]

    #store in a json string 
    example_code_strings = {}
    for task_id in task_ids:
        example_code_strings[task_id] = eval_dict[task_id]["code"]
    #store in a json file
    with open(example_code_strings_name, "w") as f:
        json.dump(example_code_strings, f, indent=4)
        
