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



if __name__ == "__main__":
    #clean old documented dir 
    total_part = 11
    documented_part = 10
    old_doced_ckpts_dir = "/home/user_name/DAC_2024/ckpts"
    old_dataset_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_renamed"
    new_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content/"
    src_code_metadata_file = "/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/verilogeval_datagen/process_data/module_inst.json"
    module_to_task_id_map_file = "/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/verilogeval_datagen/process_data/module_name_to_task_id_mapping.json"

    ######### making sure raw_src/raw_code_src assets is in sync with the partitioned dataset #########
    for part_num in range(documented_part):
        old_doced_dir = os.path.join(old_doced_ckpts_dir, "test_10_30_{}_complete".format(part_num))
        parted_dataset_dir = os.path.join(old_dataset_dir, "part{}".format(part_num))
        #remove the parted dataset dir
        shutil.rmtree(parted_dataset_dir)
        print("rm -rf {}".format(parted_dataset_dir))
        #copy the raw_src dir from the old doced dir to the parted dataset dir
        raw_src_dir = "code_and_comment_src/raw_src/raw_code_src"
        shutil.copytree(os.path.join(old_doced_dir, "assets/verilog", raw_src_dir), parted_dataset_dir)
        print("cp -r {} {}".format(os.path.join(old_doced_dir, "assets/verilog", raw_src_dir), parted_dataset_dir))



    ############################# form diff list for partioned dataset #############################
    #form diff list
    #dataset diff list
    #convert old dir to lib_list_old
    lib_list_old = []
    for part_num in range(total_part):
        lib_list_old.append({})
        file_list = os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num)))
        for file in tqdm(file_list, total=len(file_list), desc="Convert old dir to lib_list_old"):
            if file.endswith(".v"):
                src_code_file = os.path.join(old_dataset_dir, "part{}".format(part_num), file)
                code_content = open(src_code_file, "r").read()
                output_str_list, module_name_list = part_verilog_module_string(code_content)
                if len(module_name_list) > 1:
                    raise Exception("More than one module in file {}".format(file))
                module_name_parsed = module_name_list[0]
                module_name_from_file = file.split(".v")[0]
                if module_name_parsed != module_name_from_file:
                    raise Exception("Module name parsed {} does not match module name from file {}".format(module_name_parsed, module_name_from_file))
                for output_str, module_name in zip(output_str_list, module_name_list):
                    #intentionally set to None; as this task id should not be used
                    lib_list_old[-1][module_name] = None


    #convert new dir to lib_list_new
    file_list = os.listdir(new_dir)
    lib_list_new = {}
    for file in tqdm(file_list, total=len(file_list), desc="Convert new dir to lib_list_new"):
        if file.endswith(".v"):
            src_code_file = os.path.join(new_dir, file)
            code_content = open(src_code_file, "r").read()
            output_str_list, module_name_list = part_verilog_module_string(code_content)
            task_id = file.split(".v")[0]
            for output_str, module_name in zip(output_str_list, module_name_list):
                if module_name not in lib_list_new.keys():
                    lib_list_new[module_name] = [task_id]
                else:
                    lib_list_new[module_name].append(task_id)

    #form diff list
    files_to_remove_from_old_dir = []
    files_to_add_to_old_dir = []
    files_common_in_old_and_new_dir = []
    #stuff need to be re-documented
    impact_list = []  

    #remove files from old dir that are not in new dir
    for part_num in range(total_part):
        files_to_remove_from_old_dir.append({})
        for file in tqdm(lib_list_old[part_num].keys(), total=len(lib_list_old[part_num].keys()), desc="Diff list for files from old dir that are not in new dir"):
            if file not in lib_list_new.keys():
                files_to_remove_from_old_dir[-1][file] = None

    merged_lib_list_old_keys = []
    for part_num in range(total_part):
        merged_lib_list_old_keys.extend(lib_list_old[part_num].keys())
    merged_lib_list_old_keys = list(set(merged_lib_list_old_keys))
    #add files to old dir that are not in old dir
    for file in tqdm(lib_list_new.keys(), total=len(lib_list_new.keys()), desc="Diff list for files from new dir that are not in old dir"):
        if file not in merged_lib_list_old_keys:
            files_to_add_to_old_dir.append(file)
    
    #find files that are common in old and new dir
    for part_num in range(total_part):
        old_length = len(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num))))
        files_common_in_old_and_new_dir.append({})
        impact_list.append({})
        for file in tqdm(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num))), total=len(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num)))), desc="Diff list for files that are common in old and new dir"):
            code_name = file.split(".v")[0]
            if code_name in lib_list_new.keys():
                files_common_in_old_and_new_dir[-1][code_name] = None
                #need to find which code, as some may have the same module name
                old_code = open(os.path.join(old_dataset_dir, "part{}".format(part_num), code_name+".v"), "r").read().replace("\n", "").replace(" ", "").replace("\t", "")
                for task_id in lib_list_new[code_name]:
                    new_code = open(os.path.join(new_dir, task_id+".v"), "r").read().replace("\n", "").replace(" ", "").replace("\t", "")
                    if old_code == new_code:
                        files_common_in_old_and_new_dir[-1][code_name] = task_id
                        break
                if files_common_in_old_and_new_dir[-1][code_name] is None:
                    # print("No matching code found for file {}".format(code_name))
                    #copy task_id.v to old_dataset_dir
                    #randomly pick one if repitition occurs
                    files_common_in_old_and_new_dir[-1][code_name] = lib_list_new[code_name][0]
                    shutil.copy(os.path.join(new_dir, lib_list_new[code_name][0]+".v"), os.path.join(old_dataset_dir, "part{}".format(part_num), code_name+".v"))
                    impact_list[-1][code_name] = lib_list_new[code_name][0]
        new_lengh = len(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num))))
        assert old_length == new_lengh, "old_length: {}, new_lengh: {}".format(old_length, new_lengh)


    for part_num , impact_part in enumerate(impact_list):
        print("impact_part {}: {}".format(part_num, len(impact_part)))
        if len(impact_list[part_num]) > 0 and len(impact_list[part_num]) < 10:
            print(impact_list[part_num])

    #shared lib diff list
    # old_shared_lib_dir = "/home/user_name/DAC_2024/data_crunch/old_dataset_bkup/ckpt3_user_name_valid_content_shared_lib"
    old_shared_lib_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib"
    new_shared_lib_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_shared_lib"
    shared_lib_list_old = os.listdir(old_shared_lib_dir)
    shared_lib_list_new = os.listdir(new_shared_lib_dir)

    #form diff list
    shared_lib_files_to_remove_from_old_dir = []
    shared_lib_files_to_add_to_old_dir = []

    #remove files from old dir that are not in new dir
    for file in tqdm(shared_lib_list_old, total=len(shared_lib_list_old), desc="Diff list for shared lib files from old dir that are not in new dir"):
        if file not in shared_lib_list_new:
            shared_lib_files_to_remove_from_old_dir.append(file)
    #add files to old dir that are not in old dir
    for file in tqdm(shared_lib_list_new, total=len(shared_lib_list_new), desc="Diff list for shared lib files from new dir that are not in old dir"):
        if file not in shared_lib_list_old:
            shared_lib_files_to_add_to_old_dir.append(file)


    

    #add the files to the old dataset
    #evenly distribute the new files to the old dataset
    for idx, new_file in enumerate(tqdm(files_to_add_to_old_dir, total=len(files_to_add_to_old_dir), desc="Evenly distribute the new files to the old dataset")):
        #find the part with the least number of files
        #randomly pick one if repitition occurs
        fname = lib_list_new[new_file][0]+".v"
        part_num = idx % total_part
        #copy the file to the part
        shutil.copy(os.path.join(new_dir, fname), os.path.join(old_dataset_dir, "part{}".format(part_num), new_file+".v"))

    

    #remove the files from the old dataset
    for part_num in range(total_part):
        for idx, old_file in enumerate(tqdm(files_to_remove_from_old_dir[part_num], total=len(files_to_remove_from_old_dir[part_num]), desc="Remove the files from the old dataset")):
            #find the part with the least number of files
            fname = old_file+".v"
            if os.path.exists(os.path.join(old_dataset_dir, "part{}".format(part_num), fname)):
                os.remove(os.path.join(old_dataset_dir, "part{}".format(part_num), fname))
                print("removed {}".format(fname))

    #refresh the files in the old dataset
    for part_num in range(total_part):
        for idx, common_file in enumerate(tqdm(files_common_in_old_and_new_dir[part_num], total=len(files_common_in_old_and_new_dir[part_num]), desc="Refresh the common files in the old dataset")):
            #find the part with the least number of files
            fname = common_file+".v"
            if os.path.exists(os.path.join(old_dataset_dir, "part{}".format(part_num), fname)):
                os.remove(os.path.join(old_dataset_dir, "part{}".format(part_num), fname))
                # print("removed {}".format(fname))
                #copy the file to the part
                shutil.copy(os.path.join(new_dir, files_common_in_old_and_new_dir[part_num][common_file]+".v"), os.path.join(old_dataset_dir, "part{}".format(part_num), common_file+".v"))
                #preprocess os.path.join(old_dataset_dir, "part{}".format(part_num), common_file+".v")
                preprocess(os.path.join(old_dataset_dir, "part{}".format(part_num), common_file+".v"))

    module_to_task_id_map = json.load(open(module_to_task_id_map_file, "r"))  
    metadata = json.load(open(src_code_metadata_file, "r"))
    #generate metadata dir for the old dataset
    old_metadata_dir = "/home/user_name/DAC_2024/ckpt3_user_name_valid_content_code_metadata/"
    for part_num in range(total_part):
        if os.path.exists(os.path.join(old_metadata_dir, "part{}".format(part_num))):
            shutil.rmtree(os.path.join(old_metadata_dir, "part{}".format(part_num)))
        os.makedirs(os.path.join(old_metadata_dir, "part{}".format(part_num)))
        old_dir_metadata = {}
        for file in tqdm(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num))), total=len(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_num)))), desc="Generate metadata dir for the old dataset"):
            if file.endswith(".v"):
                code_name = file.split(".v")[0]
                matched_task_id = None
                if code_name in files_common_in_old_and_new_dir[part_num]:
                    matched_task_id = files_common_in_old_and_new_dir[part_num][code_name]
                else:
                    task_ids = module_to_task_id_map[code_name]
                    for task_id in task_ids:
                        if "error" not in metadata[task_id].keys():
                            matched_task_id = task_id
                            break
                if matched_task_id is None:
                    raise Exception("No valid code {} found in metadata".format(code_name))
                old_dir_metadata[code_name] = metadata[matched_task_id]
        json.dump(old_dir_metadata, open(os.path.join(old_metadata_dir, "part{}".format(part_num), "codes.json"), "w"), indent=4)


    for part_num in range(total_part):
        for file in tqdm(lib_list_old[part_num].keys(), total=len(lib_list_old[part_num].keys()), desc="Stuff need to be re-documented"):
            dependency_list = []
            task_ids = module_to_task_id_map[file]
            for task_id in task_ids:
                if "module_inst_list" in metadata[task_id].keys():
                    dependency_list.extend(metadata[task_id]["module_inst_list"])
            dependency_list = list(set(dependency_list))
            for dependency in dependency_list:
                if dependency+".v" in shared_lib_files_to_remove_from_old_dir or dependency+".v" in shared_lib_files_to_add_to_old_dir:
                    if dependency not in impact_list[part_num]:
                        if dependency in files_common_in_old_and_new_dir[part_num]:    
                            impact_list[part_num][dependency] = files_common_in_old_and_new_dir[part_num][dependency]
                        else:
                            impact_list[part_num][dependency] = None
                    break
    


    redoc_files = []

    for part_num in range(total_part):
        redoc_files.append({})
        for code_name in files_to_remove_from_old_dir[part_num]:
            redoc_files[-1][code_name] = files_to_remove_from_old_dir[part_num][code_name]
        for code_name in impact_list[part_num]:
            if code_name not in redoc_files[-1]:
                redoc_files[-1][code_name] = impact_list[part_num][code_name]
    

    

    ############################# clean assets #############################
        
    for part_to_clean in range(documented_part):
        doced_dataset_dir = "/home/user_name/DAC_2024/ckpts/test_10_30_{}_complete".format(part_to_clean)
        asset_dir = "assets/verilog"
        code_and_commment_src_dir = "code_and_comment_src"
        code_summary_dir = "code_summary"
        csv_src_dir = "csv_src"
        csv_code_src_dir = "csv_code_src"
        csv_comment_src_dir = "csv_comment_src"
        csv_new_comment_src_dir = "csv_new_comment_src"
        csv_pure_gen_comment_src_dir = "csv_pure_gen_comment_src"
        raw_src_dir = "raw_src/raw_code_src"
        documented_code_dir = "documented_code"
        documented_code_src_dir = "documented_code_src"



        #clean the dataset
            #clean assets
                #clean code_and_comment_src
                    #clean code_summary
        removed_count = 0
        for code_summary_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, code_summary_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, code_summary_dir))), desc="Clean code_and_comment_src/code_summary"):
            code_name = code_summary_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, code_summary_dir, code_summary_file))
        print("Removed {} files from code_and_comment_src/code_summary".format(removed_count))
                    #clean csv_src
                        #clean csv_code_src
        removed_count = 0
        for csv_code_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir))), desc="Clean code_and_comment_src/csv_src/csv_code_src"):
            code_name = csv_code_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, csv_code_src_file))
        print("Removed {} files from code_and_comment_src/csv_src/csv_code_src".format(removed_count))
                        #clean csv_comment_src
        removed_count = 0
        for csv_comment_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir))), desc="Clean code_and_comment_src/csv_src/csv_comment_src"):
            code_name = csv_comment_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir, csv_comment_src_file))
        print("Removed {} files from code_and_comment_src/csv_src/csv_comment_src".format(removed_count))
                        #clean csv_new_comment_src
        removed_count = 0
        for csv_new_comment_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir))), desc="Clean code_and_comment_src/csv_src/csv_new_comment_src"):
            code_name = csv_new_comment_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir, csv_new_comment_src_file))
        print("Removed {} files from code_and_comment_src/csv_src/csv_new_comment_src".format(removed_count))
                        #clean csv_pure_gen_comment_src
        removed_count = 0
        for csv_pure_gen_comment_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_pure_gen_comment_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_pure_gen_comment_src_dir))), desc="Clean code_and_comment_src/csv_src/csv_pure_gen_comment_src"):
            code_name = csv_pure_gen_comment_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_pure_gen_comment_src_dir, csv_pure_gen_comment_src_file))
        print("Removed {} files from code_and_comment_src/csv_src/csv_pure_gen_comment_src".format(removed_count))
                    #clean documented_code_src
        removed_count = 0
        for documented_code_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir))), desc="Clean code_and_comment_src/documented_code_src"):
            code_name = documented_code_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir, documented_code_src_file))
            else:
                #remerge the code and comment
                csv_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, code_name+".csv")
                csv_new_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir, code_name+".csv")
                commented_code_str = merge_code_and_comment(csv_code_src_file, csv_new_comment_src_file)
                with open(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir, code_name+".v"), "w") as f:
                    f.write(commented_code_str)
        print("Removed {} files from code_and_comment_src/documented_code_src".format(removed_count))
                    #clean raw_src
        removed_count = 0
        for raw_src_file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir))), desc="Clean code_and_comment_src/raw_src/raw_code_src"):
            code_name = raw_src_file.split(".")[0]
            if code_name in redoc_files[part_to_clean]:
                removed_count += 1
                os.remove(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir, raw_src_file))
                #clean documented_list
        print("Removed {} files from code_and_comment_src/raw_src/raw_code_src".format(removed_count))
        lines = open(os.path.join(doced_dataset_dir, asset_dir, "documented_list.txt"), "r").readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line.split(".")[0] not in redoc_files[part_to_clean]]
        with open(os.path.join(doced_dataset_dir, asset_dir, "documented_list.txt"), "w") as f:
            for line in lines:
                f.write(line+"\n")    

        #skipped for now only do addition for shared lib
        #clean code_vec_store
            #clean bookkeeping
            #clean vectorembedding
        #clean documented_code
        #remove directories in the documented_code dir that are in the redoc_files[part_to_clean]
        removed_count = 0
        for dir in tqdm(os.listdir(os.path.join(doced_dataset_dir, documented_code_dir)), total=len(os.listdir(os.path.join(doced_dataset_dir, documented_code_dir))), desc="Clean documented_code"):
            if dir in redoc_files[part_to_clean]:
                removed_count += 1
                shutil.rmtree(os.path.join(doced_dataset_dir, documented_code_dir, dir))
            else:
                #remerge the code and comment
                csv_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, dir+".csv")
                csv_new_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir, dir+".csv")
                commented_code_str = merge_code_and_comment(csv_code_src_file, csv_new_comment_src_file)
                with open(os.path.join(doced_dataset_dir, documented_code_dir, dir, dir+".v"), "w") as f:
                    f.write(commented_code_str)
        print("Removed {} directories from documented_code".format(removed_count))



        #refresh raw_src/raw_code_src with stuff from partitioned dataset
        raw_src_files = os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir))
        for file in tqdm(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_to_clean))), total=len(os.listdir(os.path.join(old_dataset_dir, "part{}".format(part_to_clean)))), desc="Refresh raw_src/raw_code_src with stuff from partitioned dataset"):
            if file.split(".")[0] in impact_list[part_to_clean]:
                continue
            if file not in raw_src_files:
                raise Exception("File {} not in raw_src/raw_code_src".format(file))
            shutil.copy(os.path.join(old_dataset_dir, "part{}".format(part_to_clean), file), os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir))
            #create csv assets
            convert_raw_src_code_to_csv(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir, file), 
                                        os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, file.split(".")[0]+".csv"),
                                        os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir, file.split(".")[0]+".csv"),
                                        discard_original_comment=False)


        removed_count = 0
        #checking if csvs in csv_code_src, csv_comment_src, csv_new_comment_src, csv_pure_gen_comment_src have the same number of lines
        for file in tqdm(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir)), 
                         total=len(os.listdir(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir))), 
                         desc="Checking if csvs in csv_code_src, csv_comment_src, csv_new_comment_src, csv_pure_gen_comment_src have the same number of lines"):
            code_name = file.split(".")[0]
            csv_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, file.split(".")[0]+".csv")
            csv_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir, file.split(".")[0]+".csv")
            csv_new_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir, file.split(".")[0]+".csv")
            csv_pure_gen_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_pure_gen_comment_src_dir, file.split(".")[0]+".csv")
            documentedc_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir, file.split(".")[0]+".v")
            documented_code_file = os.path.join(doced_dataset_dir, documented_code_dir, file.split(".")[0]+".v")

            df_comment = pd.read_csv(csv_comment_src_file)
            df_code = pd.read_csv(csv_code_src_file)
            df_comment["line_number"] = df_comment["line_number"].astype(int)
            df_code["line_number"] = df_code["line_number"].astype(int)
            df_comment = df_comment.sort_values("line_number")
            df_code = df_code.sort_values("line_number")
            code_lines = df_code["content"].tolist()
            comment_lines = df_comment["content"].tolist()

            df_new_comment = pd.read_csv(csv_new_comment_src_file)
            df_new_comment["line_number"] = df_new_comment["line_number"].astype(int)
            df_new_comment = df_new_comment.sort_values("line_number")
            new_comment_lines = df_new_comment["content"].tolist()

            df_pure_gen_comment = pd.read_csv(csv_pure_gen_comment_src_file)
            pure_gen_comment_lines = df_pure_gen_comment["content"].tolist()

            if len(code_lines) != len(comment_lines) or len(code_lines) != len(new_comment_lines):
                # print("Code lines and comment lines do not match for file {}".format(file))
                lines_to_be_corrected = []
                #read the code lines from commented code
                commented_code_lines = open(documentedc_code_src_file, "r").readlines()
                lines_to_be_corrected = [line for line in commented_code_lines if not line.strip().startswith("//")]
                #sub \s+ with " " and \t+ with " "
                lines_to_be_corrected = [re.sub(r"\s+", " ", line) for line in lines_to_be_corrected]
                lines_to_be_corrected = [re.sub(r"\t+", " ", line) for line in lines_to_be_corrected]
                code_lines = [re.sub(r"\s+", " ", line) for line in code_lines]
                code_lines = [re.sub(r"\t+", " ", line) for line in code_lines]
                #find where the misalignment is
                ite = 0
                while True:
                    misalignment_idx = None
                    for i in range(len(code_lines)):
                        if re.sub(r"\s+", "", code_lines[i].strip()) != re.sub(r"\s+", "", lines_to_be_corrected[i].strip()):
                            if re.sub(r"\s+", "", code_lines[i].strip()) in re.sub(r"\s+", "", lines_to_be_corrected[i].strip()):
                                misalignment_idx = i
                                if "LUKE_PLACEHOLD" in file:
                                    print(code_lines[i].strip())
                                    print("\n\n==")
                                    print(lines_to_be_corrected[i].strip())
                                    print("\n\n==")
                                    print(misalignment_idx)
                                break
                            else:
                                code_lines[i] = lines_to_be_corrected[i]
                    if misalignment_idx is None or ite > 1000:
                        break
                    sep_list = [code_lines[misalignment_idx], lines_to_be_corrected[misalignment_idx].replace(code_lines[misalignment_idx].strip(), "", 1)]
                    comment_padding_len = len(sep_list)-1
                    lines_to_be_corrected = lines_to_be_corrected[:misalignment_idx] + sep_list + lines_to_be_corrected[misalignment_idx+1:]
                    #pad comment
                    new_comment_lines = new_comment_lines[:misalignment_idx+1] + ["no_comment"]*comment_padding_len + new_comment_lines[misalignment_idx+1:]
                    if "LUKE_PLACEHOLD" in file:
                        print("misalignment_idx: {}".format(misalignment_idx))
                        print(comment_padding_len)
                        print(len(new_comment_lines))
                        print(len(code_lines))
                        input()
                    ite += 1

                #check if padded comment lines are aligned with code lines
                try:
                    assert  len(new_comment_lines) == len(code_lines)
                except Exception as e:
                    print("Beyond repair")
                    print("Code lines and comment lines do not match for file {}".format(file))
                    # print("comment lines:")
                    # print("\n\n".join(new_comment_lines))
                    # print("=="*20)
                    # print("code lines:")
                    # print("\n\n".join(code_lines))
                    # print("=="*20)
                    # print("lines to be corrected:")
                    # print("\n\n".join(lines_to_be_corrected))
                    # print("=="*20)
                    # print("Code lines and comment lines are not aligned")
                    # exit(1)
                    removed_count += 1
                    
                    #remove code summary
                    code_summary_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, code_summary_dir, file.split(".")[0]+".txt")
                    os.remove(code_summary_file)
                    #remove csvs
                    csv_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_code_src_dir, file.split(".")[0]+".csv")
                    csv_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_comment_src_dir, file.split(".")[0]+".csv")
                    csv_new_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_new_comment_src_dir, file.split(".")[0]+".csv")
                    csv_pure_gen_comment_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, csv_src_dir, csv_pure_gen_comment_src_dir, file.split(".")[0]+".csv")
                    os.remove(csv_code_src_file)
                    os.remove(csv_comment_src_file)
                    os.remove(csv_new_comment_src_file)
                    os.remove(csv_pure_gen_comment_src_file)
                    #remove documented_code_src
                    documentedc_code_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, documented_code_src_dir, file.split(".")[0]+".v")
                    os.remove(documentedc_code_src_file)
                    #remove raw_src
                    raw_src_file = os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir, file.split(".")[0]+".v")
                    os.remove(raw_src_file)
                    #remove from documented_list
                    lines = open(os.path.join(doced_dataset_dir, asset_dir, "documented_list.txt"), "r").readlines()
                    lines = [line.strip() for line in lines]
                    lines = [line for line in lines if line.split(".")[0] != file.split(".")[0]]
                    with open(os.path.join(doced_dataset_dir, asset_dir, "documented_list.txt"), "w") as f:
                        for line in lines:
                            f.write(line+"\n")
                    #remove directories in the documented_code dir that are in the redoc_files[part_to_clean]
                    shutil.rmtree(os.path.join(doced_dataset_dir, documented_code_dir, file.split(".")[0]))
                    continue


                #save the padded comment lines to original csv
                #create a new csv for the padded comment
                df_comment = pd.DataFrame()
                df_comment["line_number"] = list(range(len(new_comment_lines)))
                df_comment["content"] = new_comment_lines
                df_comment.to_csv(csv_new_comment_src_file, index=False)
                #save the code lines to original csv
                df_code = pd.DataFrame()
                df_code["line_number"] = list(range(len(code_lines)))
                df_code["content"] = code_lines
                df_code.to_csv(csv_code_src_file, index=False)
                #save code lines to raw_src
                with open(os.path.join(doced_dataset_dir, asset_dir, code_and_commment_src_dir, raw_src_dir, file), "w") as f:
                    f.write("\n".join(code_lines))  
                #save commented code 
                commented_code_str = merge_code_and_comment(csv_code_src_file, csv_new_comment_src_file)
                with open(documentedc_code_src_file, "w") as f:
                    f.write(commented_code_str)
                with open(documented_code_file, "w") as f:
                    f.write(commented_code_str)
                #fix pure gen comment
                pure_gen_line_number = []
                for pure_gen_line in pure_gen_comment_lines:
                    found_comment = False
                    for comment_idx, comment_line in enumerate(new_comment_lines):
                        if pure_gen_line in comment_line:
                            pure_gen_line_number.append(comment_idx)
                            found_comment = True
                            break
                    if found_comment is False:
                        raise Exception("Pure gen comment {} not found in new comment lines".format(pure_gen_line))
                df_pure_gen_comment = pd.DataFrame()
                df_pure_gen_comment["line_number"] = pure_gen_line_number
                df_pure_gen_comment["content"] = pure_gen_comment_lines
                df_pure_gen_comment.to_csv(csv_pure_gen_comment_src_file, index=False)

        print("Removed {} files from code_and_comment_src/csv_src/csv_code_src due to comment and code misalignment".format(removed_count))        
        # #clean final metadata
        # metadata_dir = "/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/dataset_metadata/part{}".format(part_to_clean)
        # for metadata_file in os.listdir(metadata_dir):
        #     if metadata_file.endswith(".json"):
        #         metadata = json.load(open(os.path.join(metadata_dir, metadata_file), "r"))
        #         new_metadata = {}
        #         for key in metadata.keys():
        #             if key.split(".")[0] not in redoc_files[part_to_clean]:
        #                 new_metadata[key] = metadata[key]
        #         json.dump(new_metadata, open(os.path.join(metadata_dir, metadata_file), "w"), indent=4)
        # bookkeeping_dir = "bookkeep/codes.json"
        # bookkeeping = json.load(open(os.path.join(doced_dataset_dir, asset_dir, bookkeeping_dir), "r"))
        # new_bookkeeping = {}
        # for key in bookkeeping.keys():
        #     if key.split(".")[0] not in redoc_files[part_to_clean]:
        #         new_bookkeeping[key] = bookkeeping[key]
        # json.dump(new_bookkeeping, open(os.path.join(doced_dataset_dir, asset_dir, bookkeeping_dir), "w"), indent=4)
        