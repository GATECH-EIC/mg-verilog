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
from embedding_lookup_utils import *
from utils import *
from completion_handler import *
from code_preprocesser import *


class CodeRepoDocumentor:
    def __init__(self, code_dir, store_src_code_dir, 
                 csv_code_dir, csv_comment_dir, csv_new_comment_dir, 
                 csv_pure_gen_comment_dir, code_summary_dir, documented_code_dir,
                 code_metadata_file = None,
                 code_suffix =[".v", ".sv", ".vh"], language="verilog",
                 discard_original_comment = False,
                 code_lib_path= "./lib", code_vec_store = "./vector_store/",
                 skip_rag_db = False, skip_supplement_summary = False,
                 cb = None):

        #raw code preprocessing
        self.code_dir = code_dir
        self.code_suffix = code_suffix
        self.language = language
        self.store_src_code_dir = store_src_code_dir
        self.csv_code_dir = csv_code_dir
        self.csv_comment_dir = csv_comment_dir
        self.csv_new_comment_dir = csv_new_comment_dir
        self.csv_pure_gen_comment_dir = csv_pure_gen_comment_dir
        self.code_summary_dir = code_summary_dir
        self.documented_code_dir = documented_code_dir
        self.discard_original_comment = discard_original_comment
        self.code_vec_store = code_vec_store
        self.code_Lib_path = code_lib_path
        self.cb = cb
        self.skip_rag_db = skip_rag_db
        self.skip_supplement_summary = skip_supplement_summary
        if code_metadata_file is not None:
            self.code_metadata_file = code_metadata_file
            self.code_metadata = json.load(open(self.code_metadata_file, "r"))

        self.code_preprocesser = CodePreprocesser(code_dir, store_src_code_dir, 
                                            csv_code_dir, csv_comment_dir, 
                                            csv_new_comment_dir, csv_pure_gen_comment_dir, 
                                            code_summary_dir, documented_code_dir,
                                            code_suffix=code_suffix, discard_original_comment=discard_original_comment)
        self.documented_list = []
        self.documented_list_file = os.path.join(os.environ.get("ASSET_DIR"), os.environ.get("TARGET_LANG"), "documented_list.txt")
        if os.path.exists(self.documented_list_file):
            #ask if the user wants to remove the documented list
            print("Do you want to remove the documented list? (y/n)")
            answer = input()
            if answer == "y":
                os.remove(self.documented_list_file)
                print("Documented list removed")
            else:
                with open(self.documented_list_file, "r") as f:
                    self.documented_list = f.readlines()
                self.documented_list = [x.strip() for x in self.documented_list]
        
        
        #context embedding
        self.embedding_fields = ["Filename", "File type", "Summary", "Text", "Line_id"]
        self.system_embedder = EmbedTool0(self.embedding_fields, 
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR")),
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_EMBEDDING_DIR")),
                                    "system_context_embedding.csv")

        #code documentor
        self.documentor = Chatbot(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR"), 
                            "context.fixed_features.txt"),
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("CONVERSE_DIR")),
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR"), 
                            "context.converse_samples.txt"),
                            code_suffix=self.code_suffix,
                            language=self.language,
                            code_lib_path=self.code_Lib_path,
                            code_vec_store=self.code_vec_store,
                            skip_supplement_summary = self.skip_supplement_summary,
                            cb = self.cb
                            )

    
    def create_embedding(self):
        self.system_embedder.create_raw_system_context()
        self.system_embedder.create_embedding()
        self.system_embedder.load_embedding()
        if not self.skip_rag_db:
            self.documentor.init_code_retrival()

    def code_preprocess(self, skip_preprocess=False):
        self.code_preprocesser.raw_code_copy(self.code_dir, self.store_src_code_dir, skip_preprocess=skip_preprocess)
        if not skip_preprocess:
            self.code_preprocesser.pre_process_routines(self.store_src_code_dir, 
                                                        discard_original_comment=self.discard_original_comment, 
                                                        rtl=(".v" in self.code_suffix or ".sv" in self.code_suffix or ".vh" in self.code_suffix))
            self.code_preprocesser.create_code_assets()
        else:
            pass

    def document_repo(self):
        self.documentor.system_context_embedding = self.system_embedder.df_embed
        for code_src in self.code_preprocesser.code_files:
            start_time = time.time()
            if code_src not in self.documented_list:
                print("Documenting {}".format(code_src))
                #clear the memory of the documentor 
                self.documentor.line_by_line_comment_converse_chain.memory.clear()
                self.documentor.line_by_line_comment_converse_chain.memory_buffer = []
                csv_code_file = os.path.join(self.code_preprocesser.csv_code_dir, code_src.split(".")[0] + ".csv")
                csv_comment_file = os.path.join(self.code_preprocesser.csv_comment_dir, code_src.split(".")[0] + ".csv")
                csv_new_comment_file = os.path.join(self.code_preprocesser.csv_new_comment_dir, code_src.split(".")[0] + ".csv")
                csv_pure_gen_comment_file = os.path.join(self.code_preprocesser.csv_pure_gen_comment_dir, code_src.split(".")[0] + ".csv")
                code_summary_file = os.path.join(self.code_preprocesser.code_summary_dir, code_src.split(".")[0] + ".txt")

                #check the # of lines of code
                with open(os.path.join(self.code_preprocesser.store_src_code_dir, code_src), "r") as f:
                    lines = f.readlines()
                if len(lines) > 200:
                    print("Skip {} because it has too many lines of code".format(code_src))
                    continue

                dependent_funcs = self.code_metadata[code_src.split(".")[0]]["module_inst_list"]
                self.documentor.comment_a_code_file(csv_code_file, csv_comment_file, csv_new_comment_file, csv_pure_gen_comment_file, dependent_funcs=dependent_funcs)
                
                new_code_string = merge_code_and_comment(csv_code_file, csv_new_comment_file)
                with open(os.path.join(self.code_preprocesser.documented_code_dir, code_src), "w") as f:
                    f.write(new_code_string)

                self.documentor.summarize_code_blocks(csv_code_file, csv_new_comment_file, code_summary_file)
                # bot.reverse_code_gen(csv_pure_gen_comment_file, code_summary_file)

                self.documented_list.append(code_src)
                with open(self.documented_list_file, "w") as f:
                    f.write("\n".join(self.documented_list))
            end_time = time.time()
            print("Time left to finish this repo: {}".format((end_time - start_time) * (len(self.code_preprocesser.code_files) - self.code_preprocesser.code_files.index(code_src))))
    def package_documented_code(self, package_dir):
        #create the package dir
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
        for code_src in self.documented_list:
            #create a subdirectory for each of the documented code
            code_src = code_src.strip()
            code_src_dir = os.path.join(package_dir, code_src.split(".")[0])
            if not os.path.exists(code_src_dir):
                os.makedirs(code_src_dir)
            shutil.copy(os.path.join(self.code_preprocesser.documented_code_dir, code_src), os.path.join(package_dir, code_src.split(".")[0], code_src))
            shutil.copy(os.path.join(self.code_preprocesser.code_summary_dir, code_src.split(".")[0] + ".txt"), os.path.join(package_dir, code_src.split(".")[0], code_src.split(".")[0] + ".txt"))
    #TODO: add a function to convert the documented code to original raw code

if __name__ == "__main__":
    #NOTE: run utils.py first to partition the code first
    code_dir = "./test_repo/"
    code_lib_path =  "./test_repo/"
    code_vec_store = "../code_vec_store/DNNBuilder/"
    language = os.environ.get("TARGET_LANG")
    if os.environ.get("TARGET_LANG") == "verilog":
        code_suffix = [".v", ".sv", ".vh"]
    elif os.environ.get("TARGET_LANG") == "xilinx_hls":
        code_suffix = [".c", ".cpp", ".h", ".hpp"]
    store_src_code_dir = os.environ.get("STORE_SRC_CODE_DIR")
    csv_code_dir = os.environ.get("CSV_CODE_DIR")
    csv_comment_dir = os.environ.get("CSV_COMMENT_DIR")
    csv_new_comment_dir = os.environ.get("CSV_NEW_COMMENT_DIR")
    csv_pure_gen_comment_dir = os.environ.get("CSV_PURE_GEN_COMMENT_DIR")
    code_summary_dir = os.environ.get("CODE_SUMMARY_DIR")
    documented_code_dir = os.environ.get("DOCUMENTED_CODE_DIR")


    with get_openai_callback() as cb:
        #This switch will discard 1. the comments in the raw code copy and 2. the comments will be converted to the raw code csv 
        discard_original_comment = True
        
        code_repo_documentor = CodeRepoDocumentor(code_dir, store_src_code_dir,
                                                    csv_code_dir, csv_comment_dir, csv_new_comment_dir, 
                                                    csv_pure_gen_comment_dir, code_summary_dir, documented_code_dir,
                                                    code_suffix=code_suffix, language=language,
                                                    discard_original_comment=discard_original_comment,
                                                    code_lib_path=code_lib_path, code_vec_store=code_vec_store,
                                                    cb = cb)
        code_repo_documentor.create_embedding()
        code_repo_documentor.code_preprocess()
        code_repo_documentor.document_repo()
        code_repo_documentor.package_documented_code("./documented_code")