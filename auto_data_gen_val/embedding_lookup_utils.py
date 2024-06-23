import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../verilog_eval/verilog_eval"))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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


#langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore, LocalFileStore
import uuid
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document



from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma, FAISS # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the code
from langchain.llms import OpenAIChat # the LLM model we'll use (CHatGPT)
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback #with get_openai_callback() as cb:

from langchain.retrievers.multi_vector import MultiVectorRetriever


from chain_utils import Global_summary, gen_block_summary_chain, func_name_lookup_chain, VerilogEval, detail_steps_chain, openai_chat, SimpleConverseChain


from langchain.agents import initialize_agent, Tool
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor


def process_block_summary(fname):
    #load the block summary from json
    with open(fname, "r") as f:
        block_summaries = json.load(f)
    for summary_idx, code in enumerate(block_summaries):
        summary_text_lst = block_summaries[code]
        for i, summary_text in enumerate(summary_text_lst):
            #split by the first occurence of ":\n" and take the second part
            spllited = summary_text.split(":\n", 1)
            if len(spllited) > 1:
                summary_text_lst[i] = spllited[1]
            else:
                summary_text_lst[i] = spllited[0]
            block_summaries[code] = summary_text_lst
    return block_summaries


def merge_dict_lst(dict_lst):
    #merge the dict_lst into one dict
    merged_dict = {}
    for tmp_dict in dict_lst:
        for code in tmp_dict:
            if code not in merged_dict:
                merged_dict[code] = tmp_dict[code]
            else:
                merged_dict[code].extend(tmp_dict[code])
    return merged_dict


def evaluate_single_module(task_id, completion, eval_file):
    tmp_gen_file = task_id + ".jsonl"
    tmp_prob_file = task_id + "_prob.jsonl"
    problem_dict = None
    #retrieve the problem dict
    with jsonlines.open(eval_file) as reader:
        for obj in reader:
            if obj["task_id"] == task_id:
                problem_dict = obj
                break
    if problem_dict is None:
        raise Exception("Problem dict not found")
    #write tmp_prob_file
    with jsonlines.open(tmp_prob_file, mode='w') as writer:
        writer.write(problem_dict)
  
    module_dict = {}
    module_dict["task_id"] = task_id
    #assume no header in completion
    module_dict["completion"] = completion
    #write tmp_gen_file
    with jsonlines.open(tmp_gen_file, mode='w') as writer:
        writer.write(module_dict)
    #evaluate
    res = evaluate_functional_correctness(tmp_gen_file, problem_file=tmp_prob_file, k=[1])
    os.remove(tmp_gen_file)
    os.remove(tmp_prob_file)
    os.remove(task_id+".jsonl_results.jsonl")

    res_dict = res
    if res_dict["pass@1"] == 0:
        return False
    elif res_dict["pass@1"] > 0:
        return True

def validate_global_summary_openai(global_summary, task_id, eval_file, max_trials=1, skip_validation=True):
    system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
    question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
    problem_description = "\n\n {description} \n\n Module header:\n\n {module_header}\n"
    #retrieve the module header
    module_header = None
    with jsonlines.open(eval_file) as reader:
        for obj in reader:
            if obj["task_id"] == task_id:
                module_header = obj["prompt"]
                break
    if module_header is None:
        raise Exception("Module header not found")
    #generate the prompt
    user_prompt = question_prompt + problem_description.format(description=global_summary, module_header=module_header)
    chain = SimpleConverseChain(system_prompt=system_prompt, model="gpt-4-0613", temperature=0.7, max_tokens=512, top_p=0.95, have_memory=False, verbose=False)
    for trial in range(max_trials):
        print("Trial: {}".format(trial))
        completion = chain.chat(user_prompt, system_prompt=system_prompt)
        #otherwise, directly use ./model_eval_qlora/standalone_eval.py on the results file to easy pass rates calculation
        if not skip_validation:
            if evaluate_single_module(task_id, completion, eval_file):
                return True, completion
    return False, completion


class EmbedTool0:
    def __init__(self, fields, system_context_dir, system_context_embedding_dir, system_context_embedding_file):
        # embedding model parameters
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        self.max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
        self.fields = fields # the fields of the csv file
        self.system_context_dir = system_context_dir
        self.system_context_embedding_dir = system_context_embedding_dir
        self.system_context_embedding_file = system_context_embedding_file
        self.df_raw = None
        self.df_embed = None

    def parse_system_context_file(self, fname):
        raw = fname.split(".")
        attrs = {}
        attrs["Filename"] = fname
        attrs["Filetype"] = raw[0]
        attrs["Summary"] = raw[1]
        return attrs

    def create_raw_system_context(self):
        # create a csv file with the fields
        create_empty_csv(self.fields, self.system_context_dir+"/system_context_raw.csv")
        for fname in os.listdir(self.system_context_dir):
            #if not txt file, skip
            if not fname.endswith(".txt"):
                continue
            #if fixed_feature file or converse_samples, skip; always loaded to the system context
            if "fixed_feature" in fname or "converse_samples" in fname:
                continue
            attrs = self.parse_system_context_file(fname)
            #read the file line by line
            with open(os.path.join(self.system_context_dir, fname), "r") as f:
                lines = f.readlines()
            # create an entry for each line
            for line_id, line in enumerate(lines):
                attrs["Text"] = line
                attrs["Line_id"] = str(line_id)
                df = pd.DataFrame(attrs, index=[0])
                df.to_csv(self.system_context_dir+"/system_context_raw.csv", mode="a", header=False, index=False)
        self.df_raw = pd.read_csv(self.system_context_dir+"/system_context_raw.csv")
        

    def create_embedding(self):
        # if embedding dir does not exist, create it
        if not os.path.exists(self.system_context_embedding_dir):
            os.mkdir(self.system_context_embedding_dir)
        encoding = tiktoken.get_encoding(self.embedding_encoding)
        # omit text that are too long to embed
        self.df_raw["n_tokens"] = self.df_raw.Text.apply(lambda x: len(encoding.encode(x)))
        self.df_raw = self.df_raw[self.df_raw.n_tokens <= self.max_tokens]
        # This may take a few minutes
        self.df_raw["embedding"] = self.df_raw.Text.apply(lambda x: get_embedding(x, engine=self.embedding_model))
        # only store the embedding and filename
        self.df_embed = self.df_raw[["Filename", "embedding", "Line_id", "Text"]]
        self.df_embed.to_csv(self.system_context_embedding_dir+"/"+self.system_context_embedding_file, index=False)

    def load_embedding(self):
        self.df_embed = pd.read_csv(self.system_context_embedding_dir+"/"+self.system_context_embedding_file)
        self.df_embed["embedding"] = self.df_embed.embedding.apply(literal_eval).apply(np.array)


    
# search through the contexts for the most similar ones
def search_contexts(embedding_model, df, user_query_str, n=3):
    user_query_embeding = get_embedding(
        user_query_str,
        engine=embedding_model
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, user_query_embeding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .Text.str.replace("\n", " ")
    )

    #format the results to list of strings of Text
    results = list(results)
    return results



#Maintain user defined context
class ContextDataset:
    def __init__(self) -> None:
        pass



#################### Code embedding ####################
# For code definition lookup 
########################################################
#TODO: support multiple code libraries
#TODO: support run-time added code comments
class CodeDataset:
    def __init__(self, code_dir, bookkeeping_dir = "bookkeep/", vectorembedding_dir="embedding_data/", 
                force_refresh=False, code_suffix =[".v", ".sv", ".vh"], language="verilog", cb=None):
        self.code_dir = code_dir
        self.code_suffix = code_suffix
        self.language = language
        self.vectorembedding_dir = vectorembedding_dir
        if not os.path.exists(self.vectorembedding_dir):
            os.makedirs(self.vectorembedding_dir)
        self.bookkeeping_dir = bookkeeping_dir
        if not os.path.exists(self.bookkeeping_dir):
            os.makedirs(self.bookkeeping_dir)
        self.force_refresh = force_refresh

        self.codes = self.get_codes_recursive(code_dir)
        self.codes_inMemory = {}
        for code in self.codes:
            self.codes_inMemory[code] = {}
        self.cb = cb
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
    def get_codes_recursive(self, code_dir):
        codes = {}
        # list all the metadata to be bookkept
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith(tuple(self.code_suffix)):
                    codes[file] = {}
                    codes[file]["path"] = os.path.join(root, file)
                    codes[file]["name"] = file
                    codes[file]["genre"] = ', '.join(root.split('/')[1:])
                    #check if there is a <name>.txt file in the same directory as the code 
                    #if there is, then it is the description of the code
                    if os.path.exists(os.path.join(root, file.split(".")[0] + ".txt")):
                        codes[file]["global_summary_from_line_comments_path"] = os.path.join(root, file.split(".")[0] + ".txt")
                        with open(codes[file]["global_summary_from_line_comments_path"], "r") as f:
                            codes[file]["global_summary_from_line_comments"] = f.read()
                    else:
                        codes[file]["global_summary_from_line_comments_path"] = None
                        codes[file]["global_summary_from_line_comments"] = None
                    codes[file]["block_summary"] = None
                    codes[file]["detailed_steps"] = None
                    codes[file]["module_header"] = None
                    codes[file]["global_summary_high_level"] = None
                    codes[file]["global_summary_detailed"] = None
        if os.path.exists(os.path.join(self.bookkeeping_dir, "codes.json")) and not self.force_refresh:
           with open(self.bookkeeping_dir + "codes.json", "r") as f:
                old_codes = json.load(f)
                for code in codes:
                    if code in old_codes:
                        for key in old_codes[code]:
                            codes[code][key] = old_codes[code][key]
                    # populate the doc stores status in newly added codes
                    # currently maintain the following for doc stores:
                    # 1. related_lit_docs_per_code_large
                    # 2. related_lit_docs_per_code_small
                    # 3. related_lit_docs_global
                    # 4. related_lit_docs_global_titles
                    else:
                        codes[code]["related_lit_docs_per_code_large_in_vecstore"] = False
                        codes[code]["related_lit_docs_per_code_small_in_vecstore"] = False
                        codes[code]["related_lit_docs_global_in_vecstore"] = False
                        codes[code]["related_lit_docs_global_titles_in_vecstore"] = False
        else:
            for code in codes:
                codes[code]["related_lit_docs_per_code_large_in_vecstore"] = False
                codes[code]["related_lit_docs_per_code_small_in_vecstore"] = False
                codes[code]["related_lit_docs_global_in_vecstore"] = False
                codes[code]["related_lit_docs_global_titles_in_vecstore"] = False                               
        return codes

    def pycodeLoader(self, code):
        #load the code from the code_path
        #return as [a document type]
        code_path = code["path"]
        with open(code_path, "r") as f:
            code_page_content = f.read()
        metadata = {}
        #iterate through the keys in the code
        for key in code:
            metadata[key] = code[key]
        return [Document(page_content=code_page_content, metadata=metadata)]
    
    #TODO: Note that the current splitting will exclude the module header, they will be part of the prompt
    def line_splitter(self, doc, length=5, based_on_code_lines_only = False, csv_code = None, csv_comment = None, module_header = None):
        #split the doc's page_content by line with length
        #replicate the metadata for each splitted doc
        #return as [a list of document type]

        #The lines are splitted with commented code
        if not based_on_code_lines_only:
            page_content = copy.deepcopy(doc.page_content)
            #exclude the module header
            assert module_header is not None
            page_content = page_content.split(module_header, 1)[1]
            lines = page_content.split("\n")
            splitted_docs = []
            for i in range(0, len(lines), length):
                metadata = copy.deepcopy(doc.metadata)
                metadata["splitted_idx"] = list(range(i, min(i+length, len(lines))))
                metadata["split_based_on_code_lines_only"] = False
                splitted_docs.append(Document(page_content="\n".join(lines[i:i+length]), metadata=metadata))
        #else the lines are splitted with pure code, and the comments will follow the code
        else:
            df_comment = pd.read_csv(csv_comment)
            df_code = pd.read_csv(csv_code)
            df_comment["line_number"] = df_comment["line_number"].astype(int)
            df_code["line_number"] = df_code["line_number"].astype(int)
            df_comment = df_comment.sort_values("line_number")
            df_code = df_code.sort_values("line_number")
            
            code_lines = df_code["content"].tolist()
            comment_lines = df_comment["content"].tolist()
            
            module_header_end_line = 0
            #find the module header end line
            for i, line in enumerate(code_lines):
                if ";" in line:
                    modue_header_end_line = i
                    break

            if  len(code_lines[modue_header_end_line].split(";", 1)) > 1:
                split_start_idx = modue_header_end_line
            else:
                split_start_idx = modue_header_end_line+1

            #make sure comment lines and code lines are aligned
            try:
                assert len(code_lines) == len(comment_lines)
            except Exception as e:
                print(doc.metadata["path"])
                raise Exception("Code lines and comment lines are not aligned")
            
            splitted_docs = []
            for i in range(split_start_idx, len(code_lines), length):
                metadata = copy.deepcopy(doc.metadata)
                page_content = ""
                for j in range(i, min(i+length, len(code_lines))):
                    if "no_comment" not in comment_lines[j]:
                        page_content += "// "+comment_lines[j] + "\n"
                    if j == modue_header_end_line:
                        page_content = code_lines[j].split(";", 1)[1]
                    else:
                        page_content += code_lines[j] + "\n"
                metadata["splitted_idx"] = list(range(i, min(i+length, len(code_lines))))
                metadata["split_based_on_code_lines_only"] = True
                splitted_docs.append(Document(page_content=page_content, metadata=metadata))
        return splitted_docs
    
    
    #TODO: implement extract function header for HLS; also extend the support for HLS in general
    def extract_function_header(self, doc):
        raise NotImplementedError

    def extract_module_header(self, csv_code):
        df_code = pd.read_csv(csv_code)
        df_code["line_number"] = df_code["line_number"].astype(int)
        df_code = df_code.sort_values("line_number")
        
        code_lines = df_code["content"].tolist()        
        module_header_end_line = 0
        #find the module header end line
        for i, line in enumerate(code_lines):
            if ";" in line:
                module_header_end_line = i
                break
        module_header = "\n".join(code_lines[:module_header_end_line+1])
        module_header = module_header.split(";", 1)[0]+";"
        return module_header

    def load_and_split_code(self, skip_small_doc = False, split_by_line = False, line_length=10, based_on_code_lines_only = False, csv_code_dir = None, csv_comment_dir = None):
        self.splitted_docs_large = []
        self.splitted_docs_small = []
        for code in tqdm(self.codes, total=len(self.codes)):
            loaded_code = self.pycodeLoader(self.codes[code])
            if csv_code_dir is not None and csv_comment_dir is not None:
                self.codes[code]["csv_code"] = os.path.join(csv_code_dir, code.split(".")[0] + ".csv")
                self.codes[code]["csv_comment"] = os.path.join(csv_comment_dir, code.split(".")[0] + ".csv")
                self.codes[code]["module_header"] = self.extract_module_header(self.codes[code]["csv_code"])
            
            #TODO: tune the chunk size, overlap, etc.
            text_splitter_large = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap = 100)
            text_splitter_small = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 30)
            if split_by_line:
                docs = self.line_splitter(loaded_code[0], 
                                          length=line_length,
                                          based_on_code_lines_only=based_on_code_lines_only, 
                                          csv_code=self.codes[code]["csv_code"], 
                                          csv_comment=self.codes[code]["csv_comment"],
                                          module_header=self.codes[code]["module_header"])
                self.codes[code]["splitted_idx"] = [doc.metadata["splitted_idx"] for doc in docs]
            else:
                docs = text_splitter_large.split_documents(loaded_code)
            self.codes_inMemory[code]["splitted_docs_large"] = docs
            self.splitted_docs_large.extend(docs)
            
            #tagging the docs
            self.codes_inMemory[code]["id_key"] = "doc_id"
            self.codes_inMemory[code]["doc_ids"] = [str(uuid.uuid4()) for _ in self.codes_inMemory[code]["splitted_docs_large"]]
            #use file_name + doc_idx as the id
            # self.codes_inMemory[code]["doc_ids"] = [self.codes[code]["name"] + "_" + str(i) for i in range(len(self.codes_inMemory[code]["splitted_docs_large"]))]

            #splitting the docs further
            self.codes_inMemory[code]["splitted_docs_small"] = []     
            if not skip_small_doc:       
                for i, doc in enumerate(docs):
                    _id = self.codes_inMemory[code]["doc_ids"][i]
                    _sub_docs = text_splitter_small.split_documents([doc])
                    for _doc in _sub_docs:
                        _doc.metadata[self.codes_inMemory[code]["id_key"]] = _id
                    self.codes_inMemory[code]["splitted_docs_small"].extend(_sub_docs)
                    self.splitted_docs_small.extend(_sub_docs)

        return self.splitted_docs_large

    def create_vectorstore_per_code_small(self, code):
        # The vectorstore to use to index the child chunks
        vectorstore_per_code_small = Chroma(
            persist_directory=os.path.join(self.vectorembedding_dir, code + "_small"),
            collection_name="code_library_small_blocks",
            embedding_function=OpenAIEmbeddings()
        )
        return vectorstore_per_code_small

    def create_vectorstore_per_code_large(self, code):
        # The vectorstore to use to index the child chunks
        vectorstore_per_code_large = Chroma(
            persist_directory=os.path.join(self.vectorembedding_dir, code + "_large"),
            collection_name="code_library_large_blocks",
            embedding_function=OpenAIEmbeddings()
        )
        return vectorstore_per_code_large

    def init_vectorstore(self, global_summary_chain_from_verilog_eval = True,
                         block_summary_model="gpt-3.5-turbo-1106",
                         global_summary_model="gpt-3.5-turbo-1106",
                         global_summary_example_cstr_json = f"{os.environ.get('DATA4AIGCHIP_HOME')}/auto_data_gen_val/preprocess_data/example_code_strings_detailed_instructions.json",
                         global_summary_example_code_description_file = f"{os.environ.get('DATA4AIGCHIP_HOME')}/verilog_eval/descriptions/VerilogDescription_Machine.jsonl"
                         ):
        self.vectorstore_per_code_small = {}
        self.vectorstore_per_code_large = {}

        self.vectorstore_global = Chroma(
            persist_directory=os.path.join(self.vectorembedding_dir, "global"),
            collection_name="code_library_summary",
            embedding_function=OpenAIEmbeddings()
        )
        self.vectorstore_global_title = Chroma(
            persist_directory=os.path.join(self.vectorembedding_dir, "global_title"),
            collection_name="code_library_title",
            embedding_function=OpenAIEmbeddings()
        )
        #TODO: embed memory
        self.block_summary_chain = gen_block_summary_chain(model=block_summary_model)
        self.global_summary_chain_from_verilog_eval = global_summary_chain_from_verilog_eval
        if not self.global_summary_chain_from_verilog_eval:
            global_summary = Global_summary(model=global_summary_model)
            self.example_cstr_json = global_summary_example_cstr_json
            with open(self.example_cstr_json, "r") as f:
                self.example_code_strings = json.load(f)
            self.example_code_description_file = global_summary_example_code_description_file
            self.global_summary_chain = global_summary.gen_global_summary
        else:
            verilogeval0 = VerilogEval(model=global_summary_model)
            self.example_cstr_json = global_summary_example_cstr_json
            with open(self.example_cstr_json, "r") as f:
                self.example_code_strings = json.load(f)
            self.example_code_description_file = global_summary_example_code_description_file
            self.global_summary_chain = verilogeval0.verilog_eval_sft_data
        self.detail_steps_chain = detail_steps_chain()
        return None

    def supplement_summary(self, block_summary_placeholding=True, 
                           force_refresh_global_summary_high_level=False,
                            force_refresh_global_summary_detailed=False,
                           force_refresh_block_summary=False,
                           global_summary_example_desc_key="detail_description",
                           use_global_summary_for_block_summary=True, 
                           validate_global_summary=False):
        total_codes = len(self.codes)
        doc_count = 0
        self.block_summary_placeholding = block_summary_placeholding
        if force_refresh_global_summary_high_level:
            for code in self.codes:
                self.codes[code]["global_summary_high_level"] = None
        if force_refresh_global_summary_detailed:
            for code in self.codes:
                self.codes[code]["global_summary_detailed"] = None
        if force_refresh_block_summary:
            for code in self.codes:
                self.codes[code]["block_summary"] = None

        for code in self.codes:
            start_time = time.time()
            if not self.block_summary_placeholding:
                if self.codes[code]["block_summary"] is None or self.codes[code]["block_summary"][0] is None:
                    print("Generating block summary for " + code)
                    gen_from_scratch = True

                    # if len(self.codes_inMemory[code]["splitted_docs_large"]) == 1 and use_global_summary_for_block_summary:
                    #     print("Attempting to reuse the global summary")
                    #     if self.codes[code]["global_summary_from_line_comments"] is not None:
                    #         self.codes[code]["block_summary"] = [self.codes[code]["global_summary_from_line_comments"]]
                    #         gen_from_scratch = False
                    #     else:
                    #         print("No global summary found, generating block summary from scratch")
                    #         gen_from_scratch = True
                            
                    if gen_from_scratch:
                        doc_len = len(self.codes_inMemory[code]["splitted_docs_large"])
                        print("Doc length: " + str(doc_len))
                        
                        tmp_summary_list = [doc.page_content for doc in self.codes_inMemory[code]["splitted_docs_large"]]
                        for doc1 in tmp_summary_list:
                            #first block summary
                            if self.codes[code]["block_summary"] is None or self.codes[code]["block_summary"][0] is None:
                                self.codes[code]["block_summary"] = [self.block_summary_chain.predict(module_header=self.codes[code]["module_header"], 
                                                                                                            block_num = str(0),
                                                                                                            block_doc_history=" ", 
                                                                                                            doc1 = doc1)
                                                                    ]
                            else:
                                block_doc_history = "Here are the previous block summaries.\n"
                                for doc_idx, previous_summary in enumerate(self.codes[code]["block_summary"]):
                                    block_doc_history += "\n block_{block_num} Code: \n\n```\n{doc1}\n```\n\n".format(
                                        block_num = str(doc_idx),
                                        doc1 = tmp_summary_list[doc_idx]
                                    )
                                    block_doc_history += "\n block_{block_num} Summary: \n\n```\n{summary}\n```\n\n".format(
                                        block_num = str(doc_idx),
                                        summary = previous_summary
                                    )
                                self.codes[code]["block_summary"].append(self.block_summary_chain.predict(module_header=self.codes[code]["module_header"], 
                                                                                                            block_num = str(len(self.codes[code]["block_summary"])),
                                                                                                            block_doc_history=block_doc_history, 
                                                                                                            doc1 = doc1).replace("block_{} Summary:".format(len(self.codes[code]["block_summary"])), "") 
                                                                    )
                    block_summary_token_count = 0
                    for tmp_summary in self.codes[code]["block_summary"]:
                        print(tmp_summary)
                        block_summary_token_count += len(self.tokenizer.encode(tmp_summary))
                    print("Block summary token count: " + str(block_summary_token_count))
            else:
                #placeholding
                if self.codes[code]["block_summary"] is None or self.codes[code]["block_summary"][0] is None:
                    self.codes[code]["block_summary"] = [None for _ in self.codes_inMemory[code]["splitted_docs_large"]]

            print(self.cb)
            self.dump_bookkeeping()
            if self.codes[code]["global_summary_high_level"] is None \
            or force_refresh_global_summary_high_level \
            or self.codes[code]["global_summary_detaied"] is None \
            or force_refresh_global_summary_detailed:
                print("Generating global summary for " + code)
                if not self.global_summary_chain_from_verilog_eval:
                    if self.codes[code]["block_summary"] is None or self.codes[code]["block_summary"][0] is None:
                        raise Exception("Block summary not generated")
                    code_content = ""
                    with open(self.codes[code]["path"], "r") as f:
                        code_content = f.read()
                    if global_summary_example_desc_key == "detail_description":
                        summary_key = "global_summary_detailed"
                    else:
                        summary_key = "global_summary_high_level"
                    try:
                        self.codes[code][summary_key] = self.global_summary_chain(code_content, 
                                                                                    block_summaries=self.codes[code]["block_summary"],
                                                                                    example_code_description_file=self.example_code_description_file,
                                                                                    example_code_strings=self.example_code_strings,
                                                                                    desc_key=global_summary_example_desc_key)
                    except Exception as e:
                        #most likely token limit exceeded
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        try:
                            self.codes[code][summary_key] = self.global_summary_chain(code_content, 
                                                                                example_code_description_file=self.example_code_description_file,
                                                                                example_code_strings=self.example_code_strings,
                                                                                desc_key=global_summary_example_desc_key)
                        except Exception as e:
                            print(e)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                            raise Exception("Global summary generation failed")

                else:
                    code_content = ""
                    with open(self.codes[code]["path"], "r") as f:
                        code_content = f.read()
                    if global_summary_example_desc_key == "detail_description":
                        summary_key = "global_summary_detailed"
                    else:
                        summary_key = "global_summary_high_level"
                    try:
                        self.codes[code][summary_key] = self.global_summary_chain(code_content, 
                                                                                    example_code_description_file=self.example_code_description_file,
                                                                                    example_code_strings=self.example_code_strings,
                                                                                    desc_key=global_summary_example_desc_key)
                    except Exception as e:
                        #most likely token limit exceeded
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print("Generating global summary from block summaries: ")
                        if self.codes[code]["block_summary"] is None or self.codes[code]["block_summary"][0] is None:
                            raise Exception("Block summary not generated")
                        combined_summary = [Document(page_content=" ".join(self.codes[code]["block_summary"]))]
                        try:
                            self.codes[code][summary_key] = self.global_summary_chain(combined_summary, 
                                                                                example_code_description_file=self.example_code_description_file,
                                                                                example_code_strings=self.example_code_strings,
                                                                                desc_key=global_summary_example_desc_key)
                        except Exception as e:
                            print(e)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                            raise Exception("Global summary generation failed")

                        
            self.dump_bookkeeping()
            print(self.cb)
            finish_time = time.time()
            doc_count += 1
            print("Code summary {}/{} processed".format(doc_count, total_codes))
            print("Estimated time left: " + str(datetime.timedelta(seconds=(finish_time - start_time) * (total_codes - doc_count))))
            print("====="*10)
        return None

    def save_block_summary(self, fname, split_by_line = False):
        #dump the block summary into json
        block_summaries = {}
        for code in self.codes:
            block_summaries[code] = {}
            block_summaries[code]["block_summary"] = self.codes[code]["block_summary"]
            if split_by_line:
                block_summaries[code]["splitted_idx"] = [doc.metadata["splitted_idx"] for doc in self.codes_inMemory[code]["splitted_docs_large"]]
            block_summaries[code]["module_header"] = self.codes[code]["module_header"]
        with open(fname, "w") as f:
            json.dump(block_summaries, f)
        return None
    
    def save_global_summary(self, fname):
        #dump the global summary into json
        global_summaries = {}
        for code in self.codes:
            global_summaries[code] = {}
            global_summaries[code]["global_summary_high_level"] = self.codes[code]["global_summary_high_level"]
            global_summaries[code]["global_summary_detailed"] = self.codes[code]["global_summary_detailed"]
            global_summaries[code]["global_summary_from_line_comments"] = self.codes[code]["global_summary_from_line_comments"]
            global_summaries[code]["module_header"] = self.codes[code]["module_header"]
        with open(fname, "w") as f:
            json.dump(global_summaries, f)
        return None

    def supplement_detailed_steps(self, batch=5):
        total_codes = len(self.codes)
        doc_count = 0
        global_summary_list = []
        commented_code_list = []
        module_header_list = []
        code_in_batch = []
        for code in self.codes:
            if doc_count % batch == 0:
                start_time = time.time()
            #prepare the batch for the llm model 
            if self.codes[code]["detailed_steps"] is None:
                print("Generating detailed steps for " + code)
                self.codes[code]["detailed_steps"] = []
                global_summary = self.codes[code]["global_summary_detailed"]
                commented_code_blocks = [doc.page_content for doc in self.codes_inMemory[code]["splitted_docs_large"]]
                global_summary_list.extend([global_summary] * len(commented_code_blocks))
                commented_code_list.extend(commented_code_blocks)
                module_header_list.extend([self.codes[code]["module_header"]] * len(commented_code_blocks))
                code_in_batch.extend([code] * len(commented_code_blocks))
            if doc_count % batch == batch - 1 or doc_count == total_codes - 1:
                #generate the detailed steps
                zip_input = [{"high_level_summary": global_summary_list[i], 
                              "module_header": module_header_list[i],
                              "commented_code": commented_code_list[i]} 
                              for i in range(len(global_summary_list))]
                response = self.detail_steps_chain.batch(zip_input)
                response_str_lst = [tmp["text"] for tmp in response]
                for i, code_name in enumerate(code_in_batch):
                    self.codes[code_name]["detailed_steps"].append(response_str_lst[i])

                #clear the batch
                global_summary_list = []
                commented_code_list = []
                module_header_list = []
                code_in_batch = []
                self.dump_bookkeeping()
                print(self.cb)
                finish_time = time.time()
                print("Code detailed steps {}/{} processed".format(doc_count, total_codes))
                print("Estimated time left: " + str(datetime.timedelta(seconds=(finish_time - start_time) * (total_codes - doc_count))))
                print("====="*10)
            doc_count += 1
            
    def save_detail_steps(self,fname, split_by_line = False):
        #dump the detailed_steps into json
        detailed_steps = {}
        for code in self.codes:
            detailed_steps[code] = {}
            detailed_steps[code]["detailed_steps"] = self.codes[code]["detailed_steps"]
            if split_by_line:
                detailed_steps[code]["splitted_idx"] = [doc.metadata["splitted_idx"] for doc in self.codes_inMemory[code]["splitted_docs_large"]]
            detailed_steps[code]["module_header"] = self.codes[code]["module_header"]
        with open(fname, "w") as f:
            json.dump(detailed_steps, f)
        return None
                   
    def form_document_metadata(self):
        #per code-wise
        self.related_lit_docs_per_code_large = {}
        self.related_lit_docs_per_code_small = {}
        
        #create the doc lists for the whole codes
        self.related_lit_docs_global = []
        self.related_lit_docs_global_titles = []
        self.global_code_id_key = "doc_id"
        self.global_code_doc_ids = {}
        for code in self.codes:
            self.global_code_doc_ids[code] = str(uuid.uuid4())
            #use file_name as the id
            # self.global_code_doc_ids[code] = self.codes[code]["name"]

        for code in self.codes:
            code_title = self.codes[code]["name"]
            code_genre = self.codes[code]["genre"]
            self.related_lit_docs_per_code_large[code] = [Document(page_content=doc.page_content, metadata={
                                                self.codes_inMemory[code]["id_key"]: self.codes_inMemory[code]["doc_ids"][i], 
                                                "summary": self.codes[code]["block_summary"][i],
                                                "title": code_title,
                                                "genre": code_genre
                                                }) for i, doc in enumerate(self.codes_inMemory[code]["splitted_docs_large"])]
            #id metadata already added; if needed add summary metadata here ...
            self.related_lit_docs_per_code_small[code] = [Document(page_content=doc.page_content, metadata={
                                                self.codes_inMemory[code]["id_key"]: doc.metadata[self.codes_inMemory[code]["id_key"]],
                                                "title": code_title,
                                                "genre": code_genre
                                                }) for i, doc in enumerate(self.codes_inMemory[code]["splitted_docs_small"])]
            
            #use code summary as page content
            code_summary = self.codes[code]["global_summary_detailed"]
            if code_summary is None:
                code_summary = "None"
            self.related_lit_docs_global.append(
                Document(page_content=code_summary, metadata={
                                                self.global_code_id_key: self.global_code_doc_ids[code],
                                                "title": code_title,
                                                "genre": code_genre
                                                }
                        )
            )
            #use code title as page content
            self.related_lit_docs_global_titles.append(
                Document(page_content=code_title, metadata={
                                                self.global_code_id_key: self.global_code_doc_ids[code],
                                                "title": code_title,
                                                "summary": code_summary,
                                                "genre": code_genre
                                                }
                        )
            )

    def create_retriever_per_code_small(self, code):
        #get the first code's id_key
        id_key = self.codes_inMemory[list(self.codes_inMemory.keys())[0]]["id_key"]
        store = InMemoryStore()
        vectorstore_per_code_small = self.create_retriever_per_code_small(code)
        #the vectorstore is built on the small docs
        retriever_per_code_small = MultiVectorRetriever(
                                                vectorstore=vectorstore_per_code_small,
                                                docstore=store, 
                                                id_key=id_key,
                                            )
        #More intelligently to decide whether to add the documents to the vectorstore
        if not self.codes[code]["related_lit_docs_per_code_small_in_vecstore"]:
            self.codes[code]["related_lit_docs_per_code_small_in_vecstore"] = True
            retriever_per_code_small.vectorstore.add_documents(self.related_lit_docs_per_code_small[code])
        retriever_per_code_small.docstore.mset(list(zip(self.codes_inMemory[code]["doc_ids"], self.codes_inMemory[code]["splitted_docs_large"])))
        return retriever_per_code_small
        
    def init_retriever(self):

        #global retriever
        id_key = self.global_code_id_key
        store = InMemoryStore()
        self.retriever_global = MultiVectorRetriever(
                                                    vectorstore=self.vectorstore_global,
                                                    docstore=store, 
                                                    id_key=id_key,
                                                )
        #More intelligently to decide whether to add the documents to the vectorstore
        #get the index of the not added docs
        not_added_idx = [i for i, doc in enumerate(self.related_lit_docs_global) if not self.codes[doc.metadata["title"]]["related_lit_docs_global_in_vecstore"]]
        if len(not_added_idx) > 0:
            #add the docs to the vectorstore
            self.retriever_global.vectorstore.add_documents([self.related_lit_docs_global[i] for i in not_added_idx])
        #update the status of the docs
        for i in not_added_idx:
            self.codes[self.related_lit_docs_global[i].metadata["title"]]["related_lit_docs_global_in_vecstore"] = True
        self.retriever_global.docstore.mset(list(zip(self.global_code_doc_ids.values(), self.related_lit_docs_global)))
        #global title retriever
        id_key = self.global_code_id_key
        store = InMemoryStore()
        self.retriever_global_title = MultiVectorRetriever(
                                                    vectorstore=self.vectorstore_global_title,
                                                    docstore=store, 
                                                    id_key=id_key,
                                                )
        #More intelligently to decide whether to add the documents to the vectorstore
        #get the index of the not added docs
        not_added_idx = [i for i, doc in enumerate(self.related_lit_docs_global_titles) if not self.codes[doc.metadata["title"]]["related_lit_docs_global_titles_in_vecstore"]]
        if len(not_added_idx) > 0:
            #add the docs to the vectorstore
            self.retriever_global_title.vectorstore.add_documents([self.related_lit_docs_global_titles[i] for i in not_added_idx])
        #update the status of the docs
        for i in not_added_idx:
            self.codes[self.related_lit_docs_global_titles[i].metadata["title"]]["related_lit_docs_global_titles_in_vecstore"] = True
        self.retriever_global_title.docstore.mset(list(zip(self.global_code_doc_ids.values(), self.related_lit_docs_global_titles)))

        self.func_name_lookup_chain = func_name_lookup_chain(language=self.language)
        print(self.cb)

    #TODO: Need to rework retrieval, not robust enough
    #TODO: add similarity thresholding
    def retrieve_global(self, query):
        # code_doc_selected = self.retriever_global.get_relevant_documents(query)
        code_doc_selected_with_score = self.retriever_global.vectorstore.similarity_search_with_relevance_scores(query)
        #sort the docs by the score
        code_doc_selected_with_score = sorted(code_doc_selected_with_score, key=lambda x: x[1], reverse=True)
        #most similar score
        highest_score = [code_doc_selected_with_score[0][1]]
        if highest_score[0] < 0.5:
            return []
        else:
            code_doc_selected = [code_doc_selected_with_score[0][0]]
            return code_doc_selected
    
    def retrieve_global_title(self, query):
        # code_doc_selected = self.retriever_global_title.vectorstore.similarity_search(query)
        code_doc_selected_with_score = self.retriever_global_title.vectorstore.similarity_search_with_relevance_scores(query)
        #sort the docs by the score
        code_doc_selected_with_score = sorted(code_doc_selected_with_score, key=lambda x: x[1], reverse=True)
        #most similar score
        highest_score = [code_doc_selected_with_score[0][1]]
        if highest_score[0] < 0.5:
            return []
        else:
            code_doc_selected = [code_doc_selected_with_score[0][0]]
            return code_doc_selected
    
    def retrieve_per_code(self, query, code):
        retriever_per_code_small = self.create_retriever_per_code_small(code)
        small_docs = retriever_per_code_small.vectorstore.similarity_search(query)
        # large_docs = self.retriever_per_code_small[code].get_relevant_documents(query)
        #TODO: Need to rework retrieval; a hacky solution for now
        large_docs = retriever_per_code_small.docstore.mget([doc.metadata[self.codes_inMemory[code]["id_key"]] for doc in small_docs])
        #delete vectorestore in memory
        del retriever_per_code_small
        return small_docs, large_docs
    
    def retrieve_progressive(self, query):
        code_doc_selected = self.retrieve_global(query)
        code_name = code_doc_selected[0].metadata["title"]
        #TODO: potentially modify the query?
        small_docs, large_docs = self.retrieve_per_code(query, code_name)
        return code_doc_selected, small_docs, large_docs

    def dump_bookkeeping(self):
        with open(os.path.join(self.bookkeeping_dir, "codes.json"), "w") as f:
            json.dump(self.codes, f)
        return None
    
    def clear_block_summary(self):
        for code in self.codes:
            self.codes[code]["block_summary"] = None
            self.codes[code]["related_lit_docs_per_code_small_in_vecstore"] = False
        #clear vectorestore

        self.dump_bookkeeping()
    
    def clear_global_summary(self):
        for code in self.codes:
            self.codes[code]["global_summary_high_level"] = None
            self.codes[code]["global_summary_detailed"] = None
            self.codes[code]["global_summary_from_line_comments"] = None
            self.codes[code]["related_lit_docs_global_in_vecstore"] = False
            self.codes[code]["related_lit_docs_global_titles_in_vecstore"] = False
            #clear vectorestore
            # self.retriever_global.vectorstore.search()
        self.dump_bookkeeping()
        
        


if __name__ == "__main__":

    ############################################################
    #test code to generate the block summaries
    ############################################################

    # #move the block summaries to the documented folder
    # new_json = process_block_summary("/home/user_name/DAC_2024/chatgpt4_auto_accel/fine_tune_dataset/auto_doc_part_dataset/block_summaries/part5/block_summary.json")
    # # merged_dict = merge_dict_lst(new_json)
    # with open("/home/user_name/DAC_2024/ckpts/test_10_30_5_complete/block_summaries.json", "w") as f:
    #     json.dump(new_json, f)
    # exit()

    with get_openai_callback() as cb:
        codedb = CodeDataset(
                             "/home/user_name/DAC_2024/ckpts/test_10_30_5_complete/documented_code/", 
                             bookkeeping_dir="./langchain_test/bookkeep/", 
                             vectorembedding_dir="./langchain_test/embedding_data/", 
                             force_refresh=False,
                             cb=cb
                             )
        codedb.load_and_split_code(skip_small_doc=True, split_by_line=True, based_on_code_lines_only=True, 
                                    csv_code_dir="/home/user_name/DAC_2024/ckpts/test_10_30_5_complete/assets/verilog/code_and_comment_src/csv_src/csv_code_src",
                                    csv_comment_dir="/home/user_name/DAC_2024/ckpts/test_10_30_5_complete/assets/verilog/code_and_comment_src/csv_src/csv_new_comment_src"
                                  )
        codedb.init_vectorstore()
        codedb.supplement_summary(block_summary_placeholding=False)
        codedb.save_block_summary(
                                  "./langchain_test/block_summary.json", 
                                  split_by_line = True
                                  )
        exit()
        codedb.form_document_metadata()
        codedb.init_retriever()
        codedb.dump_bookkeeping()
        exit()
        trial = 0
        docs = []
        while trial < 10:
            print("Trial " + str(trial))
            docs = codedb.retrieve_global_title("multiplier")
            if len(docs) > 0:
                print("Found docs")
            else:
                print("No docs found")
            trial += 1
        if len(docs) == 0:
            print("No docs found")
            exit()
        print(docs[0].metadata["title"])
        print(docs[0].metadata["summary"])
        print(cb)
        exit()

    #################### legacy testing code  ####################
    # #create system context embedding
    # fields = ["Filename", "File type", "Summary", "Text", "Line_id"]
    # system_embedder = EmbedTool0(fields, 
    #                             os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR")),
    #                             os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_EMBEDDING_DIR")),
    #                             "system_context_embedding.csv")
    # # system_embedder.create_raw_system_context()
    # # system_embedder.create_embedding()
    # # exit()
    # system_embedder.load_embedding()
    # search_contexts(system_embedder.embedding_model, system_embedder.df_embed, "Zaizai's job")
    