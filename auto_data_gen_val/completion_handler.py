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
from langchain.callbacks import get_openai_callback #with get_openai_callback() as cb:
from chain_utils import SimpleConverseChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from my_pydantic import PydanticOutputParserMessages
from langchain.llms import HuggingFaceTextGenInference
from langchain.schema import SystemMessage, AIMessage, HumanMessage


def load_system_messages_all_in_one(msg_file, optional_system_messages):
    #check if msg_file is exist
    if not os.path.exists(msg_file):
        system = [{'role': "system", "content": ""}]
        print("Warning: system message loading skipped for {}".format(msg_file))
        system[0]["content"] = ""
    else:
        with open(msg_file, "r") as f:
            msg = f.read()
        system = [{'role': "system", "content": ""}]
        system[0]["content"] = msg
    for msg_str in optional_system_messages:
        system[0]["content"] += msg_str
    return system

def load_system_messages_separate(msg_file, optional_system_messages):
    with open(msg_file, "r") as f:
        msg = f.readlines()
    system = []
    for i in range(len(msg)):
        system.append({'role': "system", "content": ""})
        system[i]["content"] = msg[i]
    for msg_str in optional_system_messages:
        system.append({'role': "system", "content": ""})
        system[-1]["content"] = msg_str
    return system

def update_the_messages(messages, new_message, length = 5):
    if len(messages) < length:
        messages.append(new_message)
    else:
        #shift the messages to the left by 1
        for i in range(len(messages)-1):
            messages[i] = messages[i+1]
        #add the new message to the end
        messages[-1] = new_message
    return messages
##############################################################################################################

def load_system_messages_all_in_one_str(msg_file, optional_system_messages):
    system_msg = ""
    #check if msg_file is exist
    if not os.path.exists(msg_file):
        print("Warning: system message loading skipped for {}".format(msg_file))
    else:
        with open(msg_file, "r") as f:
            msg = f.read()
        system_msg += msg
    for msg_str in optional_system_messages:
        system_msg += msg_str + "\n"
    return system_msg



def chat_request(line_by_line_comment_converse_chain, user_input, optional_system_messages, system_message_file):
     
    system_msg = load_system_messages_all_in_one_str(system_message_file, optional_system_messages)

    response = line_by_line_comment_converse_chain.chat(human_input=user_input, system_prompt=system_msg)
        
    return response


class Line_comment_format(BaseModel):
    comment_exist: list = Field(description="List of boolean value denating if comment exist for each code line")
    comment: list = Field(description="List of string comments, each of which is the comment for the corresponding code line; empty string if comment does not exist")
    line_number: list = Field(description="list of code line numbers")

    # comment_exist should be list of boolean
    @validator('comment_exist')
    def comment_exist_must_be_list_of_boolean(cls, v, values, **kwargs):
        if not isinstance(v, list):
            raise ValueError('comment_exist must be list of boolean')
        for i in v:
            if not isinstance(i, bool):
                raise ValueError('comment_exist must be list of boolean')
        return v
    # comment should be list of string
    @validator('comment')
    def comment_must_be_list_of_string(cls, v, values, **kwargs):
        if not isinstance(v, list):
            raise ValueError('comment must be list of string')
        for i in v:
            if not isinstance(i, str):
                raise ValueError('comment must be list of string')
        return v
    # line_number should be list of int
    @validator('line_number')
    def line_number_must_be_list_of_int(cls, v, values, **kwargs):
        if not isinstance(v, list):
            raise ValueError('line_number must be list of int')
        for i in v:
            if not isinstance(i, int):
                raise ValueError('line_number must be list of int')
        # line numbers should be unique and consecutive (v[i] = v[i-1] + 1)
        if not len(set(v)) == len(v):
            raise ValueError('line_number must be unique')
        #TODO: a temporary fix; should cover cases not consecutive
        for i in range(1, len(v)):
            if not v[i] == v[i-1] + 1:
                raise ValueError('line_number must be consecutive')
        return v
    #validate the length of comment_exist, comment, and line_number
    @validator('line_number')
    def length_of_comment_exist_comment_line_number_must_be_equal(cls, v, values, **kwargs):
        if not len(values['comment_exist']) == len(values['comment']) == len(v):
            raise ValueError('length of comment_exist, comment, and line_number must be equal')
        return v

    # #if comment exist, then comment should not be empty
    # @validator('comment')
    # def comment_must_exist(cls, v, values, **kwargs):
    #     if values['comment_exist'] and v == "":
    #         raise ValueError('Comment must exist if comment_exist is True')
    #     return v

def Line_comment_format_fixing_chain(model="gpt-3.5-turbo-1106", temperature=0.7, max_tokens=128):
    llm = ChatOpenAI(max_retries=0, model=model, temperature=temperature, max_tokens=max_tokens, request_timeout=10)
    parser = PydanticOutputParserMessages(pydantic_object=Line_comment_format)
    prompt = "Fix the output format:\n{format_instructions}\n\n{doc}\n\n"
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["doc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = (
            {"doc": lambda x: x}
            | prompt_template
            | llm
            | parser
        )
    return chain

def summarize_comments_fixing_chain(model="gpt-3.5-turbo-1106", temperature=0.7, max_tokens=512):
    llm = ChatOpenAI(max_retries=0, model=model, temperature=temperature, max_tokens=max_tokens, request_timeout=10)
    parser = PydanticOutputParserMessages(pydantic_object=Code_summary_format)
    prompt = "Fix the output format:\n{format_instructions}\n\n{doc}\n\n"
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["doc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = (
            {"doc": lambda x: x}
            | prompt_template
            | llm
            | parser
        )
    return chain




class Code_summary_format(BaseModel):
    usage: str = Field(description="Usage of the code block")
    summary: str = Field(description="Summary of the code")


class Chatbot:
    def __init__(self, system_message_file, convers_store_dir, convers_samples_file,
                 code_suffix = [".v", ".sv", ".vh"], language="verilog",
                 code_lib_path= "./lib", code_vec_store = "./vector_store/",
                 skip_supplement_summary=False,
                 cb = None):
        #check if the system message file exists
        if system_message_file is None:
            #produce warning
            print("Warning: system message file None!")
        else:
            self.system_message_file = system_message_file
        if convers_store_dir is None:
            #produce warning
            print("Warning: conversation store dir None!")
        else:
            self.convers_store_dir = convers_store_dir
        if convers_samples_file is None:
            #produce warning
            print("Warning: conversation samples file None!")
        else:
            self.convers_samples_file = convers_samples_file
        #code libs for code retrieval
        self.code_lib_path = code_lib_path
        self.code_vec_store = code_vec_store
        self.skip_supplement_summary = skip_supplement_summary
        

        self.convers_memory_length = 1
        #if the convers store dir does not exist, create it
        if not os.path.exists(convers_store_dir):
            os.makedirs(convers_store_dir)
        
        self.system_context_embedding = None
        self.embedding_model = "text-embedding-ada-002"
        #cost tracker
        self.cb = cb

        #code dataset for retrieval
        self.code_suffix = code_suffix
        self.language = language
        self.codedb = CodeDataset(code_lib_path, 
                                  bookkeeping_dir=os.path.join(self.code_vec_store, "bookkeeping/"),
                                  vectorembedding_dir=os.path.join(self.code_vec_store, "vectorembedding/"),
                                  code_suffix=self.code_suffix,
                                  language=self.language,
                                  force_refresh=False,
                                  cb = self.cb)
    
        self.converse_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, input_key="human_input", k=self.convers_memory_length)       
        self.line_by_line_comment_response_parser = PydanticOutputParserMessages(pydantic_object=Line_comment_format)
        self.code_summary_response_parser = PydanticOutputParserMessages(pydantic_object=Code_summary_format)

        self.line_by_line_format_instructions = """Format your answer in json format, with entries of "comment_exist", "comment", and "line_number"; \n "comment_exist" is a List of boolean value denating if comment exist for each code line.\n "comment" is list of string comments, each of which is the comment for the corresponding code line; do not include original code here; empty string if comment does not exist. \n "line_number" is {line_numbers}. \n Here is the response format: {"comment_exist": [bool, bool], "comment": [str comment, str comment], "line_number": {line_numbers}}\n Only include the json response! Do not include anything else!\n"""
        self.line_by_line_comment_converse_chain = SimpleConverseChain(model="llama2", temperature=0.7, max_tokens=512, 
                                                  verbose=False, 
                                                  memory=self.converse_memory,
                                                  memory_length=self.convers_memory_length,
                                                  fixing_chain=Line_comment_format_fixing_chain(),
                                                  customized_format_instructions=self.line_by_line_format_instructions,
                                                  output_parser=self.line_by_line_comment_response_parser,
                                                  json_mode=True)
        
        self.summary_format_instructions = """Format your answer in json format, with entries of "usage" and "summary", denoting usage of the code block and summary of the code, respectively\n Do not include answer other than the json string.\n"""
        self.summarize_comments_chain = SimpleConverseChain(model="llama2", temperature=0.7, max_tokens=384,
                                                            verbose=False,
                                                            have_memory=False,
                                                            customized_format_instructions=self.summary_format_instructions,
                                                            output_parser=self.code_summary_response_parser)
        self.reverse_code_gen_chain = SimpleConverseChain(model="llama2", temperature=0.7, max_tokens=256,
                                                            verbose=False,
                                                            have_memory=False)



        self.converse_id_key = "convers_id"
        self.converse_vecstore = Chroma(
            persist_directory=self.convers_store_dir,
            collection_name="conversation_history",
            embedding_function=OpenAIEmbeddings()
        )
        store = InMemoryStore()
        self.converse_retriever = MultiVectorRetriever(
                                                vectorstore=self.converse_vecstore,
                                                docstore=store, 
                                                id_key=self.converse_id_key,
                                            )
        
        # self.load_most_recent_n_conversation_to_memory(n=self.convers_memory_length)
        

    def init_code_retrival(self):
        print("building code lib dataset for retrieval...")
        self.codedb.load_and_split_code()
        if len(self.codedb.codes) > 0:
            self.codedb.init_vectorstore()
            if not self.skip_supplement_summary:
                self.codedb.supplement_summary()
            self.codedb.form_document_metadata()
            self.codedb.init_retriever()
            self.codedb.dump_bookkeeping()
        else:
            print("Empty code lib dataset for retrieval!")
        if self.cb is not None:
            print(self.cb)

    def chat(self, user_input, code_name, convers_aware = True):
        #system embedding lookup 
        optional_system_messages = search_contexts(self.embedding_model, self.system_context_embedding, user_input, n=3)
        
        #whether globally search for the previous conversation history
        if False:
            #some previous conversation history
            relevant_converse_docs = self.converse_retriever.vectorstore.similarity_search(user_input, k=5)
            doc_str = ""
            for doc in relevant_converse_docs:
                doc_str += doc.metadata["raw_converse"] + "\n"
            if not doc_str == "":
                summary_prompt = """Summarize the following conversation history:\n{converse_histry}\n"""
                # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=128)
                llm = HuggingFaceTextGenInference(
                                inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                                max_new_tokens=128
                                )
                llm = llm.with_fallbacks([ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7, max_tokens=128)])
                summary_prompt_template = PromptTemplate(
                    template=summary_prompt,
                    input_variables=["converse_histry"]
                )
                summary_chain = (
                        {"converse_histry": lambda x: x}
                        | summary_prompt_template
                        | llm
                        | StrOutputParser()
                )
                summary = summary_chain.batch([doc_str])[0]
                user_input += " here is the summary of relevant conversation :\n" + summary + "\n"
            
        response = chat_request(self.line_by_line_comment_converse_chain, user_input, optional_system_messages, self.system_message_file)
        print(self.cb)

        #get the current time by hour
        now = datetime.datetime.now()
        time_stamp_ymh = now.strftime("%Y-%m-%d-%H")
        time_stamp_hminu = now.strftime("%H-%M")

        raw_converse = "user: {}\n assistant: {}\n".format(user_input, response)
        message_docunment = Document(page_content=raw_converse,
                                     metadata={
                                                self.converse_id_key: code_name,
                                                "raw_converse": raw_converse, 
                                                "time_stamp_ymh": time_stamp_ymh,
                                                "time_stamp_hminu": time_stamp_hminu,
                                                }
                            )
        self.converse_retriever.vectorstore.add_documents([message_docunment])
        #currently do not really add additional code document data for efficiency
        self.converse_retriever.docstore.mset([(code_name, code_name)])
        return response

    def load_most_recent_n_conversation_to_memory(self, n=5):
        #this will preload self.converse_memory
        converse_docs = self.converse_vecstore._collection.peek(n)
        for doc in converse_docs:
            self.converse_memory.save_context({"human_input": doc.metadata["raw_converse"].split("\n")[0].split(": ")[1]}, {"output": doc.metadata["raw_converse"].split("\n")[1].split(": ")[1]})
        return converse_docs
        
    def load_most_recent_n_coversations(self, n=20):
        #retrive the most recent n conversations from the convers_vecstore
        converse_docs = self.converse_vecstore._collection.peek(n)
        user_messages = []
        assistant_messages = []
        for doc in converse_docs:
            user_messages.append(doc.metadata["raw_converse"].split("\n")[0].split(": ")[1])
            assistant_messages.append(doc.metadata["raw_converse"].split("\n")[1].split(": ")[1])
        most_recent_time_stamp = converse_docs[0].metadata["time_stamp_hminu"]
        return user_messages, assistant_messages, most_recent_time_stamp
    
    def comment_a_code_file(self, csv_code_file, csv_comment_file, csv_new_comment_file, csv_pure_gen_comment_file, dependent_funcs = None):
        #load the code lines
        df_code = pd.read_csv(csv_code_file)
        #load the comment lines
        df_comment = pd.read_csv(csv_comment_file)
        #create an empty comment file with the same format as the csv_comment_file
        create_empty_csv(["content", "line_number"], csv_new_comment_file)
        create_empty_csv(["content", "line_number"], csv_pure_gen_comment_file)


        batch_comment_lines = 5

        total_code_lines = len(df_code)
        for line_number in range(0, total_code_lines, batch_comment_lines):
            print("Commenting lines: [", line_number, ", ", min(line_number + batch_comment_lines-1, total_code_lines-1), "]")
            line_ids = list(range(line_number, min(line_number + batch_comment_lines, total_code_lines)))
            
            #check if code line contains function call or module instantiation
            code_lines = form_function_retrival_prompt(df_code, df_comment, line_ids)
            #if the user does not provide the dependent functions
            if dependent_funcs is None:
                print("Using llm to check if there is function call or module instantiation in the code lines...")
                #use llm to check if there is function call or module instantiation in the code lines
                try:
                    func_lookup = self.codedb.func_name_lookup_chain[0].predict(doc=code_lines)
                except Exception as e:
                    if "Max retries exceeded" in str(e) or "Server error" in str(e):
                        print("Error: ", e)
                        exit()
                    #use a more robust model
                    func_lookup = self.codedb.func_name_lookup_chain[1].predict(doc=code_lines)
                #json string to dict
                func_lookup = json.loads(func_lookup.json())
            else:
                func_to_lookup = []
                for func in dependent_funcs:
                    if func in code_lines:
                        func_to_lookup.append(func)
                if len(func_to_lookup) == 0:
                    func_lookup = {"function_exist": False, "function_name": []}
                else:
                    func_lookup = {"function_exist": True, "function_name": func_to_lookup}

            print("func_lookup: ", func_lookup)
            if func_lookup["function_exist"]:
                func_names = func_lookup["function_name"]
                print("looking for: ", func_names)
                func_summarys = []
                func_names_found = []
                for func_name in func_names:
                    docs = self.codedb.retrieve_global_title(func_name)
                    if len(docs) > 0:
                        func_summary = docs[0].metadata["summary"]
                        func_name_found = docs[0].metadata["title"].split(".")[0]
                        print("Found function: ", func_name_found)
                        if func_name_found in func_names and func_name_found != csv_code_file.split("/")[-1].split(".")[0]:
                            func_summarys.append(func_summary)
                            func_names_found.append(func_name_found)
            else:
                func_names = []
                func_names_found = []
                func_summarys = []
                
            print("func_names found: ", func_names_found)
            print("func_summary: ", func_summarys)

            #only keep the function names that are found
            

            #line by line commenting
            prompt = form_commenting_prompt(df_code, df_comment, 
                                            line_ids, helper_function_names=func_names_found, 
                                            helper_function_summarys=func_summarys,
                                            format_instruction=None,
                                            # format_instruction=self.line_by_line_comment_response_parser.get_format_instructions()
                                            )
            self.line_by_line_comment_converse_chain.customized_format_instructions = self.line_by_line_format_instructions.replace("{line_numbers}", str(line_ids))

            #response in json format
            response = self.chat(prompt,csv_code_file)
            print("response: ", response)
            #json string to dict
            response = json.loads(response)
            
            for idx, line_id in enumerate(line_ids):
                #check if the code line returned is correct
                assert response["line_number"][idx] == line_id

                original_comment = df_comment[df_comment["line_number"] == line_id]["content"].values[0]
                if response["comment_exist"][idx]:
                    if response["comment"][idx] == "":
                        new_comment = original_comment
                    else:
                        #append the comment to the original comment
                        new_comment = original_comment.replace("no_comment", "") + response["comment"][idx] + "\n"
                        #add to the pure generated comment file
                        df = pd.DataFrame({"content": [response["comment"][idx]], "line_number": [line_id]})
                        df.to_csv(csv_pure_gen_comment_file, mode='a', header=False, index=False)
                else:
                    new_comment = original_comment
                #add new dataframes
                df = pd.DataFrame({"content": [new_comment], "line_number": [line_id]})
                df.to_csv(csv_new_comment_file, mode='a', header=False, index=False)



    def summarize_code_blocks(self, csv_code_file, csv_new_comment_file, code_summary_file):
        #summarize the code blocks based on the line by line comments in csv_pure_gen_comment_file
        prompt = ""
        prompt += "Summarize the following code:\n"
        prompt += "Code:\n"
        prompt += "```\n"
        prompt += merge_code_and_comment(csv_code_file, csv_new_comment_file)
        prompt += "\n```\n"
        prompt += "Be concise but informative.\n"
        prompt += self.code_summary_response_parser.get_format_instructions()
        system_prompt  = load_system_messages_all_in_one_str(self.system_message_file, [])
        response = self.summarize_comments_chain.chat(human_input=prompt,system_prompt=system_prompt)
        print("response: ", response)
        #dump the response to the code_summary_file
        with open(code_summary_file, "w") as f:
            f.write(response)
        print(self.cb)
        return response
        
    def reverse_code_gen(self, csv_pure_gen_comment_file, code_summary_file):
        prompt = "Try to generate the code based on the following input/output, usage, summary, and comments:\n"
        #load from the code_summary_file
        with open(code_summary_file, "r") as f:
            summary = f.read()
        prompt += summary + "\n"
        #load the comment lines
        df_comment = pd.read_csv(csv_pure_gen_comment_file)
        prompt += "Comment lines:\n"
        for i in sorted(df_comment["line_number"].values):
            prompt += df_comment[df_comment["line_number"] == i]["content"].values[0] + "\n"
        prompt += 'Format the response as: Code:\n```\n xxx \n```'
        system_prompt  = load_system_messages_all_in_one_str(self.system_message_file, [])
        response = self.reverse_code_gen_chain.chat(human_input=prompt,system_prompt=system_prompt)
        print("response: ", response)
        print(self.cb)
            
    def code_gen_line_by_line(self):
        pass



if __name__ == "__main__":
    #create system context embedding
    fields = ["Filename", "File type", "Summary", "Text", "Line_id"]
    system_embedder = EmbedTool0(fields, 
                                os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR")),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_EMBEDDING_DIR")),
                                "system_context_embedding.csv")
    system_embedder.create_raw_system_context()
    system_embedder.create_embedding()
    # exit()
    system_embedder.load_embedding()

    bot = Chatbot(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR"), 
                        "context.fixed_features.txt"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("CONVERSE_DIR")),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SYSTEM_CONTEXT_DIR"), 
                        "context.converse_samples.txt")
                        )
    bot.system_context_embedding = system_embedder.df_embed

    src_code_file = "assets/{}/code_and_comment_src/raw_src/raw_code_src/addr4.v".format(os.environ.get("TARGET_LANG"))
    csv_code_file = "assets/{}/code_and_comment_src/csv_src/csv_code_src/addr4.csv".format(os.environ.get("TARGET_LANG"))
    csv_comment_file = "assets/{}/code_and_comment_src/csv_src/csv_comment_src/addr4.csv".format(os.environ.get("TARGET_LANG"))
    csv_new_comment_file = "assets/{}/code_and_comment_src/csv_src/csv_new_comment_src/addr4.csv".format(os.environ.get("TARGET_LANG"))
    csv_pure_gen_comment_file = "assets/{}/code_and_comment_src/csv_src/csv_pure_gen_comment_src/addr4.csv".format(os.environ.get("TARGET_LANG"))
    code_summary_file = "assets/{}/code_and_comment_src/code_summary/addr4.txt".format(os.environ.get("TARGET_LANG"))
    # convert_raw_src_code_to_csv(src_code_file, csv_code_file, csv_comment_file, discard_original_comment = False)
    # bot.comment_a_code_file(csv_code_file, csv_comment_file, csv_new_comment_file, csv_pure_gen_comment_file)

    new_code_string = merge_code_and_comment(csv_code_file, csv_new_comment_file)

    with open("new_test_code.h", "w") as f:
        f.write(new_code_string)
    bot.summarize_code_blocks(csv_code_file, csv_new_comment_file, code_summary_file)
    bot.reverse_code_gen(csv_pure_gen_comment_file, code_summary_file)
