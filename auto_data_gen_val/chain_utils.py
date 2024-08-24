import pandas as pd    
import json
import copy
import os
import sys
import openai
import time


from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat

from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceTextGenInference
from langchain.output_parsers import OutputFixingParser
from transformers import GPT2Tokenizer

#NOTE: if langchain version is too old, change this /home/user_name/anaconda3/envs/tvm/lib/python3.8/site-packages/langchain/callbacks/openai_info.py 
# MODEL_COST_PER_1K_TOKENS 


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


##############################################################################################################
#LEGACY: changed to langchain framework

def completion_request(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    return response

def gpt3_5_request(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=prompt,
        temperature=0.7,
        max_tokens=4096
    )
    return response

def gpt4_request(prompt):
    #gpt-4, gpt-4-0314, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
    # print("prompt: ", prompt)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            # model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.7,
            max_tokens=1024
        )
    except Exception as e:
        #check if openai.error.RateLimitError: Rate limit reached. 
        print("Error: ", e)
        print("Rate limit reached, sleep for 1 min")
        time.sleep(60)
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            # model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.7,
            max_tokens=1024
        )
    return response

def openai_chat(system_prompt, humain_input, model="gpt-4-0613", max_token=4096):
      
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": humain_input},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        # temperature=0.7,
        max_tokens=max_token,
    )
    response_str = response["choices"][0]["message"]["content"]
    return response_str

class Usage_summary(BaseModel):
    usage: str = Field(description="usage")
    summary: str = Field(description="summary")

class Global_summary:
    def __init__(self, model="gpt-3.5-turbo-1106", system_prompt=None, detailed = True) -> None:
        self.detailed = detailed
        if system_prompt is None:
            if self.detailed:
                self.system_prompt = """
                                    - Please act as an expert in hardware design using Verilog or SystemVerilog. 
                                    - Explain the high-level functionality of the module, whose definition is provided below. 
                                    - Use as many high-level concepts that are directly applicable to describe the code of the whole design. 
                                    - Use text-based truth tables and state transition graphs when necessary. 
                                    - You are only required to describe the top module's functionality. 
                                    - Explicitly mention the specifications of inputs and outputs in terms of their bit-width, range, and any other constraints or considerations.
                                    - Pay special attention to the temporal logic of the signals; e.g., how the registers are updated, how the state machines transition, etc.
                                    - Pay attention that the logic to decide a signal state can be spread across different places in the code, be sure to note them all.
                                    - Assume your response will be used by an experienced hardware designer as the only basis for implementing the equivalent functionality and provide the same top module input/output interface as described in the code.
                                    - Assume the experienced hardware designer will implement all functionalities in just one module. 
                                    """
            else:
                self.system_prompt = """
                                    - Please act as an expert in hardware design using Verilog or SystemVerilog. 
                                    - Explain the high-level functionality of the module, whose definition is provided below. 
                                    - Use as many high-level concepts that are directly applicable to describe the code of the whole design. 
                                    - You are only required to describe the top module's functionality. 
                                    - Assume your description will even be understood by a non-hardware expert; or this non-hardware expert can deliver this description with reasonable amount of training.
                                    - Strict rule: Be very concise and high-level, avoid low-level details.
                                    """
        self.sft_data_gen_chain =  SimpleConverseChain(system_prompt=self.system_prompt, model=model, temperature=0.95, max_tokens=2048, top_p=0.95, have_memory=False, verbose=False)


    def gen_global_summary(self, 
                              code_string, 
                              block_summaries = None,
                              desc_key = "detail_description", 
                              example_code_description_file=None, 
                              example_code_strings={}):

        example_prompt = ""
        if example_code_description_file is not None and len(example_code_strings) > 0:
            #load the jsonl file
            description_df = pd.read_json(example_code_description_file, lines=True)
            description_df = description_df[description_df["task_id"].isin(example_code_strings.keys())]
            example_description_strings = {row["task_id"]: row[desc_key] for _, row in description_df.iterrows()}
            example_prompt = "Here are some example of explaining the code for your reference:\n"
            example_prompt = "```\n"
            example_id = 0
            for example_code_id, example_code_string in example_code_strings.items():
                example_prompt += "Example: {}\n".format(example_id)
                example_prompt += "Question: Explain the high-level functionality of the Verilog module.\n"
                example_prompt += example_code_string
                example_prompt += "Answer: "
                #append the description
                example_prompt += example_description_strings[example_code_id]
                example_prompt += "\n"
                example_id += 1
            example_prompt += "```\n"
        
        context_prompt = ""
        if block_summaries is not None:
            if len(block_summaries) > 0:
                context_prompt = "Here are the summaries of the code blocks to be explained:\n"
                for block_summary in block_summaries:
                    context_prompt += block_summary + "\n"
                context_prompt += "\n"
                
        user_prompt = "User Question: Explain the high-level functionality of the Verilog module.:\n"
        user_prompt += "```\n"
        user_prompt += code_string
        user_prompt += "```\n"
        
        #store the prompt in "global_summary_detailed_{self.detailed}.txt"
        with open("global_summary_detailed_{}.txt".format(self.detailed), "w") as f:
            f.write(example_prompt + "\n" + context_prompt + "\n" + user_prompt + "\n" + self.system_prompt + "\n" )

        response = self.sft_data_gen_chain.chat(example_prompt + "\n" + context_prompt + "\n" + user_prompt + "\n" + self.system_prompt + "\n" )
        return response
    
    def code_gen(self, description_file, eval_file, result_file, repeat=10):
        #description_file: jsonl
        #result_file: jsonl
        system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions. Only include code and nothing else."
        question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
        description_df = pd.read_json(description_file, lines=True)
        eval_df = pd.read_json(eval_file, lines=True)
        results = []
        #get the detail description 
        for idx, row in description_df.iterrows():
            for r in range(repeat):
                print("Processing task_id: {}".format(row["task_id"]))
                global_summary = row["detail_description"]
                module_header = eval_df[eval_df["task_id"] == row["task_id"]]["prompt"].values[0]
                problem_description = "\n\n" + global_summary + "\n\n Module header:\n\n"  + module_header + "\n"
                human_input = question_prompt + problem_description
                response = self.sft_data_gen_chain.chat(human_input, system_prompt=system_prompt)
                #append task_id and response
                results.append({"task_id": row["task_id"], "completion": response})
        #save the results
        pd.DataFrame(results).to_json(result_file, orient="records", lines=True)
        return results

def gen_block_summary_chain(model="llama2", temperature=0.7, max_tokens=1024):
    if "gpt" in model:
        llm = ChatOpenAI(max_retries=4, model=model, temperature=temperature, max_tokens=max_tokens, request_timeout=40)
    else:
        llm = HuggingFaceTextGenInference(
                        inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                        max_new_tokens=max_tokens,
                        repetition_penalty=1.17,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.1
                        )
    prompt = "You are helping me write high-level summaries of code blocks from a larger verilog code piece.\n"
    prompt += "Here is the module header \n\n```\n{module_header}\n```\n\n"
    prompt += "{block_doc_history}\n" 
    prompt += "Summarize the following next code block.\n block_{block_num} Code: \n\n```\n{doc1}\n```\n\n"
    prompt += "Be very specific, do not include code and avoid ambiguity.\n"
    if "gpt" not in model:
        prompt = llama2_prompt_without_memory_without_sys.format(human_input=prompt)

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["module_header", "block_doc_history", "block_num", "doc1"],
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False,
        output_parser=StrOutputParser()
    )
    
    return chain


def detail_steps_chain(model="llama2", temperature=0.7, max_tokens=256): 
    if "gpt" in model:
        llm = ChatOpenAI(max_retries=0, model=model, temperature=temperature, max_tokens=max_tokens, request_timeout=40)
    else:
        llm = HuggingFaceTextGenInference(
                        inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                        max_new_tokens=max_tokens,
                        repetition_penalty=1.17,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.1
                        )
    prompt = "You are helping me prepare instructions for a large language model to generate RTL code.\n"
    prompt += """
                For your context, here is the high-level summary of the code to be generated:\n\n```\n{high_level_summary}\n```\n\n 
                For your context, here is the module header for the code to be generated:\n\n```\n{module_header}\n```\n\n
                For your context, here is part of the commented code:\n\n```\n{commented_code}\n```\n\n
                Please provide the detailed steps as prompt to write the above part of the code, be very concise, do not include code, and do not include step number as prefix:
               """
    if "gpt" not in model:
        prompt = llama2_prompt_without_memory_without_sys.format(human_input=prompt)

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["high_level_summary", "module_header", "commented_code"],
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False,
        output_parser=StrOutputParser()
    )
    
    return chain


class Func_lookup(BaseModel):
    function_exist: bool = Field(description="If function/module instantiation exists")
    function_name: list = Field(description="List of string function/module name; empty list if it doesn't exist")

    @validator("function_name", pre=True)
    def function_name_validator(cls, v):
        if v is None:
            return []
        else:
            return v

def func_name_lookup_chain(model="llama2", temperature=0.7, max_tokens=128, language="verilog"):

    if "gpt" in model:
        cheap_model = ChatOpenAI(max_retries=2, model=model, temperature=temperature, max_tokens=max_tokens,request_timeout=40)
    else:
        cheap_model = HuggingFaceTextGenInference(
                        inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                        max_new_tokens=max_tokens,
                        )

    # gpt-3.5-turbo or gpt-4-0613
    expensive_model = ChatOpenAI(max_retries=0, model="gpt-4-1106-preview", temperature=temperature, max_tokens=max_tokens, request_timeout=40,
                                 model_kwargs={"response_format": { "type": "json_object" }})

    parser = PydanticOutputParser(pydantic_object=Func_lookup)
    prompt = ""
    prompt += "Here are lines of code and comments:\n"
    prompt += "```\n\n"
    prompt += "{doc}\n\n"
    prompt += "```\n"
    if language == "verilog":
        prompt += "Are there module instantiations? if yes, please provide the list of module names.\n"
    elif language == "c" or language =="cpp" or "hls" in language:
        prompt += "Are there function calls? if yes, please provide the list of function names.\n"
    prompt += '{format_instructions}\n'

    if "gpt" not in model:
        prompt = llama2_prompt_without_memory_without_sys.format(human_input=prompt)

    if "gpt" in model:
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["doc"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
    else:
        prompt = prompt.format(format_instructions="Format your answer in json format with function_exist and function_name as entries. function_name is a list of string function/module name; empty list if it doesn't exist.", doc = "{doc}")
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["doc"],
        )

    # llm = cheap_model.with_fallbacks([expensive_model])
    # chain = (
    #         {"doc": lambda x: x}
    #         | prompt_template
    #         | llm
    #         | parser
    #     )

    cheap_chain = LLMChain(
        llm=cheap_model,
        prompt=prompt_template,
        verbose=False,
        output_parser=parser
    )

    expensive_chain = LLMChain(
        llm=expensive_model,
        prompt=prompt_template,
        verbose=False,
        output_parser=parser
    )
    
    #TODO: potentially add outputparser retry instead for fallback: https://python.langchain.com/docs/modules/model_io/output_parsers/retry 
    return cheap_chain, expensive_chain


class SimpleConverseChain:
    def __init__(self, model="gpt-3.5-turbo-1106", temperature=0.7, max_tokens=256, top_p=1.0, 
                 system_prompt="", have_memory=True, memory=None, memory_length=5, 
                 output_parser=StrOutputParser(), fixing_chain=None,
                 customized_format_instructions=""" """, json_mode=False,
                 verbose=True):
        #default system prompt
        self.system_prompt = system_prompt
        self.output_parser = output_parser
        self.have_memory = have_memory
        self.model = model
        self.max_memory_length = memory_length
        self.verbose = verbose
        self.fixing_chain = fixing_chain
        self.customized_format_instructions = customized_format_instructions
        self.json_mode = json_mode
        
        #a customized buffer to keep track of the conversation for models other than gpt
        self.memory_buffer = []
        self.current_memory_length = 0
        
        if "gpt" not in model:
            assert "llama" in model.lower(), "model name should be llama or gpt series"

        #human and ai is actually good prefix
        #under the hood in langchain/chat_models/openai.py, they comply with openai's role naming
        #e.g., {'messages': [{'role': 'system', 'content': 'you are an AI, you are talking to a human'}, {'role': 'user', 'content': 'hello, who are you? I am user_name'}, {'role': 'assistant', 'content': 'Hello Luke, nice to meet you! I am an AI, specifically a language model designed to assist with various tasks and engage in conversations. How can I help you today?'}, {'role': 'user', 'content': 'who am I?'}], 'model': 'gpt-3.5-turbo', 'request_timeout': None, 'max_tokens': 256, 'stream': False, 'n': 1, 'temperature': 0.7, 'top_p': 1.0, 'api_key': 'your_openai_key_if_you_want_to_use_it', 'api_base': '', 'organization': ''}
        if self.have_memory:
            #same format for gpt series models
            self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("{system_message}"), # The persistent system prompt
                MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
            ])

            
            #designed only for gpt series models
            if memory is None:
                self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, input_key="human_input", k=memory_length)
            else:
                self.memory = memory


            if "gpt" in model:
                if not self.json_mode:
                    cheap_model = ChatOpenAI(max_retries=5, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, request_timeout=40)
                else:
                    cheap_model = ChatOpenAI(max_retries=5, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, request_timeout=40,  
                                             model_kwargs={"response_format": {"type": "json_object"}})
            else:
                cheap_model = HuggingFaceTextGenInference(
                                inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                                max_new_tokens=max_tokens,
                                # repetition_penalty=1.17,
                                # temperature=0.7,
                                # top_k=40,
                                # top_p=0.1
                                )
            if not self.json_mode:
                expensive_model = ChatOpenAI(max_retries=5, model="gpt-3.5-turbo-1106", temperature=temperature, max_tokens=max_tokens, top_p=top_p, request_timeout=40)
            else:
                expensive_model = ChatOpenAI(max_retries=5, model="gpt-3.5-turbo-1106", temperature=temperature, max_tokens=max_tokens, request_timeout=40,  
                                         model_kwargs={"response_format": { "type": "json_object" }})

            if "gpt" in model:
                #TODO: better to swap to expression format: https://python.langchain.com/docs/expression_language/cookbook/memory
                chat_llm_chain = LLMChain(
                    llm=cheap_model,
                    prompt=self.prompt,
                    verbose=verbose,
                    memory=self.memory,
                )
                chat_llm_chain_alternative = LLMChain(
                    llm=expensive_model,
                    prompt=self.prompt,
                    verbose=verbose,
                    memory=self.memory,
                )
            else:
                chat_llm_chain = LLMChain(
                    llm=cheap_model,
                    prompt=PromptTemplate(template=llama2_prompt_with_memory, input_variables=["system_message", "chat_history", "human_input"]),
                    verbose=verbose,
                )
                chat_llm_chain_alternative = LLMChain(
                    llm=expensive_model,
                    prompt=self.prompt,
                    verbose=verbose,
                    memory=self.memory,
                )


            self.chain = chat_llm_chain
            self.chain_alternative = chat_llm_chain_alternative
        else:

            self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("{system_message}"), # The persistent system prompt
                HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
            ])

            if "gpt" in model:
                if not self.json_mode:
                    cheap_model = ChatOpenAI(max_retries=5, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, request_timeout=40)
                else:
                    cheap_model = ChatOpenAI(max_retries=5, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p,  
                                             model_kwargs={"response_format": { "type": "json_object" }},
                                             request_timeout=40)
            else:
                cheap_model = HuggingFaceTextGenInference(
                                inference_server_url=os.environ.get("LLAMA_INFERENCE_SERVER_URL"),
                                max_new_tokens=max_tokens
                                )
            if not self.json_mode:
                expensive_model = ChatOpenAI(max_retries=5, model="gpt-3.5-turbo-1106", temperature=temperature, max_tokens=max_tokens, request_timeout=40)
            else:
                expensive_model = ChatOpenAI(max_retries=5, model="gpt-3.5-turbo-1106", temperature=temperature, max_tokens=max_tokens,  
                                         model_kwargs={"response_format": { "type": "json_object" }},
                                         request_timeout=40)
                
            if "gpt" in model:
                chat_llm_chain = LLMChain(
                    llm=cheap_model,
                    prompt=self.prompt,
                    verbose=verbose,
                )
                chat_llm_chain_alternative = LLMChain(
                    llm=expensive_model,
                    prompt=self.prompt,
                    verbose=verbose,
                )
            else:
                chat_llm_chain = LLMChain(
                    llm=cheap_model,
                    prompt=PromptTemplate(template=llama2_prompt_without_memory, input_variables=["system_message", "human_input"]),
                    verbose=verbose,
                )
                chat_llm_chain_alternative = LLMChain(
                    llm=expensive_model,
                    prompt=self.prompt,
                    verbose=verbose,
                )

            self.chain = chat_llm_chain
            self.chain_alternative = chat_llm_chain_alternative

    def chat(self, human_input, system_prompt=None):
        if self.have_memory:
            print("Memory buffer length: {}".format(len(self.memory_buffer)))
            print("Memory length: {}".format(len(self.memory.buffer)))
        if self.have_memory and "gpt" in self.model:
            if system_prompt is None:
                system_prompt = self.system_prompt
            old_converse_length = len(self.memory.buffer)
            try:
                response = self.chain.predict(system_message=system_prompt, human_input=human_input)
                response = self.output_parser.parse(response)
            except Exception as e:
                if self.verbose:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                new_converse_length = len(self.memory.buffer)
                if new_converse_length > old_converse_length:
                    #delete the new conversation
                    length_diff = new_converse_length - old_converse_length
                    self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-length_diff]
                else:
                    #probably something wrong with api call as no new response is generated
                    response = self.chain_alternative.predict(system_message=system_prompt, human_input=human_input)
                    response = self.output_parser.parse(response)
                    return response
                
                #first try to fix the output parsing
                try:
                    if self.fixing_chain is None:
                        print("Response failed: {}".format(response))
                        raise Exception("fixing_chain is None")
                    else:
                        response = self.fixing_chain.batch([response])[0]
                    #add the new conversation got deleted
                    self.memory.save_context({"human_input": human_input}, {"output": response})
                except Exception as e:
                    if self.verbose:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                    
                    #then use expensive model
                    response = self.chain_alternative.predict(system_message=system_prompt, human_input=human_input)
                    response = self.output_parser.parse(response)
        else:
            if system_prompt is None:
                system_prompt = self.system_prompt
            #used for falling back from non-gpt models to gpt models
            orig_human_input = copy.deepcopy(human_input)
            try:
                #models other than gpt have poor output format enforcement
                if "gpt" not in self.model:
                    #take out the output format
                    #TODO: fix this via format instructions mod to the form_commenting_prompt function in utils.py
                    #format instructions from langchain is a little too complicated
                    # human_input = human_input.split("The output should be formatted")[0].replace("The output should be formatted", "")
                    # human_input += self.customized_format_instructions
                    tmp_human = None
                    tmp_model = None
                    if self.have_memory:
                        #sync from memory to memory buffer
                        #Note that the messages are parsed already in the buffer and memory
                        for msg_idx, memory_message in enumerate(self.memory.buffer):
                            if msg_idx % 2 == 0:
                                #Human
                                tmp_human = memory_message.content
                            else:
                                tmp_msg = copy.deepcopy(llama2_memory_prompt)
                                #AI
                                tmp_model = memory_message.content
                                tmp_msg = tmp_msg.format(human_input=tmp_human, model_reply=tmp_model)
                                #add the new conversation
                                if len(self.memory_buffer ) == 0:
                                    self.memory_buffer.append(tmp_msg)
                                else:
                                    self.memory_buffer[int(msg_idx/2)] = tmp_msg
                        #unparsed response
                        response = self.chain.predict(system_message=system_prompt, human_input=human_input, chat_history="".join(self.memory_buffer))
                        #it like to use null ...
                        response = response.replace("null", '""')
                    else:
                        #unparsed response
                        response = self.chain.predict(system_message=system_prompt, human_input=human_input)
                        #it like to use null ...
                        response = response.replace("null", '""')
                else:
                    #unparsed response
                    response = self.chain.predict(system_message=system_prompt, human_input=orig_human_input)
                if self.verbose:
                    print("Response before parsing: {}".format(response))
                #parsing the response
                if "gpt" not in self.model:
                    #try using the ouput parser first
                    try:
                        response = response.replace("False", "false")
                        response = response.replace("True", "true") 
                        response = self.output_parser.parse(response)
                    except Exception as e:
                        if "Max retries exceeded" in str(e) or "Server error" in str(e):
                            print("Error: ", e)
                            exit()
                        if self.verbose:
                            print(e)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)   
                        if self.fixing_chain is None:
                            print("Response failed: {}".format(response))
                            raise Exception("fixing_chain is None")
                        else:
                            response = self.fixing_chain.batch([response])[0]
                    #currently both gpt and non-gpt models have the same input format in memory with output format enforcement listed
                    if self.have_memory:
                        new_message = llama2_memory_prompt.format(human_input=human_input, model_reply=response)
                        self.memory_buffer.append(new_message)
                        if self.current_memory_length < self.max_memory_length:
                            self.current_memory_length += 1
                        else:
                            self.memory_buffer = self.memory_buffer[1:]
                        #sync from memory buffer to memory
                        self.memory.chat_memory.messages.append(HumanMessage(content=human_input))
                        self.memory.chat_memory.messages.append(AIMessage(content=response))
                else:
                    #parsing the response
                    response = self.output_parser.parse(response)
            except Exception as e:
                if "Max retries exceeded" in str(e) or "Server error" in str(e):
                    print("Error: ", e)
                    exit()
                if self.verbose:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)   

                #first try to fix the output
                try:
                    if self.verbose:
                        print("Response before parsing / fixded: {}".format(response))
                    if self.fixing_chain is None:
                        print("Response failed: {}".format(response))
                        raise Exception("fixing_chain is None")
                    else:
                        response = self.fixing_chain.batch([response])[0]
                    #currently both gpt and non-gpt models have the same input format in memory with output format enforcement listed
                    if self.have_memory:
                        new_message = llama2_memory_prompt.format(human_input=human_input, model_reply=response)
                        self.memory_buffer.append(new_message)
                        if self.current_memory_length < self.max_memory_length:
                            self.current_memory_length += 1
                        else:
                            self.memory_buffer = self.memory_buffer[1:]
                        #sync from memory buffer to memory
                        self.memory.chat_memory.messages.append(HumanMessage(content=human_input))
                        self.memory.chat_memory.messages.append(AIMessage(content=response))
                except Exception as e:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    if self.verbose:
                        print("Response failed: {}".format(response))
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                    #then use expensive model
                    if self.have_memory:
                        old_converse_length = len(self.memory.buffer)
                    try:
                        response = self.chain_alternative.predict(system_message=system_prompt, human_input=orig_human_input)
                        response = self.output_parser.parse(response)
                    except Exception as e:
                        if self.have_memory:
                            new_converse_length = len(self.memory.buffer)
                            if new_converse_length > old_converse_length:
                                #delete the new conversation
                                length_diff = new_converse_length - old_converse_length
                                self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-length_diff]
                        response = self.chain_alternative.predict(system_message=system_prompt, human_input=orig_human_input)
                        response = self.output_parser.parse(response)
        return response


class VerilogEval:
    def __init__(self, model="gpt-3.5-turbo-1106", system_prompt=None) -> None:
        if system_prompt is None:
            self.system_prompt = "Explain the high-level functionality of the Verilog module. Use as many high-level concepts that are directly applicable to describe the code, say at the level of an undergraduate EECS major, but do not include extraneous details that aren't immediately applicable. Use text-based truth tables and state transition graphs when necessary. Speak concisely as if this was a specification for a circuit designer to implement. You should only reply with descriptive natural language and not use any code."
            # self.system_prompt = "Explain the high-level functionality of the Verilog module. Use as many high-level concepts that are directly applicable to describe the code, say at the level of an undergraduate EECS major, but do not include extraneous details that aren't immediately applicable. Use text-based truth tables and state transition graphs when necessary. You should only reply with descriptive natural language and not use any code."
            # self.sft_data_gen_chain =  SimpleConverseChain(system_prompt=self.system_prompt, 
            #                                                model=model, 
            #                                                temperature=0.7, 
            #                                                max_tokens=1024, 
            #                                                top_p=0.95, 
            #                                                have_memory=False, 
            #                                                verbose=False)
        self.sft_data_gen_chain =  SimpleConverseChain(system_prompt=self.system_prompt, model=model, temperature=0.95, max_tokens=512, top_p=0.95, have_memory=False, verbose=False)


    def verilog_eval_sft_data(self, code_string, desc_key = "detail_description", example_code_description_file=None, example_code_strings={}):
        #example_code_description_file: jsonl
        #example_code_strings: {"task_id": "shift18", "code_string": "module shift18(input [17:0] a, input [4:0] b, output [17:0] y);\n"}

        example_prompt = ""
        if example_code_description_file is not None and len(example_code_strings) > 0:
            #load the jsonl file
            description_df = pd.read_json(example_code_description_file, lines=True)
            description_df = description_df[description_df["task_id"].isin(example_code_strings.keys())]
            example_description_strings = {row["task_id"]: row[desc_key] for _, row in description_df.iterrows()}
            example_prompt = "Here are some example of explaining the code for your reference:\n"
            example_prompt = "```\n"
            example_id = 0
            for example_code_id, example_code_string in example_code_strings.items():
                example_prompt += "Example: {}\n".format(example_id)
                example_prompt += "Question: Explain the high-level functionality of the Verilog module.\n"
                example_prompt += example_code_string
                example_prompt += "Answer: "
                #append the description
                example_prompt += example_description_strings[example_code_id]
                example_prompt += "\n"
                example_id += 1
            example_prompt += "```\n"
        
        user_prompt = "User Question: Explain the high-level functionality of the Verilog module.:\n"
        user_prompt += "```\n"
        user_prompt += code_string
        user_prompt += "```\n"
        
        # Initialize the tokenizer (use the appropriate tokenizer for your model)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # or the appropriate model name
        
        
        user_prompt_tokens = tokenizer.encode(user_prompt)

        if len(user_prompt_tokens) > 15000:
            # Calculate the number of excess tokens
            excess_tokens = len(user_prompt_tokens) - 15000

            # Tokenize user_prompt and truncate it to remove the excess tokens
            truncated_user_prompt_tokens = user_prompt_tokens[:-excess_tokens]

            # Convert the truncated tokens back to string
            user_prompt = tokenizer.decode(truncated_user_prompt_tokens)

            user_prompt += "```\n"

        # Tokenize the combined prompt
        combined_prompt = example_prompt + user_prompt
        combined_tokens = tokenizer.encode(combined_prompt)
        
        # Check if the combined token length exceeds 16000
        if len(combined_tokens) > 16000:
            # Calculate the number of excess tokens
            excess_tokens = len(combined_tokens) - 16000
            
            # Tokenize example_prompt and truncate it to remove the excess tokens
            example_prompt_tokens = tokenizer.encode(example_prompt)
            truncated_example_prompt_tokens = example_prompt_tokens[:-excess_tokens]
            
            # Convert the truncated tokens back to string
            example_prompt_reduced = tokenizer.decode(truncated_example_prompt_tokens)

            example_prompt_reduced += "```\n"

            combined_prompt = example_prompt_reduced + user_prompt


        
        # Make the API call with the combined prompt
        response = self.sft_data_gen_chain.chat(combined_prompt)
        return response
    
    def code_gen(self, description_file, eval_file, result_file, repeat=10):
        #description_file: jsonl
        #result_file: jsonl
        system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions. Only include code and nothing else."
        question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
        description_df = pd.read_json(description_file, lines=True)
        eval_df = pd.read_json(eval_file, lines=True)
        results = []
        #get the detail description 
        for idx, row in description_df.iterrows():
            for r in range(repeat):
                print("Processing task_id: {}".format(row["task_id"]))
                global_summary = row["detail_description"]
                module_header = eval_df[eval_df["task_id"] == row["task_id"]]["prompt"].values[0]
                problem_description = "\n\n" + global_summary + "\n\n Module header:\n\n"  + module_header + "\n"
                human_input = question_prompt + problem_description
                response = self.sft_data_gen_chain.chat(human_input, system_prompt=system_prompt)
                #append task_id and response
                results.append({"task_id": row["task_id"], "completion": response})
        #save the results
        pd.DataFrame(results).to_json(result_file, orient="records", lines=True)
        return results




            







if __name__ == "__main__":
    import sys
    import os 
    from dotenv import load_dotenv
    load_dotenv()
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.environ.get("CHATBOT_BACKEND_DIR"),os.environ.get("SRC_DIR")))
    test_chain = func_name_lookup_chain(model="llama2")
    print(json.loads(test_chain[0].predict(doc='test_mod test_mod_inst (.clk(clk), .rst(rst)); \n adder my_adder(.clk(clk), .rst(rst));').json()))
    exit()
    print("testing converse chain:")
    test_chain = SimpleConverseChain(system_prompt="you are an AI, you are talking to a human", model="llama2", verbose=True)
    test_chain.memory.save_context({"human_input": "what'up"}, {"output": "good"})
    # print(type(test_chain.memory.chat_memory.messages[0].content))
    response = test_chain.chat("hello, who are you? I am user_name")
    print(response, type(response))
    response = test_chain.chat("Do you like GOT?")
    print(response, type(response))

