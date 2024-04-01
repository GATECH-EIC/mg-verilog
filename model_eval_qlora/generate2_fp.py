import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from torch import nn
import numpy as np
import math

from transformers import AutoTokenizer, CodeGenForCausalLM, GenerationConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
import pandas as pd

import jsonlines

sys.path.append("../verilog_eval/verilog_eval")
from evaluation import evaluate_functional_correctness
from datetime import datetime
from tqdm import tqdm

from accelerate import PartialState, Accelerator
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from tokenizers import AddedToken

import warnings
from packaging import version
from packaging.version import parse


# PROMPT = (
#     "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
#     "\n\nImplement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated.\n\n"
#     "\n\n{description}\n\nModule header:\n{module_header}\n"
# )


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

system_prompt = "You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions."
question_prompt = "Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
problem_description = "\n\n {description} \n\n Module header:\n\n {module_header}\n"
PROMPT = llama2_prompt_without_memory.format(system_message = system_prompt, human_input = question_prompt + problem_description)

def mem():
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    return torch.cuda.memory_allocated(0)/1024/1024/1024

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

import importlib
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from os.path import exists, join, isdir


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    if "codellama" in model.name_or_path:
        for key, value in special_tokens_dict.items():
            special_tokens_dict[key] = AddedToken(value, lstrip=True, rstrip=True, normalized=False)
            
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_accelerate_model(model_name, checkpoint, bf16, fp16):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{24000}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        print("distributed", local_rank)
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir="/home/user_name/HF_cache/",
        load_in_4bit=False,
        load_in_8bit=False,
        device_map=device_map,
        max_memory= max_memory,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=False,
        #     load_in_8bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
        torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),
        trust_remote_code=False,
        # use_auth_token=False
        token=False
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))

    # Tokenizer
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/home/user_name/HF_cache/",
        padding_side="left",
        use_fast=True, # Fast tokenizer giving issues.
        tokenizer_type="code_llama",
        trust_remote_code=False,
        # use_auth_token=False,
        token=False
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    if "codellama" in model_name:
        tokenizer.add_special_tokens({
                "eos_token": AddedToken(tokenizer.convert_ids_to_tokens(model.config.eos_token_id), lstrip=True, rstrip=True, normalized=False),
                "bos_token": AddedToken(tokenizer.convert_ids_to_tokens(model.config.bos_token_id), lstrip=True, rstrip=True, normalized=False),
                "unk_token": AddedToken(tokenizer.convert_ids_to_tokens(
                    # model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                    tokenizer.pad_token_id
                ), lstrip=False, rstrip=False, normalized=False),

        })
        tokenizer.add_tokens(
                    [AddedToken(tokenizer.eot_token, lstrip=True, rstrip=True, normalized=False),
                    AddedToken(tokenizer.prefix_token, lstrip=True, rstrip=True, normalized=False),
                    AddedToken(tokenizer.suffix_token, lstrip=True, rstrip=True, normalized=False),
                    AddedToken(tokenizer.middle_token, lstrip=True, rstrip=True, normalized=False),
                    AddedToken(tokenizer.pad_token, lstrip=True, rstrip=True, normalized=False),
                    ],
                    special_tokens=True
        ) 
    elif 'llama' in model_name or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    # model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                    tokenizer.pad_token_id
                ),
        })  
    
    # if not args.full_finetune:
    if True:
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # if not args.full_finetune:
    if True:
        if checkpoint is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint, 'adapter_model'), is_trainable=True)
        else:
            raise Exception("Only loading checkpoint is allowed.")
            # print(f'adding LoRA modules...')
            # modules = find_all_linear_names(args, model)
            # config = LoraConfig(
            #     r=args.lora_r,
            #     lora_alpha=args.lora_alpha,
            #     target_modules=modules,
            #     lora_dropout=args.lora_dropout,
            #     bias="none",
            #     task_type="CAUSAL_LM",
            # )
            # model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def load(
    model_name: str = None,
    model_type: str = "hf",
    base_model: str = None,
    bf16: bool = False,
    fp16: bool = False,
    dev: str=None
):
    start_time = time.time()

    if model_type == "hf":
        model = CodeGenForCausalLM.from_pretrained(model_name, cache_dir="/home/user_name/HF_cache/")

        if dev is not None:
            model = model.to(dev)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    else:
        print("Loading base model", base_model, "Checkpoint", model_name)
        model, tokenizer = get_accelerate_model(base_model, model_name, bf16, fp16)
        # model.to(dev)

    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

class VerilogDataset(torch.utils.data.Dataset):
  def __init__(self, descriptions, headers, tokenizer, device, k=20):
    self.device = device
    self.tokenizer = tokenizer

    self.prompts = []
    for desc in descriptions:
      for i in range(k):
        self.prompts.append({
          "task_id": desc["task_id"],
          "description": desc["detail_description"] if pd.notna(desc["detail_description"]) else desc["simple_description"],
          "module_header": headers[desc["task_id"]]
        })

  def __len__(self):
    return len(self.prompts)

  def __getitem__(self, index):
    prompt = PROMPT.format_map(self.prompts[index])
    return torch.tensor(self.tokenizer(prompt).input_ids).to(self.device)

def main(
    model_name: str = None,
    model_type: str = "hf",
    base_model: str = None,
    bf16: bool = True,
    fp16: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.75,
    sample_k: int = 1,
    desc_file: str = "../verilog_eval/descriptions/VerilogDescription_Machine.jsonl",
    eval_file: str = "../verilog_eval/data/VerilogEval_Machine.jsonl",
    result_name: str = None,
    batch_size: int = 1,
    skip_gen: bool = False
):

    with torch.no_grad():
        accel = Accelerator()
        accel.print("Starting at", timestamp())

        accel.print("Loading model...")
        # device = torch.device("cuda:0")
        device = accel.device
        # device = None
        # model, tokenizer = load(model_name, model_type, base_model=base_model, bf16=bf16, fp16=fp16, dev=device)
        model, tokenizer = load(model_name, model_type, base_model=base_model, bf16=bf16, fp16=fp16)
        model.eval()

        # fix bf16 issue
        if bf16 and "llama" in base_model.lower():
            print("Fixing bf16")

            class RMSWrapper(nn.Module):
                def __init__(self, actual):
                    super().__init__()
                    self.actual = actual
                
                def __call__(self, hidden_states):
                    init_dtype = hidden_states.dtype
                    hidden_states = self.actual(hidden_states)
                    return hidden_states.to(init_dtype)

            llama_model = model.base_model.model.model
            llama_model.norm = RMSWrapper(llama_model.norm)
            for layer in llama_model.layers:
                layer.input_layernorm = RMSWrapper(layer.input_layernorm)
                layer.post_attention_layernorm = RMSWrapper(layer.post_attention_layernorm)


        accel.print("Loading descriptions...")
        tasks = pd.read_json(path_or_buf=desc_file, lines=True)

        headers = {}
        with jsonlines.open(eval_file) as file:
            for obj in file:
                headers[obj["task_id"]] = obj["prompt"]


        # clear temp file
        with open("./data/gen.jsonl", "w") as file:
            pass


        outputs = []
        accel.print("Starting...")

        with accel.split_between_processes(tasks.to_dict(orient="records")) as prompts:
            # print("Distr", len(prompt), [r["task_id"] for r in prompt])
            tasks = prompts

        # create new dataset for each process to help with prompt batch balancing
        task_ids = [t["task_id"] for t in tasks]
        dataset = VerilogDataset(tasks, headers, tokenizer, device, k=sample_k)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        accel.print("Starting generation...")

        # for ind, task in tqdm(tasks.iterrows(), total=len(tasks)):
        # for i in tqdm(range(math.ceil(len(tasks) / batch_size)), total=math.ceil(len(tasks) / batch_size), desc="Batch"):
        for i, prompts in enumerate(tqdm(dataloader, total=len(dataloader), desc="Process " + str(accel.process_index))):
            # start = i * batch_size
            # end = start + batch_size

            # ids = []
            # prompts = []
            # headlens = []

            # for task in tasks[start:end]:
            #     tid = task["task_id"]
            #     ids.append(tid)
            #     desc = task["detail_description"] if pd.notna(task["detail_description"]) else task["simple_description"]
            #     header = headers[tid]

            #     prompt_full = PROMPT.format_map({"description": desc, "module_header": header})
            #     prompts.append(prompt_full)

            #     header = torch.tensor([tokenizer(header).input_ids], dtype=torch.int64).to(device)
            #     headlens.append(header.shape[1])

            # tok = tokenizer(prompts, padding=True)

            # prompts = torch.tensor(tok.input_ids).to(device)
            # masks = torch.tensor(tok.attention_mask).to(device)

            prompts = prompts["input_ids"]

            output = model.generate(
                inputs=prompts,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )

            strings = tokenizer.batch_decode(output[:, prompts.shape[1]:], skip_special_tokens=True)
            full_responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            prompt_strings = tokenizer.batch_decode(prompts[:, 30:], skip_special_tokens=True)

            for j, comp in enumerate(strings):
                tid = (i * batch_size + j) // sample_k
                obj = {
                    "process": accel.process_index,
                    "task_id": task_ids[tid],
                    "completion": comp,
                    "prompt": prompt_strings[j],
                    "full_response": full_responses[j],
                }
                outputs.append(obj)

            # for j in range(sample_k):
            #     qout = output_list[j][:, prompts.shape[1]:]
            #     strings = tokenizer.batch_decode(qout, skip_special_tokens=True) 
            #     for id, row in enumerate(strings):
            #         obj = {
            #             "process": accel.process_index,
            #             "task_id": ids[id],
            #             "completion": row
            #         }
            #         outputs.append(obj)
                    
        with jsonlines.open("./data/gen.jsonl", mode="a") as writer:
            print(timestamp(), "starting write", accel.process_index, "length", len(outputs))
            writer.write_all(outputs)
            print(timestamp(), "done writing", accel.process_index)


        # evaluate
        accel.wait_for_everyone()

        if accel.is_main_process:
            accel.print("running eval")
            res = evaluate_functional_correctness("./data/gen.jsonl", problem_file=eval_file, k=[1,5,10])
            accel.print("Eval Results:", res)

            with open("./data/results.txt", mode="a") as f:
                ts = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' | ' + result_name if result_name is not None else ''}] "
                f.write(ts + str(res) + "\n")

if __name__ == "__main__":
    fire.Fire(main)
