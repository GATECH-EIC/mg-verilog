# MG-Verilog: Multi-grained Dataset Towards Enhanced LLM-assisted Verilog Generation
This is a repository for Data4AIGChip, an automated framework for data generation and validation, designed to enhance the fine-tuning of Large Language Models (LLMs) in accelerator code generation. This repository encompasses the following key components
- Dataset generation
- Supervised fine-tuning
- Collected datasets and fine-tuned model checkpoints


## Environment Setup

### Basic Environment
For a seamless setup experience, we strongly recommend utilizing our provided setup script to install all necessary dependencies. Please note that this script requires ```sudo``` privileges for installing the Icarus Verilog simulator. In case you lack ```sudo``` access, you will need to manually install the simulator. Additionally, adjust the script to accommodate a prefix installation approach. 

The script also leverages PyVerilog for dataset pre-processing. This may necessitate installing some extra dependencies. For detailed information, please refer to [pyverilog](https://github.com/PyHDI/Pyverilog/tree/develop). PyVerilog's [parser](https://github.com/PyHDI/Pyverilog/blob/develop/pyverilog/vparser/parser.py#L2357C1-L2373C27) can cause unwanted behavior in multi-processing. please refer to the [change here to patch](./imgs/pyverilog_patch.png). Our evaluation flow, which assesses model performance, is adapted from [VerilogEval](https://github.com/NVlabs/verilog-eval), using their sample descriptions and metric calculation and tailored to our specific requirements.
```
$ ./setup.sh
```

### Inference Server and Model API Key
For data generation, design creation, and validation purposes, we employ the `LLAMA2` and `openai-gpt` models. Ensure you have acquired the necessary credentials: `OPENAI_API_KEY` and `HUGGINGFACEHUB_API_TOKEN`.

For setting up the inference server, detailed guidance is provided in our dedicated section. Please refer to [inference_server_setup](./inference_server_setup/README.md) for comprehensive steps and tips


## Quick Run with Customized Raw Source Code Repo
Setup API key and inference server in [.env](auto_data_gen_val/.env). Default to OpenAI APIs for convenience (No need for local inference server if you only use OpenAI APIs).


`cd document_customized_repo`

`./document_customized_repo.sh test_dir output_test`

Replace `test_dir` with your own Verilog code source repo. Currently it does not support nested directories. For resume functionality, please follow the following detailed instructions (Uncertainty in LLM generation can cause the line-by-line comment output format check fail and require resuming if you do not want to lose the progress in this repo).

## Dataset Generation

### Raw Dataset Pre-processing
Begin by pre-processing the raw dataset, sourced from [Benchmarking Large Language Models for Automated Verilog RTL Code Generation](https://huggingface.co/datasets/shailja/Verilog_GitHub). This dataset comprises over 100k code-only samples from open-source implementations.

Prior to initiating this process, ensure that the setup has been fully completed. The pre-processing relies on PyVerilog and Icarus Verilog (and iverilog) to parse the raw dataset. This will take a while to finish.
```
$ cd auto_data_gen_val/preprocess_data/process_data
$ python preprocess.py ./raw_dataset_output_path/ # This will generate the pre-processed dataset with correct syntax and token length limit (default to 1024)
$ cd ../../
```

To facilitate more flexible description generation later and to organize metadata, such as module dependencies, we will partition the dataset into distinct parts.
```
$ python utils.py \
    --src_code_dir ./preprocess_data/process_data/raw_dataset_output_path/ \
    --src_code_metadata_file ./preprocess_data/process_data/module_inst.json \
    --output_dir ./partitioned_dataset_output_path/ \
    --shared_lib_dir ./directory_to_store_common_modules/ \
    --output_code_metadata_dir ./output_dir_for_code_metadata/ \
    --output_code_metadata_file codes.json \
    --module_to_task_id_map_file ./preprocess_data/process_data/module_name_to_task_id_mapping.json
```

### PoT Dataset Generation

Proceeding to the next step, we will generate the Pyramid-of-Thoughts (PoT) dataset. The primary model used will be `LLAMA2-70B-Chat`, with `openai-gpt3.5-turbo/4` serving as fallback models. These model choices are configurable to suit your needs. Ensure that the `OPENAI_API_KEY` and `LLAMA_INFERENCE_SERVER_URL` are correctly set up in the `./auto_data_gen_val/.env` file. All other settings should remain at their default values. ***Before proceeding, ensure that the earlier pre-processing step has been completed.***

#### Line-by-Line Comments Generation

This process generates line-by-line comments for all the partitioned datasets. Upon completion of each partition, you will receive a prompt asking whether to overwrite the existing intermediate results. Enter `y` to overwrite them. Initially, summaries will be generated for the shared library modules.

Be aware that there are instances where the `LLAMA2` model may not adhere to the required output format (i.e., JSON for straightforward parsing). Concurrently, the fallback `openai-gpt` models may also fail, either due to format enforcement issues or unstable API service. In such scenarios, the generation process will terminate with an error, necessitating a manual re-run for that specific partition. During this, type `n` to bypass the existing intermediate results.

Alternatively, the [Auto Restart Script](./auto_data_gen_val/auto_restart_script.sh) can be modified for automatic re-initiation of the process. If you choose the automatic restart approach, remember to manually update the starting partition number in the script. Under the current settings and assuming the inference server is hosted on 8xA5000 GPUs, the generation rate is approximately 1000 code samples every 12 hours.
```
$ python line_by_line_comments_gen.py \
    --total_parts 10 \
    --output_dir ./documented_code \
    --src_code_dir ./partitioned_dataset_output_path/ \
    --code_metadata_dir ./output_dir_for_code_metadata/ \
    --code_lib_path ./directory_to_store_common_modules/ \
    --code_vec_store ../code_vec_store/test/ \
    --discard_original_comment
```

#### Block Summary Generation

This step involves generating block summaries based on the line-by-line documented code. By default, code blocks are partitioned based on the number of lines, with a cap of 10 lines per block for the current dataset. Character-based partitioning is available by setting the `split_by_line` flag to `False`. However, empirical evaluations suggest that line-based partitioning tends to be more effective.

For generating these summaries, we recommend using `openai-gpt` models due to their capacity for handling larger token lengths.
```
$ python gen_block_summaries.py 0 10 \
    --code_metadata_dir ./output_dir_for_code_metadata/ \
    --documented_code_dir ./documented_code \
    --block_line_length 10 \
    --model gpt-3.5-turbo-1106
```
#### Global Summary Generation

This phase focuses on generating detailed and high-level global summaries for the dataset. It is required to complete the block summary generation prior to this step. The detailed global summary is derived from the block summaries and line-by-line commented code, whereas the high-level global summary is based on the detailed global summary.
```
$ python gen_global_summary.py 0 10 \
    --documented_code_dir ./documented_code \
    --model gpt-3.5-turbo-1106 \
    --detailed
$ python gen_global_summary.py 0 10 \
    --documented_code_dir ./documented_code \
    --model gpt-3.5-turbo-1106
```
#### Packaging the Dataset

This script packages the dataset into `description` and `code` pairs. The `descriptions` include `Detailed Global Summary`, `High-Level Global Summaries`, and `Block Summaries`. Additionally, a merged dataset encompassing all three types of description-code pairs will be created. By default, these datasets are saved to `./packaged_dataset/`.
```
$ python dataset_utils.py \
    --doced_dataset_dir ./documented_code \
    --total_part 10 \
    --packaged_dir ./packaged_dataset \
    --package_detailed_description \
    --package_simple_description \
    --package_llm2_block_summary_to_pure_code_one_shot_dataset \
    --package_merged_dataset
```

### Preparing the Benchmark

The following code will generate various description types for the benchmark code samples in VerilogEval, which are adapted from [HDLBits](https://hdlbits.01xz.net/wiki/Main_Page)
```
#prepare the src code from VerilogEval problem file
$ python verilog_eval_to_part_data.py \
    --eval_file ../verilog_eval/data/VerilogEval_Machine.jsonl \
    --data_dir ./benchmark_code_files/ \
    --meta_data_dir ./benchmark_metadata_files/
    
#generate line-by-line comments
$ python line_by_line_comments_gen.py \
    --total_parts 1 \
    --output_dir ./benchmark_documented_code \
    --src_code_dir ./benchmark_code_files/ \
    --code_metadata_dir ./benchmark_metadata_files/ \
    --code_lib_path ./benchmark_code_files/ \
    --code_vec_store ../benchmark_code_vec_store/test/ \
    --skip_supplement_summary \
    --discard_original_comment
#generate block summaries
$ python gen_block_summaries.py 0 1 \
    --documented_code_dir ./benchmark_documented_code \
    --block_line_length 10 \
    --model gpt-3.5-turbo-1106
#generate global summaries
$ python gen_global_summaries.py 0 1 \
    --documented_code_dir ./benchmark_documented_code \
    --model gpt-3.5-turbo-1106 \
    --detailed
$ python gen_global_summaries.py 0 1 \
    --documented_code_dir ./benchmark_documented_code \
    --model gpt-3.5-turbo-1106
#package the dataset
$ python dataset_utils.py \
    --doced_dataset_dir ./benchmark_documented_code \
    --total_part 1 \
    --packaged_dir ./benchmark_packaged_dataset \
    --package_hdlbits_global_summary_description_file \
    --package_hdlbits_block_summary_description_file
```

### Dataset Validation

This script validates the dataset by utilizing the generated descriptions to reverse-generate the code, which is then compiled using Icarus Verilog (iverilog). GPT-4 is employed for this reverse code generation process. Empirical observations suggest that it is most effective to apply validation only to the 'Detailed Global Summary' datasets, leaving the other datasets unchanged to ensure a more diverse training set.

Due to the intermittent stability of the OpenAI API service with multi-processing, this feature is currently disabled. Please note that validation may be time-consuming due to the extensive size of the dataset.

```
$ python code_validate.py \
    --dataset_dir ./packaged_dataset/detailed_description_dataset \
    --output_dir ./packaged_dataset/detailed_description_dataset_val
```

## Supervised Fine-Tuning

We utilize the PoT dataset for model fine-tuning and performance evaluation. The objective is to use various description formats within the PoT dataset for code generation. As with dataset and benchmark generation, we have prepared data in description-code pairs. The packaged dataset can be seamlessly integrated into the qlora framework with minimal adaptation. Ensure that dataset paths in the script are correct, and set a unique checkpoint path for each training run to prevent accidental overwriting. 

***The dataset paths in the script assumes that the previous dataset generation steps have been completed. Otherwise, please use the given links at the end of this README to download the packaged datasets and modify the paths accordingly.***

### Training Process
- To train on different dataset formats, use the `--dataset_dir` flag.
- The default settings are configured for 4 GPUs. Adjust the `CUDA_VISIBLE_DEVICES` and `WORLD_SIZE` according to your setup. Correspondingly, inversely scale the `--gradient_accumulation_steps` based on the number of GPUs.
- Typically, a training loss below 0.01 or completion of 10 epochs is indicative of sufficient training for achieving decent pass rates.

```
cd ..
cd sft_code
./train.sh 
```

### Evaluation Process

To evaluate models trained on different dataset formats, modify the `--checkpoint_dir` parameter. You can evaluate the trained models against various description formats by altering `--desc_file`, `--desc_key`, and `--prompt_type`. Below is an example of how to implement these changes for evaluation:

On high-level global summaries:
```
--desc_file ../auto_data_gen_val/benchmark_packaged_dataset/hdlbits_description_simple_description.jsonl \
--desc_key simple_description \
--prompt_type baseline
```

On detailed global summaries:
```
--desc_file ../auto_data_gen_val/benchmark_packaged_dataset/hdlbits_description_detailed_description.jsonl \
--desc_key detailed_description \
--prompt_type baseline
```

On block summaries:
```
--desc_file ../auto_data_gen_val/benchmark_packaged_dataset/hdlbits_for_llm2_eval.jsonl \
--desc_key block_to_code_description \
--prompt_type llm2_block_to_code
```

***By using the example settings above, you can evaluate the performance of the models fine-tuned on different dataset formats in PoT structure.***


Launch the evaluation:
```
cd ..
cd model_eval_qlora
./gen.sh
```

## Collected Datasets and fine-tuned model checkpoints
Datasets: [drive_link](https://drive.google.com/drive/folders/1NJHFGX1wgGZV8pky3W7sUkjNf2pTHxfx?usp=sharing)

Model checkpoints: [drive_link](https://drive.google.com/drive/folders/184TAdFog46g6lPvoWpNsGcaA_-3ygkPt?usp=sharing)


## Citation

Please cite with the following format; formal format will be updated after publication or release:

```
@inproceedings{zhang2024mgverilog,
  title={{MG-Verilog:} Multi-grained Dataset Towards Enhanced LLM-assisted Verilog Generation},
  author={Zhang, Yongan and Yu, Zhongzhi and Fu, Yonggan and Wan, Cheng and Lin, Yingyan (Celine)},
  booktitle={The First IEEE International Workshop on LLM-Aided Design (LAD'24)}, 
  year={2024}
}
```
