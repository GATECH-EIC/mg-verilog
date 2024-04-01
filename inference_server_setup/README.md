# codellm

This is the example of using codellama within langchain framework

## Env Setup

```
conda create -n codellm python==3.9
conda activate codellm
conda install langchain -c conda-forge
pip install langchain[all]
pip install huggingface_hub
pip install git+https://github.com/huggingface/transformers.git@main accelerate
```

## Set Huggingface cache dir and access token

By default, huggingface will use ~/.cache/huggingface/ for cache datasets and models. However, in some servers, you only have limited space in home dir or you want this cache stored in a folder that can be shared among different servers. In such cases, you need to set your huggingface cache dir manully.

```
export HF_HOME=/path/to/cache/directory
export HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

You can also add the above cmd to your bashrc, if you want to set it permanently.

## Setup HuggingFace Inference Server

```
model=codellama/CodeLlama-34b-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model
```

## Run the test example

```
python test.py
```