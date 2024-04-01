wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod u=rxw Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh

echo y | conda create -n aigchip-sft python=3.8
conda activate aigchip-sft


echo y | conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install jsonlines
pip install accelerate
pip install pandas 
pip install fire 
pip install transformers[torch]
pip install peft
pip install bitsandbytes
pip install accelerate
pip install scipy
pip install sentencepiece 
pip install transformers[dev]
pip install deepspeed
pip install transformers[deepspeed]


echo y | conda install -c conda-forge gperf flex bison
echo y | sudo apt install -y autoconf gperf make gcc g++ bison flex
git clone https://github.com/steveicarus/iverilog.git 
cd iverilog
autoconf 
./configure
make 
sudo make install
cd ..

pip install -e verilog_eval
cd model_eval_qlora_kevin/
mkdir data
cd .. 


git clone https://github.com/PyHDI/Pyverilog.git
cd Pyverilog
git checkout 1.3.0
sudo apt install graphviz libgraphviz-dev
pip install jinja2 ply
pip install pytest pytest-pythonpath  
pip install pygraphviz
python3 setup.py install
cd ..

