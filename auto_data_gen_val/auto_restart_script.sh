#!/bin/bash
source /home/user_name/init_conda.sh
conda activate tvm
while true; do echo -e "n\nn\nn\nn\nn\nn\nn\nn\n" | python line_by_line_comments_gen.py && break; done
