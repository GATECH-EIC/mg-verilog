#!/bin/bash
source /home/user_name/init_conda.sh
conda activate tvm
#first argument start_id
#second argument end_id

echo "python $1 $2 $3"
while true; do python $1 $2 $3 && break; done
