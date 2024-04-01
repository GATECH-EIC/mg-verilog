#!/bin/bash
./move_dataset.sh 1 0
sed -i 's/code_part = [0-9]*/code_part = 0/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 0

./move_dataset.sh 1 1
sed -i 's/code_part = [0-9]*/code_part = 1/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 1

./move_dataset.sh 1 2
sed -i 's/code_part = [0-9]*/code_part = 2/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 2

./move_dataset.sh 1 3
sed -i 's/code_part = [0-9]*/code_part = 3/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 3

./move_dataset.sh 1 4
sed -i 's/code_part = [0-9]*/code_part = 4/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 4

./move_dataset.sh 1 5
sed -i 's/code_part = [0-9]*/code_part = 5/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 5

./move_dataset.sh 1 6
sed -i 's/code_part = [0-9]*/code_part = 6/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 6

./move_dataset.sh 1 7
sed -i 's/code_part = [0-9]*/code_part = 7/' test_10_30.py
./auto_restart_script.sh
./move_dataset.sh 0 7

# ./move_dataset.sh 1 8
# sed -i 's/code_part = [0-9]*/code_part = 8/' test_10_30.py
# ./auto_restart_script.sh
# ./move_dataset.sh 0 8

# ./move_dataset.sh 1 9
# sed -i 's/code_part = [0-9]*/code_part = 9/' test_10_30.py
# ./auto_restart_script.sh
# ./move_dataset.sh 0 9


