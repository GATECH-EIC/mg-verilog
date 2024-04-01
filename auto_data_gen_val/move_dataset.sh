#!/bin/bash

mode=$1
part_num=$2

if [ "$mode" = 0 ]; then
    folder_to_remove=/home/user_name/DAC_2024/ckpts/test_10_30_${part_num}_complete
    echo "Removing $folder_to_remove"
    rm -rf $folder_to_remove
    mkdir $folder_to_remove
    echo "Copying assets to $folder_to_remove"
    cp -r assets $folder_to_remove
    echo "Copying code_vec_store to $folder_to_remove"
    cp -r ../code_vec_store $folder_to_remove
    echo "Copying documented_code to $folder_to_remove"
    cp -r documented_code $folder_to_remove
elif [ "$mode" = 1 ]; then
    assets_dir=/home/user_name/DAC_2024/ckpts/test_10_30_${part_num}_complete/assets
    code_vec_store_dir=/home/user_name/DAC_2024/ckpts/test_10_30_${part_num}_complete/code_vec_store
    echo "Copying assets from $assets_dir to assets"
    rm -rf assets
    cp -r $assets_dir assets
    # rm -rf ../code_vec_store
    # echo "Copying code_vec_store from $code_vec_store_dir to ../code_vec_store"
    # cp -r $code_vec_store_dir ../code_vec_store
    rm -rf documented_code/*
    ./clean.sh
fi
