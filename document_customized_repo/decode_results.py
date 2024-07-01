import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="output dir where the results are stored")

    args = parser.parse_args()
    output_dir = args.output_dir

    metadatapath = "/documented_code/dataset_metadata/part0/global_high_level_summary.json"

    #load metadata
    with open(output_dir+metadatapath, "r") as f:
        metadata = json.load(f)
    #store "global_summary_high_level" to global_summary_high_level.txt
    with open("global_summary_high_level.txt", "w") as f:
        f.write(metadata["priority_encoder.v"]["global_summary_high_level"])
    #store "global_summary_detailed" to global_summary_detailed.txt
    with open("global_summary_detailed.txt", "w") as f:
        f.write(metadata["priority_encoder.v"]["global_summary_detailed"])


    #block metadata
    block_metadatapath = "/documented_code/dataset_metadata/part0/block_summary.json"
    #load block metadata
    with open(output_dir+ block_metadatapath, "r") as f:
        block_metadata = json.load(f)
    #store "block_summary" to block_summary.txt
    block_idx = 0
    with open( "block_summary.txt", "w") as f:
        for block_summary in block_metadata["priority_encoder.v"]["block_summary"]:
            f.write(f"Block {block_idx}: {block_summary}\n\n")
            block_idx += 1
    
    documented_code_path = "/documented_code/part0/priority_encoder/priority_encoder.v"
    #store "documented_code" to documented_code.v
    with open( "documented_code.v", "w") as f:
        with open(output_dir+ documented_code_path, "r") as f2:
            f.write(f2.read())
            




