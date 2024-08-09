import jsonlines
import sys
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append("../verilog_eval/verilog_eval")
from evaluation import evaluate_functional_correctness

def process_jsonl_file(src_file, dst_file):
    with jsonlines.open(src_file) as reader:
        with jsonlines.open(dst_file, mode='w') as writer:
            for obj in reader:
                split = obj['completion'].split(';', 1)
                if len(split) > 1:
                    obj['completion'] = split[1]
                    writer.write(obj)
                else:
                    writer.write(obj)



def evaluate(gen_file, prob_file):
    res = evaluate_functional_correctness(gen_file, problem_file=prob_file, k=[1,5,10])
    print("Eval Results:", res)

def results_profile(result_file, prob_file):
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    passed_list = []
    failed_list = []
    with jsonlines.open(result_file) as reader:
        for obj in reader:
            if obj['passed']:
                passed_list.append(obj)
            else:
                failed_list.append(obj)

    problems = {}
    with jsonlines.open(prob_file) as reader:
        for obj in reader:
            problems[obj['task_id']] = obj
    
    for obj in passed_list:
        obj['module_header'] = problems[obj['task_id']]['prompt']
        obj['canonical_solution'] = problems[obj['task_id']]['canonical_solution']
        obj["code_lines"] = len(obj['module_header'].split('\n')) + len(obj['canonical_solution'].split('\n'))
        obj["code_token_count"] = len(tokenizer.encode(obj["module_header"] + "\n" + obj["canonical_solution"]))
        obj["prompt_token_count"] = len(tokenizer.encode(obj["prompt"]))

    for obj in failed_list:
        obj['module_header'] = problems[obj['task_id']]['prompt']
        obj['canonical_solution'] = problems[obj['task_id']]['canonical_solution']
        obj["code_lines"] = len(obj['module_header'].split('\n')) + len(obj['canonical_solution'].split('\n'))
        obj["code_token_count"] = len(tokenizer.encode(obj["module_header"] + "\n" + obj["canonical_solution"]))
        obj["prompt_token_count"] = len(tokenizer.encode(obj["prompt"]))



    data1 = [obj["code_token_count"] for obj in passed_list]
    data2 = [obj["code_token_count"] for obj in failed_list]
    data3 = data1 + data2


    # Plotting the distributions
    sns.set(style="whitegrid")  # Setting the style of the plot
    plt.figure(figsize=(10, 6))  # Setting the size of the plot
    #bin size 10 
    sns.histplot(data1, kde=True, color="blue", label="Passed", bins=10)
    sns.histplot(data2, kde=True, color="red", label="Failed", bins=10)

    plt.title('Distribution of Code Token Count')
    plt.xlabel('Code Token Count')
    plt.ylabel('Frequency')
    #save figure
    plt.savefig("passed_code_token_count.png")
    plt.clf()



    # Define common bin edges
    bins = np.linspace(min(np.min(data) for data in [data1, data2, data3]), 
                    max(np.max(data) for data in [data1, data2, data3]), 
                    10)
    # Calculate histograms
    hist1, _ = np.histogram(data1, bins=bins)
    hist2, _ = np.histogram(data2, bins=bins)
    hist3, _ = np.histogram(data3, bins=bins)
    # Normalize histograms
    normalized_hist1 = hist1 / (hist3 + 1e-6)  # Adding a small constant to avoid division by zero
    normalized_hist2 = hist2 / (hist3 + 1e-6)
    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(bins[:-1], normalized_hist1, label='Normalized Dataset 1', marker='o', color="blue")
    plt.plot(bins[:-1], normalized_hist2, label='Normalized Dataset 2', marker='o', color="red")

    plt.title('Success / Failure Rates')
    plt.xlabel('Code Token Count')
    plt.ylabel('Success / Failure Rates')
    #save figure
    plt.savefig("success_failure_rates.png")
    plt.clf()



    
        


if __name__ == '__main__':
    prob_file = "../verilog_eval/data/VerilogEval_Machine.jsonl"
    gen_file = "./data/gen.jsonl" 
    result_file = "./data/gen.jsonl_results.jsonl"
    #process_jsonl_file(gen_file, "test.jsonl")
    evaluate(gen_file=gen_file, prob_file=prob_file)
    #results_profile(result_file, prob_file)
