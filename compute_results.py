import os 
import numpy as np
from utils import eval_output
from tabulate import tabulate
import sys 
import argparse 
import pandas as pd

parser = argparse.ArgumentParser(description='Compute results')
parser.add_argument('--model', type=str, default="gpt4", help='model to compute results for')
args = parser.parse_args()

results = []
avg_per_method = {} 
total_file_numbers = 0 

main_dir = "outputs"
data_full = {
    "method": [],
    "dataset": [], 
    "accuracy": [],
}

for file_name in os.listdir(f"{main_dir}/output_{args.model}"): 
    dataset = file_name.split("_")[0]
    method = file_name.split("_")[1:]
    method = "_".join(method)

    if method not in avg_per_method: 
        avg_per_method[method] = []


    acc = 0 
    total = 0 
    count_equal = 0
    for line in open(f"{main_dir}/output_{args.model}/{file_name}", "r").readlines(): 
        
        _, out, answer  = line.strip().split(",")
        acc += eval_output(out, answer)
        total += 1

    acc = acc/total
    results.append([dataset, method, acc, total])    
    avg_per_method[method].append(acc)
    total_file_numbers += 1

    data_full["method"].append(method)
    data_full["dataset"].append(dataset)
    data_full["accuracy"].append(acc)



results = sorted(results, key=lambda x: x[0])
print(tabulate(results, headers=["Dataset", "Method", "Accuracy", "Total"]))

avg_per_method = [(method, np.mean(avg_per_method[method])) for method in avg_per_method if method.split("_")[-1] != "no.txt"]
avg_per_method = sorted(avg_per_method, key=lambda x: x[1], reverse=True)

print(tabulate(avg_per_method, headers=["Method", "Accuracy"]))
print(f"Total number of files: {total_file_numbers}")

