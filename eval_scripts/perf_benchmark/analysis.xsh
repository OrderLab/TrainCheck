$XONSH_SHOW_TRACEBACK = True

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--res_folder", type=str, required=True)
args = parser.parse_args()

res_folder = args.res_folder

# only need to handle the marco benchmark results
# list all the files in the folder
files = $(ls @(res_folder)).split()
files = [f for f in files if f.startswith("e2e_")]

"""
FORMAT OF THE DATA TO PRODUCE FOR E2E

task,method,overhead
MNIST,systrace,549.57
ResNet18,systrace,338.43
Transformer,systrace,205.34
MNIST,monkey-patch,148.22
ResNet18,monkey-patch,29.62
Transformer,monkey-patch,63.12
MNIST,selective,1.61
ResNet18,selective,1.07
Transformer,selective,1.17
"""

all_results = {}
for f in files:
    if "completion-time.csv" in f:
        continue
    series = np.loadtxt(f"{res_folder}/{f}")
    task = "_".join(f.split("_")[1:-1])
    method = f.split("_")[-1].split(".")[0]
    if task not in all_results:
        all_results[task] = {}
    # if series is not a list, convert it to a list
    if not isinstance(series, list):
        series = series.tolist()
    if not isinstance(series, list):
        series = [series]
    all_results[task][method] = series
    print(f"Task: {task}, method: {method}, len: {series}")

overhead_results = []
for task in all_results:
    assert "naive" in all_results[task], f"naive (base situtation) not found in {task}"
    min_len = len(all_results[task]["naive"])
    for method in all_results[task]:
        print(f"Task: {task}, method: {method}")
        min_len = min(len(all_results[task][method]), min_len)
        if method == "naive":
            continue
        overhead_series = np.array(all_results[task][method][:min_len]) / np.array(all_results[task]["naive"][:min_len])
        overhead = np.mean(overhead_series)
        std = np.std(overhead_series)
        overhead_results.append([task, method, overhead, std])

df = pd.DataFrame(overhead_results, columns=["task", "method", "overhead", "std"])
df
# dump to csv
df.to_csv(f"{res_folder}/overhead_e2e.csv", index=False)
