# TrainCheck Artifact Evaluation Guide

Welcome to the artifact evaluation guide for **TrainCheck** (OSDI'25). This document will help you reproduce the key experiments presented in the paper.

We provide pre-collected traces and pre-inferred invariants to simplify and speed up reproduction. Full instructions cover both end-to-end runs and shortcut paths using prepared data.

## ✅ Quick Checklist

- [ ] Environment set up (Python, dependencies, 2 CUDA GPUs with ≥ 12GiB memory each)
- [ ] (*Optional*) Downloaded pre-collected / pre-computed data
- [ ] Ran **[Silent Issue Detection](#eval-silent-issue-detection)** experiment
- [ ] Ran **[False Positive Rate](#false-positive-rate)** evaluation
- [ ] Ran **[Transferability](#eval-transferability)** evaluation
- [ ] Ran **[Performance Overhead](#eval-performance-overhead)** measurement
- [ ] Verified outputs match expected results (tolerances noted per experiment)

## 1. Overview

**TrainCheck** is an invariant-based tool for detecting silent correctness issues in PyTorch training pipelines.

This artifact allows you to reproduce the major 4 evaluation results presented in the paper.

- [ ] Ran **[Silent Issue Detection (Section 5.1 and 5.2)](#eval-silent-issue-detection)** experiment
- [ ] Ran **[False Positive Rate (Section 5.3)](#false-positive-rate)** evaluation
- [ ] Ran **[Transferability (Section 5.4)](#eval-transferability)** evaluation
- [ ] Ran **[Performance Overhead (Section 5.5)](#eval-performance-overhead)** measurement

### ⏱️ Recommended Evaluation Order

We suggest running the evaluations in the following order, based on automation level and runtime requirements:
1. Performance Overhead

    Fully automated (~1.5 hours)

2. False Positive Rate / Transferability 

    Automated
    - ~6 hours for trace collection (skippable)
	- ~48 hours for invariant inference (skippable)
	- ~6 hours for invariant checking

    We also provide pre-collected data, allowing you to skip or only partially run the trace collection / invariant inference stage.

3. Silent Issue Detection
    
    Partially automated — requires manual inspection to confirm that reported invariant violations correspond to true silent bugs.

Before starting the evaluation, we encourage you to go through the [**5 min tutorial with TrainCheck**](./5-min-tutorial.md) that provides some basic concepts about TrainCheck and walks you through using TrainCheck workflows, making you more familiar with our artifact as well.

## 2. What to Expect and Resources Provided

We aim TrainCheck to be open source project that can be used by the community, and thus we try hard to provide you and general developers with all the resources you need. 

Our evaluations are usually long-running, but we provide mostly end-to-end scripts that automate the process. These scripts will also automate generation for plots / tabular data.

- **Code Repository**: [GitHub - OrderLab/TrainCheck](https://github.com/OrderLab/traincheck)
- **Technical Documentation & FAQ**: Detailed guides covering usage, algorithms, data formats, caveats, and more.
- [**5 min tutorial with TrainCheck**](./5-min-tutorial.md): A quick walkthrough to get started with core workflows.
- 📦 **Experiment Automation Scripts & Workloads**: locations will be introduces in each experiment's specific evaluation.
- ⚡ **Pre-Collected Data**: TBD

Proceed to [Environment Setup](#2-environment-setup) to get started.

## Env Setup

For a full and efficient AE experience, we recommend the following setup:
- 🖥 1 machine with 2× CUDA-enabled GPUs
- Each GPU should have at least 12 GiB memory
- Compatible with CUDA 11.8 or 12.1
- 🧠 200 GiB host memory (recommended)
- Required for invariant inference and checking when running with high parallelism

### Recommended Hardware (Chameleon Cloud)

We recommend using the following hardware on the TACC cluster via Chameleon Cloud:
- Nodes: `liqid01` or `liqid02` of type `compute_liqid`. These nodes have 2 A100 per machine and reserving either of them should suffice.
    
⚠️ If you’re unable to reserve such machines, you may use a single GPU node and:
- Use our pre-collected traces for distributed workloads (see experiment sections), or
- Contact us for help replicating specific evaluations.

⚠️ ⚠️ If you do not have access to any machine with CUDA GPUs, please contact us.

### Software Notes

1. If you’re using Chameleon instances:
    - Please start your machine with an Ubuntu 22.04 image that includes recent GPU drivers.
    - (TBD: we will provide the specific image ID or setup instructions once finalized.)

2. Follow [Installation Guide](./installation-guide.md) to install TrainCheck.

⏭️ Once your environment is set up, we recommend starting with the [5-Minute Tutorial with TrainCheck](./5-min-tutorial.md).
It will help you get familiar with the workflow and also verify that your installation is working correctly.

## Eval: Silent Issue Detection

## Eval: False Positive Rate

## Eval: Transferability

## Eval: Performance Overhead

⏳ Estimated Completion Time: 1.5 hour.

### 🎯 Goal

This evaluation measures the runtime overhead introduced by TrainCheck’s instrumentation compared to uninstrumented runs across a set of representative ML workloads, during the invariant checking stage. The results correspond to Section 5.5 of the paper.


### 📂 Resources & Scripts

- Automation Script: 
  - `eval_scripts/perf_benchmark/run_all.xsh`: run the experiments and collect data.
  - `eval_scripts/perf_benchmark/analysis.xsh`: analyze raw data and produce input for the plot script.
  - `eval_scripts/perf_benchmark/plot_e2e.py` and `eval_scripts/perf_benchmark/plot_micro.py`: plot the figures in Section 5.5.
  
- Workloads (You probably won't need to touch this ):
    - Located in [overhead-e2e](../eval_scripts/perf_benchmark/overhead-e2e) and [overhead-micro](../eval_scripts/perf_benchmark/overhead-micro)
	- No pre-collected data is required—this evaluation runs end-to-end automatically and is pretty light weight

- Deployed 100 invariants:
    [eval_scripts/perf_benchmark/overhead-e2e/sampled_100_invariants.json](../eval_scripts/perf_benchmark/overhead-e2e/sampled_100_invariants.json)


### 🛠 How to Run

1. Navigate to the performance benchmark directory:
    ```bash
    cd eval_scripts/perf_benchmark/
    ```

2. Run the full benchmark suite using:
    ```bash
    xonsh eval_scripts/perf_benchmark/run_all.xsh
    ```
This script will:
- Execute each workload in three modes:
    - No instrumentation
	- TrainCheck selective instrumentation with 100 invariants deployed
	- Python settrace baseline (a lightweight instrumentation baseline)
- Measure per-iteration training time.
- Save raw results in a folder named: `perf_eval_res_<commit_hash>`

You should then execute the below commands that analyze the data and produce plots.
```bash
xonsh analysis.xsh --res_folder perf_eval_res_<commit_hash>

python3 plot_e2e.py -o perf_eval_res_<commit_hash>/macro.pdf -i perf_eval_res_<commit_hash>/overhead_e2e.csv -t <commit_hash>

python3 plot_micro.py -o perf_eval_res_<commit_hash>/micro.pdf -i perf_eval_res_<commit_hash>/wrapper_overhead_micro.csv -t <commit_hash>
```

### Expected Output
Key files in `perf_eval_res_<commit_hash>`:
- `overhead_e2e.csv` and `marco.pdf` data and plot for benchmarks presented in Section 5.5.
- `wrapper_overhead_micro.csv` and `micro.pdf`: data and plot for the pure wrapper overhead on individual APIs.

### ✅ How to Verify
	•	Check that the overhead percentages in overhead_results.csv are consistent with those reported in Section 5.5.
	•	Variations (within ±15% TODO confirm) are expected due to runtime and hardware differences.


### ⚠️ Notes & Troubleshooting
1. **Do Not Run Other GPU Tasks in Parallel**

    For stable performance measurements, the evaluation scripts will periodically terminate all CUDA processes to ensure a clean environment. 
    Please avoid running any other GPU workloads during this evaluation.

2. **Handling Failed Workloads**

    If an end-to-end workload fails:
    - Navigate to the corresponding workload folder.
    - Manually rerun it using:
    ```bash
    traincheck-collect --use-config --config md-config-var.yml -i ../sampled_100_invariants.json
    ```
	- If the issue does not reproduce consistently, simply delete the result folder and rerun the full benchmark.
	- If the failure is consistent, please contact us for support.