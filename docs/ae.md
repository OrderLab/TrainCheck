# TrainCheck Artifact Evaluation Guide

Welcome to the artifact evaluation guide for **TrainCheck** (OSDI'25). This document outlines the procedures needed to reproduce our results and guides you through the key experiments presented in the paper.

> **Note:** We may update both the main TrainCheck repository and the evaluation workloads repository during the evaluation period.  
> Please make sure to **pull the latest version** of each repository before proceeding.

## ✅ Checklist

- [ ] Environment set up (Python, dependencies, 2 CUDA GPUs with ≥ 12GiB memory each)
- [ ] Installed `xonsh` via `pip3 install 'xonsh[full]'` in the conda environment
- [ ] Ran **[Silent Issue Detection](#eval-silent-issue-detection)** experiment
- [ ] Ran **[Invariant Transferability](#eval-transferability)** evaluation
- [ ] Ran **[False Positive Rate](#false-positive-rate)** evaluation
- [ ] Ran **[Performance Overhead](#eval-performance-overhead)** measurement
- [ ] Verified outputs match expected results (tolerances noted per experiment)

## 📎 Resources You Need

In addition to this guide, you will need the following resources throughout the evaluation process:

1. [**5-Minute Tutorial**](./5-min-tutorial.md) — A quick walkthrough that introduces TrainCheck’s workflow using a real-world bug.
2. [**TrainCheck Installation Guide**](./installation-guide.md) — Step-by-step instructions for setting up TrainCheck.
3. [**Technical Usage Guide**](./technical-doc.md) — Detailed documentation on how to use TrainCheck, configure instrumentation, and interpret outputs.
4. [**Evaluation Workloads Repository**](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads) — Contains all evaluation workloads and automation scripts used in the experiments.

## 1. Overview

**TrainCheck** is an invariant-based tool for detecting silent correctness issues in PyTorch training pipelines.

This artifact enables reproduction of the four main evaluation results from the paper:

- **[Silent Issue Detection (Section 5.1)](#eval-silent-issue-detection)**
- **[Invariant Transferability (Section 5.3)](#eval-transferability)**
- **[False Positive Rate (Section 5.4)](#false-positive-rate)**
- **[Performance Overhead (Section 5.5)](#eval-performance-overhead)**

To get familiar with TrainCheck, we recommend starting with the [**5-Minute Tutorial**](./5-min-tutorial.md), which walks you through detecting a real-world bug from Section 5.1.

### ⏱️ Recommended Evaluation Order

We suggest running the evaluations in the following order, based on automation level and runtime requirements:

1. Kick the tires – [5 min tutorial with TrainCheck](./5-min-tutorial.md)
2. Performance Overhead (~10 minutes)
3. False Positive Rate (~1.5 hours)
4. Transferability (~30 minutes)
5. Silent Issue Detection (~ variate, should be able to finish within one day)

## 2. Environment Requirements

Many of our experiment scripts are written in xonsh, a shell that combines Python and Bash.
Please install it with:

```bash
conda activate traincheck
pip3 install 'xonsh[full]'
```

For a full and efficient AE experience, we recommend the following setup:
- 🖥 1 machine with 2× CUDA-enabled GPUs
- Each GPU should have at least 12 GiB memory.
- Compatible with CUDA 11.8 or 12.1
- 🧠 32 host memory (recommended)

### 🔧 Recommended Hardware: Chameleon Cloud

Most experiments require **2× CUDA-enabled GPUs** with support for **CUDA 11.8+**. While some workloads can run on GPUs with as little as 2 GiB memory, the main experiments (e.g., Section 5.1) benefit from higher-capacity GPUs.

We recommend using the `compute_liqid` node type on [Chameleon Cloud](https://www.chameleoncloud.org):

- ✅ `liqid01` and `liqid02`:  
  These nodes each have **2× A100 GPUs (40 GiB)** and allow you to reproduce **all results** in the paper.

- 🆗 Other `compute_liqid` nodes with **1× A100 GPU**:  
  These are sufficient for all **single-GPU experiments** and let you reproduce **~90%** of results.

Please consult the estimated runtimes in each evaluation section before making reservations.  
⏱️ If working full-time on the artifact, **2 days should be sufficient**, but we recommend reserving **at least 5 days** to allow for possible setup delays or debugging.

### Software Notes

1. If you’re using Chameleon instances:
    - Please start your machine with an Ubuntu 22.04 image that includes recent GPU drivers.
    - (TBD: we will provide the specific image ID or setup instructions once finalized.)

2. Follow [Installation Guide](./installation-guide.md) to install TrainCheck.

⏭️ Once your environment is set up, we recommend starting with the [5-Minute Tutorial with TrainCheck](./5-min-tutorial.md).
It will help you get familiar with the workflow and also verify that your installation is working correctly.

## Eval: Silent Issue Detection

📌 *This section is currently under preparation.*  
We are working on providing automated environment setup scripts for each specific issue detected by TrainCheck to streamline reproduction.

We expect this section to be completed within a week.  
If you finish all other experiments before then, please let us know—we’ll prioritize releasing this part so you can proceed without delay.

## Eval: False Positive Rate

⏳ Estimated Completion Time: 2 hour.
- Trace Collection: ~10 minutes
- Invariant Inference & Checking: ~1.5 hours

### 🎯 Goal

This evaluation measures the false positive rate of alarms reported by TrainCheck's invariants.  
The target results are discussed in the main text of **Section 5.4** of the paper.

### 📂 Resources & Scripts

- **Automation Scripts**:
  - [`TrainCheck-Evaluation-Workloads/fp_rate/ae_fp.py`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/fp_rate/ae_fp.py): The script to collect traces, perform invariant inference, and check invariants on supposedly-correct programs to see if there are any false alarms.
  - [`TrainCheck-Evaluation-Workloads/fp_rate/compute_fp_rate.py`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/fp_rate/compute_fp_rate.py): The script to compute false positive rates from the invariant checking results.

- **Workloads**:
  - The evaluation uses official PyTorch training pipelines located at [`TrainCheck-Evaluation-Workloads/fp_rate/workloads`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/tree/main/fp_rate/workloads).
    We have shortened the training runs for faster execution.
    For AE purposes, you do not need to modify or understand the workload code—`ae_fp.py` will automatically handle the entire process.

### 🛠 How to Run

1. Make sure you have a working TrainCheck installation by following [TrainCheck Installation Guide](./installation-guide.md).

> All steps described below assumes you are already in the `TrainCheck-Evaluation-Workloads` repo. If not, clone the repository and go to it.
> ```bash
> git clone https://github.com/OrderLab/TrainCheck-Evaluation-Workloads.git
> cd TrainCheck-Evaluation-Workloads
> ```

2. Install necessary dependencies for the false positive evaluation workloads.
    ```bash
    conda activate traincheck # change this if you installed TrainCheck in a different environment.
    cd fp_rate
    pip3 install -r requirements.txt
    ```

3. Execute `ae_fp.py` to collect traces, perform invariant inference, and check the invariants on validation programs.

    The workload `ddp-multigpu` will need 2 GPUs. We have provided the trace for `ddp-multigpu` in case you do not have two GPUs.

    If you need to use our pre-computed trace for `ddp-multigpu`, remove the `--overwrite-existing-results` argument.
    ```bash
    python3 ae_fp.py --bench workloads
    ```

    Or, if you have a machine with 2 GPUs, execute the below command, such that the original results will be re-computed.
    ```bash
    python3 ae_fp.py --bench workloads --overwrite-existing-results
    ```

4. Execute `compute_fp_rates.py` to compute the false positive rates.

    ```bash
    python3 compute_fp_rates.py
    ```

### What to Expect During Execution

The `ae_fp.py` script is long running. It performs three tasks at same time. 
1. It collects trace for all the workloads.
2. It infers invariants for three setups in Section 5.4.
3. It checks inferred invariants on the validation workloads.

The experiments might fail if environment installation issues or disruption happens. When you run into problems, please refer to [⚠️ Notes & Troubleshooting](#️-notes--troubleshooting).

### ⚠️ Notes & Troubleshooting

The script will automatically detect any errors in any (1) trace collection, (2) inference tasks, (3) checking tasks. If you encounter any trace collection issues, please check for any missing environment dependencies.

If you encounter any issues on invariant inference tasks or invariant checking tasks, please try to rerun the experiment by adding `--overwrite-existing-results` or delete all `trace_*` folders except for `trace_ddp-multigpu`.

If you see persistent issues, it will likely be a environment issue or software bug. Please contact us for help.

### How to verify the results?

The `compute_fp_rates.py` script generates a file called `fp_rates.csv` under the current directory. Looking like this

```csv
setup,fp_rate
1-input,0.3105
4-input,0.1127
6-input,0.1066
```

These values correspond to the results reported in Section 5.4 of the paper.
You should verify that the false positive rates are similar or lower. Since the OSDI submission, we have fixed multiple bugs in TrainCheck, so the false positive rates are expected to be significantly lower in most cases.

In our run of the script, we obtained the following results:
```csv
setup,fp_rate
1-input,0.0387
4-input,0.0143
6-input,0.0119
```

## Eval: Transferability

⏳ **Estimated Completion Time**: 40 minutes
- Environment Setup: ~10 minutes  
- Trace Collection: ~10 minutes  
- Invariant Inference: ~20 minutes

### 🎯 Goal

This evaluation measures the **transferability** of invariants inferred by TrainCheck across library versions and training environments.  
The results to be reproduced correspond to the final paragraph of **Section 5.3** of the paper.

Other claims in Section 5.3—specifically, that invariants inferred from reference pipelines can detect all known bugs—are validated as part of the [Silent Issue Detection Evaluation](#eval-silent-issue-detection).

### 📂 Resources & Scripts

- **Automation Script**:  
  - [`transferability/ae_transferability.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/ae_transferability.sh) Runs the full transferability evaluation pipeline described in Section 5.3 of the paper. It executes invariant inference, applies inferred invariants to other pipelines, and collects applicability (invariant should be checked and not cause false alarms) statistics.
  - [`transferability/install-traincheck-torch251-cu121.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/install-traincheck-torch251-cu121.sh) Creates a conda environment named traincheck-torch251 with Python 3.10 and installs TrainCheck from the latest GitHub version.
  - [`transferability/install-traincheck-torch251-cu118.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/install-traincheck-torch251-cu118.sh) Same as above but installs the CUDA 118 version of PyTorch 2.5.1.

This evaluation uses the **GCN** training pipeline from PyTorch's official examples, tested across different PyTorch versions.  
The pipeline is included in the artifact repository and will be automatically handled by the script—no manual setup is required.

### 🛠 How to Run

1. Go to [TrainCheck-Evaluation-Workloads/transferability](`https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/tree/main/transferability`). Clone the repo if you do not have it.
    ```bash
    git clone https://github.com/OrderLab/TrainCheck-Evaluation-Workloads.git
    cd TrainCheck-Evaluation-Workloads/transferability
    ```

2. Create a new conda environment named `traincheck-torch251`, and install **PyTorch 2.5.1** along with TrainCheck.  

    Run the appropriate script based on your GPU's CUDA compatibility (likely executing either will be fine):
    ```bash
    bash install-traincheck-torch251-cu121.sh  # for CUDA 12.1
    ```
    or
    ```bash
    bash install-traincheck-torch251-cu118.sh  # for CUDA 11.8
    ```

3. Run the transferability evaluation script:
    ```bash
    bash ae_transferability.sh
    ```

    This script will:
	  - Collect traces from the GCN training pipeline using both PyTorch 2.2.2 and 2.5.1.
	  - Infer invariants from the 2.2.2 version.
	  - Apply them to the 2.5.1 trace to assess transferability.

### ✅ How to Verify the Results

After the script finishes, it generates a file named `applied_rates.csv` that reports the percentage of applicable invariants. You should verify that the rate is no lower than the paper’s reported value:

> 🟢 "94.2% remain valid and applicable up to PyTorch 2.5.1" (Section 5.3)

### ⚠️ Notes & Troubleshooting

If invariant inference or checking fails, please first verify that the environment is correctly set up (e.g., correct PyTorch version, dependencies installed).  
Then try re-running `ae_transferability.py`.

If the issue persists, please contact us for assistance。

## Eval: Performance Overhead

⏳ Estimated Completion Time: 10 minutes.

### 🎯 Goal

This evaluation measures the runtime overhead introduced by TrainCheck’s instrumentation compared to un-instrumented runs across a set of representative ML workloads, during the invariant checking stage. The results correspond to Section 5.5 of the paper.

### 📂 Resources & Scripts

> Files described below are all in the [TrainCheck-Evaluation-Workloads](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/) repo.

- Automation Scripts:
  - [`performance_overhead/ae_perf.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/performance_overhead/ae_perf.sh): End-to-end script for running the performance overhead benchmarks (Section 5.5) and generating Figure 7. It internally calls:
    - `run_all.xsh`: Runs the experiments and collects raw data (per-iteration duration).
    - `analysis.xsh`: Analyzes the raw data and prepares input for plotting.
    - `plot_e2e.py`: Plots the final results.
  
- Workloads (You won't need to touch this):
    - Located in [overhead-e2e](../eval_scripts/perf_benchmark/overhead-e2e)

- The deployed 100 invariants:
    [eval_scripts/perf_benchmark/overhead-e2e/sampled_100_invariants.json](../eval_scripts/perf_benchmark/overhead-e2e/sampled_100_invariants.json)


### 🛠 How to Run

1. Make sure you have a working TrainCheck installation by following [TrainCheck Installation Guide](./installation-guide.md).

> All steps described below assumes you are already in the `TrainCheck-Evaluation-Workloads` repo. If not, clone the repository and go to it.
> ```bash
> git clone https://github.com/OrderLab/TrainCheck-Evaluation-Workloads.git
> cd TrainCheck-Evaluation-Workloads
> ```

2. Execute `ae_perf.sh`.

    ```bash
    conda activate traincheck
    cd performance_overhead

    bash ae_perf.sh
    ```

### Expected Output

After execution completes, a plot will be generated at `performance_ae.pdf`. All the raw data are stored at a folder named `perf_res_ae`.

### ✅ How to Verify

- Open the generated file performance_ae.pdf and compare it against Figure 7 in the paper.
- Small differences in the overhead numbers (within ±20%) are expected.
TrainCheck’s overhead is sensitive to CPU performance, since trace serialization is blocking and CPU-bound.
- Despite minor variations, the key takeaway should remain clear:
TrainCheck’s selective instrumentation incurs significantly lower overhead compared to other methods.

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
