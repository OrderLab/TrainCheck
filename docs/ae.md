# TrainCheck Artifact Evaluation Guide

Welcome to the artifact evaluation guide for **TrainCheck** (OSDI'25). This document outlines the procedures needed to reproduce our results and guides you through the key experiments presented in the paper.

> **Note:** We may update both the main TrainCheck repository and the evaluation workloads repository during the evaluation period.  
> Please make sure to **pull the latest version** of each repository before proceeding.

## ✅ Checklist

- [ ] Environment set up (Python, dependencies, 2 CUDA GPUs with ≥ 12GiB memory each)
- [ ] Installed `xonsh` via `pip3 install 'xonsh[full]'` in the conda environment
- [ ] Ran **[Silent Issue Detection](./ae-eval-s5.1-silent-issue-detection.md)** experiment
- [ ] Ran **[Invariant Transferability](./ae-eval-s5.3-transferability.md)** evaluation
- [ ] Ran **[False Positive Rate](./ae-eval-s5.4-fp-rate.md)** evaluation
- [ ] Ran **[Performance Overhead](./ae-eval-s5.5-perf-overhead.md)** measurement
- [ ] Verified outputs match expected results (tolerances noted per experiment)

## 📎 Resources You Need

In addition to this guide, you will need the following resources throughout the evaluation process:

1. [**5-Minute Tutorial**](./5-min-tutorial.md) — A quick walkthrough that introduces TrainCheck’s workflow using a real-world bug.
2. [**TrainCheck Installation Guide**](./installation-guide.md) — Step-by-step instructions for setting up TrainCheck.
3. [**Technical Usage Guide**](./technical-doc.md) — Detailed documentation on how to use TrainCheck, configure instrumentation, and interpret outputs.
4. [**Evaluation Workloads Repository**](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads) — Contains all evaluation workloads and automation scripts used in the experiments.

## Overview

**TrainCheck** is an invariant-based tool for detecting silent correctness issues in PyTorch training pipelines.

This artifact enables reproduction of the four main evaluation results from the paper:

- **[Silent Issue Detection (Section 5.1)](./ae-eval-s5.1-silent-issue-detection.md)**
- **[Invariant Transferability (Section 5.3)](./ae-eval-s5.3-transferability.md)**
- **[False Positive Rate (Section 5.4)](./ae-eval-s5.4-fp-rate.md)**
- **[Performance Overhead (Section 5.5)](./ae-eval-s5.5-perf-overhead.md)**

To get familiar with TrainCheck, we recommend starting with the [**5-Minute Tutorial**](./5-min-tutorial.md), which walks you through detecting a real-world bug from Section 5.1.

### ⏱️ Recommended Evaluation Order

We suggest running the evaluations in the following order, based on automation level and runtime requirements:

1. Kick the tires – [5 min tutorial with TrainCheck](./5-min-tutorial.md)
2. Performance Overhead (~10 minutes)
3. False Positive Rate (~1.5 hours)
4. Transferability (~30 minutes)
5. Silent Issue Detection (~ variate, should be able to finish within one day)

## Environment Requirements

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
    - We recommend using the `CC-Ubuntu22.04-CUDA` OS image.

2. Follow [Installation Guide](./installation-guide.md) to install TrainCheck.

⏭️ Once your environment is set up, we recommend starting with the [5-Minute Tutorial with TrainCheck](./5-min-tutorial.md).
It will help you get familiar with the workflow and also verify that your installation is working correctly.

## 🚀 Kick-the-Tires: Try TrainCheck in 5 Minutes

Get started quickly by using TrainCheck to detect and diagnosis a real-world bug report: [PyTorch-FORUM-84911](https://discuss.pytorch.org/t/obtaining-abnormal-changes-in-loss-and-accuracy/84911).

See details in [5-min-tutorial](./5-min-tutorial.md).

## 📊 Start Full Artifact Evaluation

Follow the below specific instructions to reproduce our evaluation results:

1. [Section 5.5: Performance Overhead](./ae-eval-s5.5-perf-overhead.md)
2. [Section 5.4: False Positives](./ae-eval-s5.4-fp-rate.md)
3. [Section 5.3: Invariant Transferability](./ae-eval-s5.3-transferability.md)
4. [Section 5.1: Silent Issue Detection](./ae-eval-s5.1-silent-issue-detection.md)

