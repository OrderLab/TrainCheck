# Eval: Silent Issue Detection

⏳ **Estimated Completion Time**: ~5 hours (if running everything from scratch)

## 🎯 Goal

In our evaluation, **TrainCheck detects 18 silent issues**. Your goal is to reproduce and validate:

1. ✅ TrainCheck successfully detects each issue.  
2. ⏱️ Detection occurs within **at most 1 iteration** of the issue being triggered.  
3. 🔍 The reported invariant violations are **close to the root cause**.

To accomplish this, a full evaluation involves:
1. **Inferring invariants** from clean PyTorch example pipelines.
2. **Running each buggy pipeline** to produce a trace.
3. **Checking invariants** against these traces.
4. **Manually verifying** that each violation corresponds to a silent issue, is timely, and aligns with the root cause.

## 🛠️ Evaluation Adjustments for Reproducibility

To ease your evaluation effort, we’ve made the following simplifications:

### 1. 🧪 Pre-provided Bug-Detecting Invariants
We provide a curated set of invariants that are known to catch the issues.  
➡️ This eliminates the need to manually infer and filter a large number of candidate invariants.  
(You may optionally rerun inference yourself to verify reproducibility.)

### 2. 📦 Pre-collected Buggy Traces
Some bugs require:
- Complex library setups (e.g., DS-1801 requires source builds of HuggingFace's own fork of `Megatron-DeepSpeed` and a DeepSpeed installation that needs you to manually modify a few lines of code.)
- Large datasets (~100 GiB+)

To avoid these barriers:
- We **pre-collected traces** for all 18 bugs.
- You can run TrainCheck directly on these traces without reproducing the full training environments.

> Note: We have setup instructions in the `README.md` doc within each bug's folder. Please let us know if you are following these instructions to collect the traces yourself and have encountered any issues.

### 3. ⏱️ Early Bug Triggering + Ground Truth Iterations
We modified the buggy scripts to **trigger the silent issue as early as possible** and documented the **exact iteration** when each bug manifests.  
You can verify the provided iteration number by inspecting the buggy code or logs as desired.

## 📂 Resources & Scripts

> Files described below are all in the [TrainCheck-Evaluation-Workloads](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/) repo.

- **Automation Scripts**  
  Scripts for running TrainCheck to check invariants on buggy traces.

- **Pre-collected Traces**  
  Traces collected from buggy pipelines to avoid complex reproduction steps.

- **Curated Invariants List**  
  A small, hand-picked set of invariants that are known to detect each bug effectively.

- ***[Optional]* Reproduction Scripts & Environment Setup**
  Provided for each bug (in its respective folder) in case you want to collect your own trace.  
  This includes environment installation instructions and original buggy training scripts.

- ***[Optional]* Example Pipelines for Invariant Inference**
  Clean training pipelines used for inferring relevant invariants prior to checking.