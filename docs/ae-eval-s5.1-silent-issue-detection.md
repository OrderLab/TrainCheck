# Eval: Silent Issue Detection

⏳ **Estimated Completion Time**: ~30 minutes

## 🎯 Goal

TrainCheck detects **18 real-world silent issues** in our evaluation. Your goal in this artifact evaluation is to **verify detection for the subset of issues that are currently AE-supported** (see [bug table](#-bug-summary-table) below).

For each supported bug, you should confirm:

✅ **TrainCheck successfully detects the issue** by reporting one or more invariant violations on the provided trace.

The artifact provides all necessary resources to automate this confirmation.  
Additional insights—such as when the issue is triggered and how the violation aligns with the root cause—can be explored by examining the scripts, logs, or violation reports, though they are not required for core validation.

## 📂 Resources Provided

All files are located in the [`TrainCheck-Evaluation-Workloads`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads) repository.

| Resource | Description |
|---------|-------------|
| **Curated Invariants** | Small set of known-effective invariants per bug. |
| **Pre-collected Traces** | Captured execution traces from the buggy pipelines. |
| **Silent Issue Reproduction Scripts and Descriptions** | https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/tree/main/silent-issue-detection/bug-reprod-scripts | 

### 🐛 Silent Issue Summary Table

| **Bug ID**                | **Failure Location** | **AE?** | **AE Limitation (if any)**                                     |
|---------------------------|----------------------|--------|------------------------------------------------------------------|
| `baichuan2-86`            | HW/Driver            | ✅ Yes | Similar root cause as pytorch-84803, reuses pytorch-104336 trace                     |
| `deepspeed-1801`          | Framework            | ✅ Yes |                                                                  |
| `deepspeed-5794`          | Framework            | ❌ No  | Invariant relation still under evaluation                        |
| `lightning-thunder-725`   | Framework            | ✅ Yes |                                                                  |
| `mmpretrain-702`          | Framework            | ✅ Yes |                                                                  |
| `pytorch-51800`           | Framework            | ✅ Yes |                                                                  |
| `pytorch-84803`           | HW/Driver            | ✅ Yes | Different root cause, but low-level manifest is similar, reuses pytorch-104336 trace |
| `pytorch-96600`           | HW/Driver            | ✅ Yes | Similar root cause as pytorch-84803 reuses pytorch-104336 trace                      |
| `pytorch-104336`          | Framework            | ✅ Yes |                                                                  |
| `pytorch-115607`          | Compiler             | ✅ Yes |                                                                  |
| `pytorch-forum-84911`     | User Code            | ✅ Yes |                                                                  |
| `stackoverflow-60335387`  | User Code            | ✅ Yes |                                                                  |
| `stackoverflow-67180955`  | Framework            | ❌ No  | Requires older Python version no longer supported                |
| `transformers-17877`      | Framework            | ✅ Yes |                                                                  |
| `transformers-23723`      | Framework            | ✅ Yes |                                                                  |
| `transformers-33844`      | Framework            | ✅ Yes |                                                                  |
| `transformers-34204`      | Framework            | ❌ No  | Invariant support still in progress                              |
| `x-jxmnop-ddp-out-of-sync`| User Code            | ✅ Yes | Reuses pytorch-104336 trace                                      |

We currently support **15 out of 18 bugs** for artifact evaluation.  
You have already detected `pytorch-forum-84911` in our 5-min tutorial. You will need to detect the rest of the 14 bugs.

Bugs not included in this AE release typically depend on:
- Unsupported or unstable library versions
- Very old Python environments
- Invariant support still in development

Additionally, a few bugs stem from very specific issues such as faulty hardware, which are inherently difficult to reproduce.
For such cases—and for bugs that share the same root cause/manifest—we may provide a **shared/simulated trace** and a **shared invariant** that is reused across multiple bug IDs.

## 🧪 Reproducing Silent Issue Detection

> All steps described below assumes you are already in the `TrainCheck-Evaluation-Workloads` repo. If not, clone the repository and go to it.
> ```bash
> git clone https://github.com/OrderLab/TrainCheck-Evaluation-Workloads.git
> cd TrainCheck-Evaluation-Workloads
> ```

1. Make sure you have a working TrainCheck installation by following [TrainCheck Installation Guide](./installation-guide.md).

2. Execute `ae_detection.sh` to automatically apply invariants to the pre-collected trace. This script generates results into a folder named `checker_output`.

3. Compare the detection result folder with our claimed checker results, to verify that the checking process makes sense.
    ```bash
    diff -r checker_output reference_checker_output/
    ```

## Expected Results

The `diff -r` command should return no or very little output.

You might see outputs like this
```bash
(traincheck) (base) yuxuan@ring18:~/TrainCheck-Evaluation-Workloads/silent-issue-detection$ diff -r checker_output reference_checker_output/
diff -r checker_output/trace_pytorch-115607/failed.log reference_checker_output/trace_pytorch-115607/failed.log
43,44c43,44
<                                     "init",
<                                     "testing"
---
>                                     "testing",
>                                     "init"
261,262c261,262
<                                     "init",
<                                     "testing"
---
>                                     "testing",
>                                     "init"
diff -r checker_output/trace_transformers-33844/failed.log reference_checker_output/trace_transformers-33844/failed.log
247c247
<                 "enabled": {
---
>                 "cache_enabled": {
250c250
<                 "cache_enabled": {
---
>                 "enabled": {
```

Such differences are expected and benign. TrainCheck do not enforce the ordering when serializing data structures like `set`, which might cause you to see diffs like this.
