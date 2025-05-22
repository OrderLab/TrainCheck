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
   ```bash
   cd silent-issue-detection
   bash ae_detection.sh
   ```

## Expected Results

The `checker_output` folder contains checkering results for each trace.
```bash
(base) ➜  checker_output git:(main) tree .
.
├── invariants.json
├── trace_deepspeed-1801
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_mmpretrain-702
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_pytorch-104336
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_pytorch-115607
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_pytorch-51800
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_stackoverflow-60335387
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_transformers-17877
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_transformers-23723
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
├── trace_transformers-33844
│   ├── failed.log
│   ├── not_triggered.log
│   └── passed.log
└── trace_x-jxmnop-ddp-out-of-sync
    ├── failed.log
    ├── not_triggered.log
    └── passed.log
```
You want to make sure **all `failed.log` files are non-empty**, indicating silent errors detected in these traces.

---
Optionally, we provided a `reference_checker_output` folder containing the expected detection results. You can compare the checking results you get versus our reference results.

When you do this comparison, do keep in mind that your results and our reference results will not be exactly the same, as TrainCheck does not enforce (1) the order of checking when multiple checkable entities are available at the same time, which may lead to invariants violated on different processes or different pairs of variables, (2) the order of fields when serializing checker results. 

Thus, if you do the comparison in a mechanical way (e.g. via `diff -r checker_output reference_checker_output`), you might see outputs like this. The differences are expected and benign.
```bash
(base) ➜  silent-issue-detection git:(main) diff -r checker_output reference_checker_output
diff --color -r checker_output/trace_pytorch-104336/failed.log reference_checker_output/trace_pytorch-104336/failed.log
75,76c75,76
<     "detection_time": 2437523672783,
<     "detection_time_percentage": 0.11841431018590723,
---
>     "detection_time": 2437539087694,
>     "detection_time_percentage": 0.11851643004648255,
81,82c81,82
<             "process_id": 9539,
<             "thread_id": 140711397648192,
---
>             "process_id": 9591,
>             "thread_id": 140324043503424,
86c86
<             "attributes._ML_DAIKON_data_ID": 140704882109040,
---
>             "attributes._ML_DAIKON_data_ID": 140317529048544,
116,117c116,117
<             "time": 2437523672783,
<             "meta_vars._DATA_PARALLEL_RANK": 4.0,
---
>             "time": 2437504805077,
>             "meta_vars._DATA_PARALLEL_RANK": 5.0,
123,124c123,124
<             "process_id": 9429,
<             "thread_id": 140050208577344,
---
>             "process_id": 9747,
>             "thread_id": 140028492969792,
128c128
<             "attributes._ML_DAIKON_data_ID": 140043703504144,
---
>             "attributes._ML_DAIKON_data_ID": 140021978318304,
158,159c158,159
<             "time": 2437502499438,
<             "meta_vars._DATA_PARALLEL_RANK": 2.0,
---
>             "time": 2437539087694,
>             "meta_vars._DATA_PARALLEL_RANK": 7.0,
diff --color -r checker_output/trace_pytorch-115607/failed.log reference_checker_output/trace_pytorch-115607/failed.log
43,44c43,44
<                                     "init",
<                                     "testing"
---
>                                     "testing",
>                                     "init"
78,80d77
<             "exception": NaN,
<             "exception_msg": NaN,
<             "proxy_obj_names": NaN,
113c110,113
<             "attributes._ML_DAIKON_grad_ID": NaN
---
>             "attributes._ML_DAIKON_grad_ID": NaN,
>             "exception": NaN,
>             "exception_msg": NaN,
>             "proxy_obj_names": NaN
180,182d179
<             "exception": NaN,
<             "exception_msg": NaN,
<             "proxy_obj_names": NaN,
215c212,215
<             "attributes._ML_DAIKON_grad_ID": NaN
---
>             "attributes._ML_DAIKON_grad_ID": NaN,
>             "exception": NaN,
>             "exception_msg": NaN,
>             "proxy_obj_names": NaN
261,262c261,262
<                                     "init",
<                                     "testing"
---
>                                     "testing",
>                                     "init"
296,298d295
<             "exception": NaN,
<             "exception_msg": NaN,
<             "proxy_obj_names": NaN,
331c328,331
<             "attributes._ML_DAIKON_grad_ID": NaN
---
>             "attributes._ML_DAIKON_grad_ID": NaN,
>             "exception": NaN,
>             "exception_msg": NaN,
>             "proxy_obj_names": NaN
diff --color -r checker_output/trace_pytorch-51800/failed.log reference_checker_output/trace_pytorch-51800/failed.log
34,56d33
<             "func_call_id": "b39a4a81b2c24473ba916ab1832fbf12_19876858668012869",
<             "thread_id": 140254285555520,
<             "process_id": 3499981,
<             "meta_vars.step": 0,
<             "type": "function_call (pre)",
<             "function": "torch.nn.modules.module.Module.eval",
<             "is_bound_method": true,
<             "obj_id": 140250960790096,
<             "args": {
<                 "0": {
<                     "__main__.SimpleCNN": {
<                         "call_super_init": false,
<                         "dump_patches": false,
<                         "training": true
<                     }
<                 }
<             },
<             "kwargs": {},
<             "time": 19876858668088743,
<             "return_values": NaN,
<             "exception": NaN,
<             "exception_msg": NaN,
<             "proxy_obj_names": NaN,
60a38,41
>             "process_id": 3499981,
>             "thread_id": 140254285555520,
>             "time": 19876858668088743,
>             "meta_vars.step": 0,
89c70,89
<             "attributes._ML_DAIKON_grad_ID": NaN
---
>             "type": "function_call (pre)",
>             "attributes._ML_DAIKON_grad_ID": NaN,
>             "func_call_id": "b39a4a81b2c24473ba916ab1832fbf12_19876858668012869",
>             "function": "torch.nn.modules.module.Module.eval",
>             "is_bound_method": true,
>             "obj_id": 140250960790096,
>             "args": {
>                 "0": {
>                     "__main__.SimpleCNN": {
>                         "call_super_init": false,
>                         "dump_patches": false,
>                         "training": true
>                     }
>                 }
>             },
>             "kwargs": {},
>             "return_values": NaN,
>             "exception": NaN,
>             "exception_msg": NaN,
>             "proxy_obj_names": NaN
diff --color -r checker_output/trace_transformers-33844/failed.log reference_checker_output/trace_transformers-33844/failed.log
244,246d243
<                 "enabled": {
<                     "bool": true
<                 },
250a248,250
>                     "bool": true
>                 },
>                 "enabled": {
diff --color -r checker_output/trace_x-jxmnop-ddp-out-of-sync/failed.log reference_checker_output/trace_x-jxmnop-ddp-out-of-sync/failed.log
81,82c81,82
<             "process_id": 89557,
<             "thread_id": 140661207828288,
---
>             "process_id": 89558,
>             "thread_id": 140625926412096,
85c85
<             "meta_vars._DATA_PARALLEL_RANK": "0",
---
>             "meta_vars._DATA_PARALLEL_RANK": "1",
87c87
<             "attributes._ML_DAIKON_data_ID": 140656561409856,
---
>             "attributes._ML_DAIKON_data_ID": 140621279056480,
117c117
<             "time": 123297988837864,
---
>             "time": 123299970638648,
123,124c123,124
<             "process_id": 89558,
<             "thread_id": 140625926412096,
---
>             "process_id": 89557,
>             "thread_id": 140661207828288,
127c127
<             "meta_vars._DATA_PARALLEL_RANK": "1",
---
>             "meta_vars._DATA_PARALLEL_RANK": "0",
129c129
<             "attributes._ML_DAIKON_data_ID": 140621279058160,
---
>             "attributes._ML_DAIKON_data_ID": 140656561411776,
159c159
<             "time": 123299970638648,
---
>             "time": 123297988837864,
```