# Eval: False Positive Rate

⏳ Estimated Completion Time: 2 hour.
- Trace Collection: ~10 minutes
- Invariant Inference & Checking: ~1.5 hours

## 🎯 Goal

This evaluation measures the false positive rate of alarms reported by TrainCheck's invariants.  
The target results are discussed in the main text of **Section 5.4** of the paper.

## 📂 Resources & Scripts

- **Automation Scripts**:
  - [`TrainCheck-Evaluation-Workloads/fp_rate/ae_fp.py`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/fp_rate/ae_fp.py): The script to collect traces, perform invariant inference, and check invariants on supposedly-correct programs to see if there are any false alarms.
  - [`TrainCheck-Evaluation-Workloads/fp_rate/compute_fp_rate.py`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/fp_rate/compute_fp_rate.py): The script to compute false positive rates from the invariant checking results.

- **Workloads**:
  - The evaluation uses official PyTorch training pipelines located at [`TrainCheck-Evaluation-Workloads/fp_rate/workloads`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/tree/main/fp_rate/workloads).
    We have shortened the training runs for faster execution.
    For AE purposes, you do not need to modify or understand the workload code—`ae_fp.py` will automatically handle the entire process.

## 🛠 How to Run

> All steps described below assumes you are already in the `TrainCheck-Evaluation-Workloads` repo. If not, clone the repository and go to it.
> ```bash
> git clone https://github.com/OrderLab/TrainCheck-Evaluation-Workloads.git
> cd TrainCheck-Evaluation-Workloads
> ```

1. Make sure you have a working TrainCheck installation by following [TrainCheck Installation Guide](./installation-guide.md).

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

4. Execute `compute_fp_rate.py` to compute the false positive rates.

    ```bash
    python3 compute_fp_rate.py
    ```

## 👀 What to Expect During Execution

The `ae_fp.py` script is long running. It performs three tasks at same time. 
1. It collects trace for all the workloads.
2. It infers invariants for three setups in Section 5.4.
3. It checks inferred invariants on the validation workloads.

The experiments might fail if environment installation issues or disruption happens. When you run into problems, please refer to [⚠️ Notes & Troubleshooting](#️-notes--troubleshooting).

## ⚠️ Notes & Troubleshooting

The script will automatically detect any errors in any (1) trace collection, (2) inference tasks, (3) checking tasks. If you encounter any trace collection issues, please check for any missing environment dependencies.

If you encounter any issues on invariant inference tasks or invariant checking tasks, please try to rerun the experiment by adding `--overwrite-existing-results` or delete all `trace_*` folders except for `trace_ddp-multigpu`.

If you see persistent issues, it will likely be a environment issue or software bug. Please contact us for help.

## 🧐 How to verify the results?

The `compute_fp_rate.py` script generates a file called `fp_rates.csv` under the current directory. Looking like this

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
1-input,0.039
4-input,0.021
6-input,0.015
```
