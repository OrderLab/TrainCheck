
# Eval: Performance Overhead

⏳ Estimated Completion Time: 15 minutes.

## 🎯 Goal

This evaluation measures the runtime overhead introduced by TrainCheck’s instrumentation compared to un-instrumented runs across a set of representative ML workloads, during the invariant checking stage. The results correspond to Section 5.5 of the paper.

## 📂 Resources & Scripts

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


## 🛠 How to Run

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

## 🧑‍💻 Expected Output

After execution completes, a plot will be generated at `performance_ae.pdf`. All the raw data are stored at a folder named `perf_res_ae`.

## 🧐 How to Verify

- Open the generated file performance_ae.pdf and compare it against Figure 7 in the paper.
- Small differences in the overhead numbers (within ±20%) are expected.
TrainCheck’s overhead is sensitive to CPU performance, since trace serialization is blocking and CPU-bound.
- Despite minor variations, the key takeaway should remain clear:
TrainCheck’s selective instrumentation incurs significantly lower overhead compared to other methods.

## ⚠️ Notes & Troubleshooting
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
