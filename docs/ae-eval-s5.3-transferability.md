
# Eval: Transferability

⏳ **Estimated Completion Time**: 40 minutes
- Environment Setup: ~10 minutes  
- Trace Collection: ~10 minutes  
- Invariant Inference: ~20 minutes

## 🎯 Goal

This evaluation measures the **transferability** of invariants inferred by TrainCheck across library versions and training environments.  
The results to be reproduced correspond to the final paragraph of **Section 5.3** of the paper.

Other claims in Section 5.3—specifically, that invariants inferred from reference pipelines can detect all known bugs—are validated as part of the [Silent Issue Detection Evaluation](./ae-eval-s5.1-silent-issue-detection.md).

## 📂 Resources & Scripts

- **Automation Script**:  
  - [`transferability/ae_transferability.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/ae_transferability.sh) Runs the full transferability evaluation pipeline described in Section 5.3 of the paper. It executes invariant inference, applies inferred invariants to other pipelines, and collects applicability (invariant should be checked and not cause false alarms) statistics.
  - [`transferability/install-traincheck-torch251-cu121.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/install-traincheck-torch251-cu121.sh) Creates a conda environment named traincheck-torch251 with Python 3.10 and installs TrainCheck from the latest GitHub version.
  - [`transferability/install-traincheck-torch251-cu118.sh`](https://github.com/OrderLab/TrainCheck-Evaluation-Workloads/blob/main/transferability/install-traincheck-torch251-cu118.sh) Same as above but installs the CUDA 118 version of PyTorch 2.5.1.

This evaluation uses the **GCN** training pipeline from PyTorch's official examples, tested across different PyTorch versions.  
The pipeline is included in the artifact repository and will be automatically handled by the script—no manual setup is required.

## 🛠 How to Run

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

> ⚠️ Note:
> The scripts above assume that Conda is installed at `~/miniconda3`.
> If your installation is located elsewhere (e.g., `~/anaconda3`), please modify the first line of the scripts to reflect your actual Conda path.
>
> We also assume that you have already installed TrainCheck in an environment named `traincheck` prior to running these scripts.

## 🧐 How to Verify the Results

After the script finishes, it generates a file named `applied_rates.csv` that reports the percentage of applicable invariants. You should verify that the rate is no lower than the paper’s reported value:

> 🟢 "94.2% remain valid and applicable up to PyTorch 2.5.1" (Section 5.3)

## ⚠️ Notes & Troubleshooting

If invariant inference or checking fails, please first verify that the environment is correctly set up (e.g., correct PyTorch version, dependencies installed).  
Then try re-running `ae_transferability.py`.

If the issue persists, please contact us for assistance。