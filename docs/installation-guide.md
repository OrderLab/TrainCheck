## Compatibility

- **Python**: 3.10+ (due to reliance on type annotations)
- **PyTorch**: 1.7.0–2.5.0 (other versions have not been tested.)
- **CUDA**: 11.2–12.1 (also supports MPS on macOS; see Performance note below)  
- **Operating Systems**: Ubuntu 20.04+, macOS. Windows is untested but may work—please file an issue if you hit a problem.

> **Performance note:**  
> On non‑CUDA backends (e.g., MPS), runtime overhead can vary due to differences in tensor‑hashing efficiency. We’re actively measuring and tuning across platforms.



## Installation Steps

> **Note:** Example workloads are verified on Python 3.10 and PyTorch 2.2.2 + CUDA 12.1. If you’re not reproducing our benchmarks, feel free to install any supported versions.

> **AEC note:** For full artifact evaluation, we recommend Ubuntu 22.04 with two Nvidia Ampere‑class GPUs (≥ 12 GiB GPU memory each). For the 5‑minute tutorial, any Linux or macOS (Apple Silicon) laptop will do.

1. **Install Conda**  
  Install Miniconda by following the [official Miniconda guide](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions).

2. **Create & activate a Python 3.10 Conda Env**
    ```bash
    conda create -n traincheck python=3.10 -y
    conda activate traincheck
    ```

3. **Install PyTorch 2.2.2 with CUDA support**
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    ```

    If your GPU does not support CUDA12, CUDA11.8 is also acceptable.
    
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    ```

    If you don't have a CUDA-enabled GPU, just install the CPU version and skip step 4.

    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    ```

4. **(CUDA platforms only) Install cudatoolkit**
    ```bash
    conda install cudatoolkit
    ```

5. **Clone & install TrainCheck**
    ```bash
    git clone https://github.com/OrderLab/TrainCheck.git
    cd TrainCheck
    pip3 install .
    ```

6. **Verify Installation**
    You should now have three clis installed in your system. Do a quick test to see of these commands are available and functional.
    ```bash
    traincheck-collect --help
    traincheck-infer --help
    traincheck-check --help
    ```

## Next Steps

- **5‑Minute TrainCheck Experience**  
  Follow the [5‑Minute Tutorial](./5-min-tutorial.md) to instrument a script, infer invariants, and catch silent bugs in under five minutes.

- **Technical Documentation**  
  Explore the [TrainCheck Technical Doc](./technical-doc.md) for a comprehensive guide to features, configuration, and advanced workflows.