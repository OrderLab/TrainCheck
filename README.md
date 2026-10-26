<div align="center">
<picture>
  <img alt="TrainCheck logo" width="55%" src="https://raw.githubusercontent.com/OrderLab/TrainCheck/main/docs/assets/images/traincheck_logo.png">
</picture>
<h1>TrainCheck: Invariant Checking for AI Training</h1>

[![Chat on Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/ZvYewjsQ9D)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OrderLab/TrainCheck)

</div>

TrainCheck catches silent training bugs by learning what a healthy run does, then checking a new run against those learned invariants. It works by tracing PyTorch API calls and model state changes, so you can inspect training behavior before a loss curve or final metric tells you something went wrong.

## Install

Install TrainCheck in the same Python environment that runs your training script:

```bash
pip3 install traincheck
```

For CUDA, conda, and source-install details, see the [Installation Guide](https://orderlab.io/TrainCheck/installation-guide/).

## Use TrainCheck

TrainCheck has four main steps.

### 1. Collect a Reference Trace

Run `traincheck-collect` on a known-good training script. This should be a short run that covers the training behavior you want TrainCheck to learn.

```bash
traincheck-collect \
  --pyscript reference.py \
  --models-to-track model \
  --output-dir reference_trace
```

### 2. Infer Invariants

Turn the reference trace into invariants:

```bash
traincheck-infer -f reference_trace -o invariants.json
```

### 3. Collect a Target Trace

Run the target training script with the inferred invariants. Passing `--invariants` lets TrainCheck trace only the APIs and variables needed for those checks.

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```

For long target runs, trace fewer steps:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --sampling-interval 10 \
  --warm-up-steps 10 \
  --output-dir target_trace
```

### 4. Check the Target Run

For live checking, start `traincheck-onlinecheck` while the target run is writing traces:

```bash
traincheck-onlinecheck -f target_trace -i invariants.json
```

The easier offline path is to wait for trace collection to finish, then run:

```bash
traincheck-check -f target_trace -i invariants.json
```

Both checkers write a results directory with failure logs and a `report.html` summary.

## Learn More

- [Use TrainCheck](https://orderlab.io/TrainCheck/usage-guide/) explains the full workflow and output files.
- [5-Minute Tutorial](./docs/5-min-tutorial.md) walks through a real silent training issue.
- [Installation Guide](https://orderlab.io/TrainCheck/installation-guide/) covers environment setup.
- [Technical Documentation](https://orderlab.io/TrainCheck/technical-doc/) describes invariants, trace representation, and implementation details.

## Status

TrainCheck is under active development. Please join our [Discord server](https://discord.gg/VwxpJDvB), file a GitHub issue, or email [traincheck@umich.edu](mailto:traincheck@umich.edu).

## Contributing

We welcome contributions. See [Contributing to TrainCheck](./CONTRIBUTING.md) for setup and contribution guidance.

## License

TrainCheck is licensed under the [Apache License 2.0](./LICENSE).

## Citation

If TrainCheck is relevant to your work, please cite our paper:

```bib
@inproceedings{TrainCheckOSDI2025,
  author = {Jiang, Yuxuan and Zhou, Ziming and Xu, Boyu and Liu, Beijie and Xu, Runhui and Huang, Peng},
  title = {Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks},
  booktitle = {Proceedings of the 19th USENIX Symposium on Operating Systems Design and Implementation},
  series = {OSDI '25},
  month = {July},
  year = {2025},
  address = {Boston, MA, USA},
  publisher = {USENIX Association},
}
```

## Artifact Evaluation

OSDI AE members should use the [TrainCheck AE Guide](./docs/ae.md).
