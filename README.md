<div align="center">
<picture>
  <img alt="TrainCheck logo" width="55%" src="https://raw.githubusercontent.com/OrderLab/TrainCheck/main/docs/assets/images/traincheck_logo.png">
</picture>
<h1>TrainCheck: Invariant Checking & Observability for AI Training</h1>

[![Chat on Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/ZvYewjsQ9D)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OrderLab/TrainCheck)

</div>


**Stop flying blind.** TrainCheck gives you deep visibility into your training dynamics, continuously validating correctness and stability where standard metrics fail.

---

### Why TrainCheck?

✅ **Continuous Invariant Checking**
TrainCheck validates the "physics" of your training process in real-time. It ensures your model adheres to learned invariants—such as gradient norms, tensor shapes, and update magnitudes—effectively catching silent corruption before it wastes GPU hours.

🚀 **Holistic Observability**
Traditional tools only show you *if* your model crashed. TrainCheck shows you *why* it's degrading, analyzing internal state dynamics that loss curves miss.

🧠 **Zero-Config Validation**
No manual tests required. TrainCheck automatically learns the invariants of your specific model from healthy runs and flags deviations instantly.

⚡ **Universal Compatibility**
Drop-in support for PyTorch, Hugging Face, and industry-class workloads using DeepSpeed/Megatron and more.

---
## Installation

Install TrainCheck in the Python environment where you will run your training script:

```bash
pip3 install traincheck
```

For detailed setup (CUDA configuration, UV, conda environments), see the [Installation Guide](https://orderlab.io/TrainCheck/installation-guide/).


### How It Works

1. **Instrument**: We wrap your training loop with lightweight probes—no code changes needed.
2. **Learn**: We analyze correct runs to infer *invariants* (mathematical rules of healthy training).
3. **Check**: We monitor new runs in real-time, verifying every step against learned invariants to catch silent logic bugs and hardware faults.

![Workflow](https://raw.githubusercontent.com/OrderLab/TrainCheck/main/docs/assets/images/workflow.png)

## 🔥 Try TrainCheck

Work through [5‑Minute Experience with TrainCheck](./docs/5-min-tutorial.md). You’ll learn how to:
   - Instrument a training script and collect a trace  
   - Automatically infer invariants  
   - Uncover silent bugs in the training script

## Documentation

- **[Installation Guide](https://orderlab.io/TrainCheck/installation-guide/)**
- **[Usage Guide: Scenarios and Limitations](https://orderlab.io/TrainCheck/usage-guide/)**
- **[TrainCheck Technical Doc](https://orderlab.io/TrainCheck/technical-doc/)**

## Status

TrainCheck is under active development. Please join our 💬 [Discord server](https://discord.gg/VwxpJDvB) or file a GitHub issue for support. You can also reach the team at [traincheck@umich.edu](mailto:traincheck@umich.edu).
We welcome feedback and contributions from early adopters.

## Contributing

We welcome and value any contributions and collaborations. Please check out [Contributing to TrainCheck](./CONTRIBUTING.md) for how to get involved.

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

🕵️‍♀️ OSDI AE members, please see [TrainCheck AE Guide](./docs/ae.md).
