# 🧪 TrainCheck: Usage Guide

TrainCheck helps detect and diagnose silent errors in deep learning training runs—issues that don't crash your code but silently break correctness.

## 🚀 Quick Start

Check out the [5-minute guide](./docs/5-min.md) for a minimal working example.

## ✅ Common Use Cases

TrainCheck is useful when your training process doesn’t converge, behaves inconsistently, or silently fails. It can help you:

- **Monitor** long-running training jobs and catch issues early
- **Debug** finished runs and pinpoint where things went wrong
- **Sanity-check** new pipelines, code changes, or infrastructure upgrades

TrainCheck detects a range of correctness issues—like misused APIs, incorrect training logic, or hardware faults—without requiring labels or modifications to your training code.

**While TrainCheck focuses on correctness, it’s also useful for *ruling out bugs* so you can focus on algorithm design with confidence.**

## 🧠 Tips for Effective Use

1. **Use short runs to reduce overhead.**  
   If your hardware is stable, you can validate just the beginning of training. Use smaller models and fewer iterations to speed up turnaround time.

2. **Choose good reference runs for inference.**  
   - If you have a past run of the same code that worked well, just use that.
   - You can also use small-scale example pipelines that cover different features of the framework (e.g., various optimizers, mixed precision, optional flags).
   - If you're debugging a new or niche feature with limited history, try using the official example as a reference. Even if the example is not bug-free, invariant violations can still highlight behavioral differences between your run and the example, helping you debug faster.

3. **Minimize scale when collecting traces.**  
   - Shrink the pipeline by using a smaller model, running for only ~10 iterations, and using the minimal necessary compute setup (e.g., 2 nodes for distributed training).


## 🚧 Current Limitations

- **Eager mode only.** TrainCheck instrumentor currently works only in PyTorch eager mode. Features like `torch.compile` are disabled during instrumentation.

- **Not fully real-time (yet).** Invariant checking is semi-online. Full real-time support is planned but not yet available.

