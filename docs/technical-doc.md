# TrainCheck Documentation

🚜 This documentation is under construction. We welcome any feedback or questions through GitHub Issues or [our Discord server](https://discord.gg/DPEd7Xeg).


TrainCheck is a lightweight, invariant-based instrumentation and analysis tool for identifying silent correctness issues in PyTorch training pipelines. It infers behavioral invariants from correct reference runs (e.g., official examples or clean configurations), then checks other scripts for behavioral violations. TrainCheck is designed to be minimally intrusive—requiring no code modifications or rewrites of training logic.

## 🔧 System Overview

TrainCheck consists of three core command-line utilities:

1. **traincheck-collect** – Instruments a training pipeline and collects trace logs.
2. **traincheck-infer** – Infers behavioral invariants from the collected traces.
3. **traincheck-check** – Checks new traces against a set of inferred invariants to detect silent issues.

TrainCheck workflows are organized into two stages:

1. **🧪 Inference Stage**
    - **traincheck-collect** collects execution traces from reference training pipelines.
    - **traincheck-infer** analyzes traces and produces invariants that describe correct/expected runtime behavior.

2. **🚨 Checking Stage**
    - **traincheck-collect** is used again to trace the target (possibly buggy) pipeline.
    - **traincheck-check** verifies whether the collected trace violates any of the known invariants.

### 📦 Pre-Inferred Invariants (On the Roadmap)

In common use cases, users typically do not need to infer invariants manually. TrainCheck provides a high-quality set of pre-inferred invariants that work out-of-the-box with popular libraries such as PyTorch, HuggingFace Transformers, and DeepSpeed.

You may still want to run inference in the following cases:
- When using certain niche or uncommon features not covered by the default invariants.
- When working with custom training stacks outside supported libraries.
- When you want to increase specificity by inferring invariants from a set of related, known-good pipelines (e.g. in industrial settings).

## 📚 Component Documentation

Each utility is documented separately:

- [Collecting Traces with traincheck-collect](instr.md)
    Usage, instrumentation caveats, and trace file format.
    
- [Inferring Invariants with traincheck-infer](infer.md)
CLI usage, performance considerations, invariant format, and the inference algorithm (relations, preconditions, etc.).

- [Checking Violations with traincheck-check](check.md)
How to apply invariants to new traces, result interpretation, and result file formats.
