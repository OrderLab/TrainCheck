# TrainCheck Documentation

TrainCheck is a lightweight, invariant-based instrumentation and analysis tool for identifying silent correctness issues in PyTorch training pipelines. It infers behavioral invariants from correct reference runs (e.g., official examples or clean configurations), then checks other scripts for behavioral violations. TrainCheck is designed to be minimally intrusive—requiring no code modifications or rewrites of training logic.

## System Overview

TrainCheck consists of four user-facing command-line utilities:

1. **traincheck-collect** – Instruments a training pipeline and collects trace logs.
2. **traincheck-infer** – Infers behavioral invariants from the collected traces.
3. **traincheck-onlinecheck** – Checks a target trace folder while training is still running.
4. **traincheck-check** – Checks completed traces against inferred invariants.

TrainCheck workflows are organized into two stages:

1. **Inference Stage**
    - **traincheck-collect** collects execution traces from reference training pipelines.
    - **traincheck-infer** analyzes traces and produces invariants that describe correct/expected runtime behavior.

2. **Checking Stage**
    - **traincheck-collect** traces the target pipeline with `--invariants` for selective collection.
    - **traincheck-onlinecheck** verifies traces while the target run is active.
    - **traincheck-check** verifies completed traces after collection finishes.

### 📦 Pre-Inferred Invariants (On the Roadmap)

In common use cases, users typically do not need to infer invariants manually. TrainCheck provides a high-quality set of pre-inferred invariants that work out-of-the-box with popular libraries such as PyTorch, HuggingFace Transformers, and DeepSpeed.

You may still want to run inference in the following cases:
- When using certain niche or uncommon features not covered by the default invariants.
- When working with custom training stacks outside supported libraries.
- When you want to increase specificity by inferring invariants from a set of related, known-good pipelines (e.g. in industrial settings).

## Component Documentation

Start with [Use TrainCheck](usage-guide.md) for the workflow. Use these pages as command references:

- [Collecting Traces with traincheck-collect](instr.md)  
  Collection modes, config files, model tracking, selective collection, and sampling.
    
- [Inferring Invariants with traincheck-infer](infer.md)  
  Trace inputs, invariant outputs, relation filtering, and inference options.

- [Checking Violations](check.md)  
  Live checking, offline checking, reports, and integrations.
