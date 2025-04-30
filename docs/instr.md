# Instrumentation & Trace Representation

`traincheck-collect` is the starting point of TrainCheck's workflow. It instruments your PyTorch training script to capture runtime behavior, generating detailed execution traces for later invariant inference and issue detection.

This document explains how to use `traincheck-collect` effectively.  
TrainCheck dynamically wraps key PyTorch APIs and monitors model states—**no modifications to your original training code are required**.

Use `traincheck-collect` when you need to:
- Generate traces from **reference pipelines** for invariant inference.
- Collect traces from **target pipelines** to detect silent issues using pre-inferred invariants.

## Table of Contents

1. [Introduction](#instrumentation--trace-representation)
2. [🔧 Basic Usage](#-basic-usage)
   - [Configuration File Example](#configuration-file-example)
   - [Running traincheck-collect](#running-traincheck-collect)
   - [Selective Instrumentation for Checking](#selective-instrumentation-for-checking)
   - [Output Structure](#output-structure)
   - [Overriding Configuration via CLI](#overriding-configuration-via-cli)
3. [Adding Meta Variables to Traces](#adding-meta-variables-to-traces)
   - [How Meta Variables Improve Inference](#learn-how-meta-variables-improve-invariant-inference)
   - [Examples of Useful Meta Variables](#-examples-of-useful-meta-variables)
   - [How to Annotate Meta Variables](#how-to-annotate-meta-variables)
4. [Trace Representation](#trace-representation)
5. [Instrumentation Mechanisms](#instrumentation-mechanisms)
6. [Advanced Usage](#advanced-usage)  <!-- Placeholder for your future section -->
7. [Algorithms Overview](#algorithms-overview)  <!-- Placeholder for your future section -->
8. [Troubleshooting & FAQs](#troubleshooting--faqs)  <!-- Optional but useful for AE -->

## 🔧 Basic Usage

`traincheck-collect` requires three types of input:

1. **Python script** to instrument.
2. **Launch arguments** (if any) for executing the script.
3. **Instrumentation-specific configurations**.

You can provide these inputs either directly via the command line or through a configuration file.  
▶️ **Recommendation**: Use a configuration file for clarity and reusability.

Here’s an example configuration:

```yaml
pyscript: ./mnist.py        # Python entry point of your training program.
shscript: ./run.sh          # [Optional] Shell script to launch with custom arguments or environment setup.
modules_to_instr:           # Libraries to instrument. Defaults to ['torch'] if omitted.
  - torch
models_to_track:            # [Optional] Variable names of models to track. Leave empty to disable model tracking.
  - model
model_tracker_style: proxy  # [Optional] Tracking method: "proxy" (default) or "sampler".
copy_all_files: false       # [Optional] Set true if your code relies on relative paths (e.g., local datasets/configs).
```

You can find example configurations and training programs in:
	•	[MNIST Example](./assets/examples/traincheck-collect/mnist-config/)
	•	[GPT-2 Pretrain Example](./assets/examples/traincheck-collect/gpt2-pretrain-config/)

Run TrainCheck trace collection with:  

```bash
traincheck-collect --use-config --config <path-to-config-file>
```

This command instruments the specified libraries and model variables, then executes your program.
(Details on instrumentation mechanisms and limitations will follow in the next section. TODO)

### Selective Instrumentation for Checking

When checking for silent issues, `traincheck-collect` supports selective instrumentation to improve efficiency.
Simply provide the invariants file:

```bash
traincheck-collect --use-config --config <path-to-config> --invariants <path-to-inv-file>
```

TrainCheck will automatically adjust instrumentation granularity based on the provided invariants.

### Output Structure
By default, TrainCheck creates a folder named:

```bash
traincheck_run_<pyscript_name>_<instr_libs>_<timestamp>
```

This folder contains:
- Collected traces
- Instrumented scripts and execution logs (if the program completes successfully)

You can also provide any additional arguments not specified in the configuration through the commandline interface, such as

### Overriding Configuration via CLI

You can override or supplement configuration settings by providing additional arguments directly via the command line. For example:

```bash
# Write trace files to ./trace_training instead of using the default auto-generated folder name
traincheck-collect --use-config --config <path-to-config-file> --output-dir trace_training
```

To view all available command-line arguments and configuration options, run:

```bash
traincheck-collect --help
```

**Note**: When using a configuration file, replace hyphens (-) in argument names with underscores (_).
For example:
- Command-line: `--output-dir trace_training`
- Configuration file: `output_dir: trace_training`

## Adding Meta Variables to Traces

You can enhance your traces by providing **custom meta variables**—semantic information about your program's execution. These annotations improve the **quality and precision** of inferred invariants by offering context that might not be directly observable from raw traces.

<details>
<summary>Learn how meta variables improve invariant inference</summary>

TrainCheck infers **preconditions** for each invariant—these are predicates that distinguish between positive and negative examples in the trace.  
- A **positive example** is a trace segment where the invariant holds.  
- A **negative example** is where it is violated.

Many invariants are inherently **conditional**, meaning they only hold true under certain contexts (e.g., during training but not initialization). TrainCheck tries to automatically discover such conditions.

However, trace data alone may lack sufficient context. This is where **meta variables** come in—they inject semantic hints (like execution phase or step number) to guide smarter inference.

</details>

### ✨ Examples of Useful Meta Variables
1. **`stage`** — Indicates whether a trace record belongs to initialization, training, or evaluation.
2. **`step_id`** — The current training step or iteration number.
3. **Custom arguments** — Any domain-specific flags or parameters relevant to your training logic.

### How to Annotate Meta Variables
📌 **[To Be Documented]**  
Instructions for defining and injecting meta variables into traces will be provided in a future update.

## Trace Representation
📌 **[To Be Documented]** 

## Instrumentation Mechanisms
📌 **[To Be Documented]**  
Details about TrainCheck’s instrumentation strategies, supported APIs, and limitations will be covered here later.