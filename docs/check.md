# TrainCheck Checker Usage Guide

`traincheck-check` is the **final stage** of the TrainCheck workflow. It verifies a set of invariants against trace files or streams from target programs, reporting any detected violations—helping you catch silent issues in your ML training pipelines.

## 🔧 Checking Modes

TrainCheck supports two checking modes:

- **Post-training Checking (`traincheck-check`)**:  
   Perform invariant checking on completed trace files after the training job finishes. ✅

- **On-the-fly Checking (`traincheck-onlinecheck`):**
   Perform real-time checking while the target training job is running. ✅

## How to Use: On-the-fly Checking

While training is in progress with `traincheck-collect`, run the following command:

```bash
traincheck-onlinecheck -f <trace_folder> -i <path_to_invariant_file>
```

- `-f <trace_folder>`: Path to the folder where traces are:
  - Already collected, or
  - **Actively being collected** by `traincheck-collect` during the training job.

- `-i <path_to_invariant_file>`: Path to the JSON file containing inferred invariants.

## How to Use: Post-training Checking

Run the following command:

```bash
traincheck-check -f <trace_folder> -i <path_to_invariant_file>
```

- `-f <trace_folder>`: Path to the folder containing traces collected by `traincheck-collect`.
- `-i <path_to_invariant_file>`: Path to the JSON file containing inferred invariants.

## Report Visualization Options

Both checkers can produce a standalone HTML report and optionally log summary metrics to external monitoring tools.

### Standalone HTML Report (default)

- Output: `<output_dir>/report.html`
- Includes summary counts, relation breakdown, and top violations.
- Disable with `--no-html-report`.

**Offline example**
```bash
traincheck-check -f <trace_folder> -i <path_to_invariant_file>
```

**Online example**
```bash
traincheck-onlinecheck -f <trace_folder> -i <path_to_invariant_file>
```

### W&B Integration

Enable with `--report-wandb`. You can also pass:
`--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-group`, `--wandb-tags`.

```bash
traincheck-check -f <trace_folder> -i <path_to_invariant_file> \
  --report-wandb --wandb-project <project>
```

```bash
traincheck-onlinecheck -f <trace_folder> -i <path_to_invariant_file> \
  --report-wandb --wandb-project <project>
```

### MLflow Integration

Enable with `--report-mlflow`. Optional:
`--mlflow-experiment`, `--mlflow-run-name`.

```bash
traincheck-check -f <trace_folder> -i <path_to_invariant_file> \
  --report-mlflow --mlflow-experiment <experiment>
```

```bash
traincheck-onlinecheck -f <trace_folder> -i <path_to_invariant_file> \
  --report-mlflow --mlflow-experiment <experiment>
```

### Online Report Refresh

The online checker refreshes the report when violations change, and also on a periodic timer.
Control the interval with `--report-interval-seconds` (default: 10).

```bash
traincheck-onlinecheck -f <trace_folder> -i <path_to_invariant_file> \
  --report-interval-seconds 30
```

**Note:** W&B and MLflow logging are optional. If the packages are not installed, TrainCheck will skip logging and emit a warning.

## Interpreting the Results

After running either checking mode, TrainCheck will output a summary of detected invariant violations. Each violation entry typically includes:

- **Trace file or stream name**: Identifies where the issue was found.
- **Invariant description**: Details the specific invariant that was violated.
- **Violation details**: Provides context, such as the step or epoch where the violation occurred.

Review these results to pinpoint silent errors or unexpected behaviors in your ML training pipeline. For more information on result formats and how to diagnose issues, see [5. Detection & Diagnosis](./5-min-tutorial.md#5-detection--diagnosis) in the **5-Minute Tutorial**.
