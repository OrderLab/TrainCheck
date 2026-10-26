# CLI Reference: Collect Traces

Start with [Use TrainCheck](usage-guide.md) if you want the full workflow. This page explains the `traincheck-collect` command.

`traincheck-collect` instruments a PyTorch training script and writes trace files. Use it for two jobs:

- Full reference collection for invariant inference.
- Selective target collection for checking with an existing invariant file.

## Full Reference Collection

Use full collection on a known-good run:

```bash
traincheck-collect \
  --pyscript train.py \
  --models-to-track model \
  --output-dir reference_trace
```

This command runs `train.py`, tracks the Python variable named `model`, and writes trace files into `reference_trace/`.

## Selective Target Collection

Use selective collection when you already have an invariant file:

```bash
traincheck-collect \
  --pyscript train.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```

`--invariants` tells TrainCheck which APIs and variables matter for checking. This usually reduces target-run overhead compared with full reference collection.

Do not combine `--invariants` with `--use-full-instr` when you want selective collection.

## Step Sampling

Sampling is also configured on `traincheck-collect`:

```bash
traincheck-collect \
  --pyscript train.py \
  --models-to-track model \
  --invariants invariants.json \
  --sampling-interval 10 \
  --warm-up-steps 10 \
  --output-dir target_trace
```

This traces the warm-up steps, then traces every tenth step. Use sampling for long target runs after you have confirmed TrainCheck works on a short run.

## Config Files

Use `--use-config` when the collection command needs repeated options:

```bash
traincheck-collect --use-config --config traincheck.yml
```

Example:

```yaml
pyscript: ./train.py
shscript: ./run.sh
modules_to_instr:
  - torch
models_to_track:
  - model
model_tracker_style: proxy
copy_all_files: false
output_dir: traincheck_trace
```

Config keys use underscores, not hyphens. For example, the CLI flag `--output-dir` becomes `output_dir` in YAML.

## Useful Options

- `--pyscript`: Python entry point for the training program.
- `--shscript`: shell script used to launch the Python program.
- `--models-to-track`: model variable names to track.
- `--modules-to-instr`: Python modules to instrument, usually `torch`.
- `--invariants`: invariant files for selective collection.
- `--output-dir`: directory for traces and logs.
- `--sampling-interval`: collect every Nth step after warm-up.
- `--warm-up-steps`: collect the first N steps.
- `--copy-all-files`: copy files beside the training script into the output directory.
- `--model-tracker-style`: choose `proxy`, `subclass`, or `sampler`.

Run the command help for the complete option list:

```bash
traincheck-collect --help
```

## Output Files

The output directory contains trace files, environment metadata, logs, and the instrumented training script. The checker accepts the full output directory through `-f` or `--trace-folders`.

```bash
traincheck-check -f target_trace -i invariants.json
```
