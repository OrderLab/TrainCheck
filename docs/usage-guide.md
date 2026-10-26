# Use TrainCheck

This guide is for an ML engineer who has a training script and wants to know which TrainCheck command to run next.

TrainCheck uses a reference run to learn invariants: rules that describe normal training behavior. It then checks a target run against those invariants and reports violations.

## The Default Workflow

### 1. Collect a Reference Trace

Start with a short, known-good run. The run can come from your own training code, a previous clean run, or an official example that uses the same framework features.

```bash
traincheck-collect \
  --pyscript reference.py \
  --models-to-track model \
  --output-dir reference_trace
```

`--models-to-track model` names the Python variable that holds the model in `reference.py`. If your script uses a different variable name, use that name instead.

TrainCheck writes trace files and an `env_dump.txt` file into `reference_trace/`.

### 2. Infer Invariants

Use the reference trace to produce an invariant file:

```bash
traincheck-infer -f reference_trace -o invariants.json
```

You can pass multiple reference trace folders when one run does not cover enough behavior:

```bash
traincheck-infer -f reference_trace_1 reference_trace_2 -o invariants.json
```

More diverse reference traces reduce overfitting. Short runs are usually enough because training loops repeat the same operations many times.

### 3. Collect a Target Trace

Run the target script with the inferred invariants:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```

Passing `--invariants` enables selective trace collection. TrainCheck traces the APIs and variables needed by the invariant file instead of collecting a full reference-style trace.

For long target runs, sample steps during collection:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --sampling-interval 10 \
  --warm-up-steps 10 \
  --output-dir target_trace
```

This traces the warm-up steps, then traces every tenth step. Sampling is a `traincheck-collect` option. The checker reads the collected trace; it does not control which steps were traced.

### 4. Check the Target Run

For live checking, start the online checker while `traincheck-collect` is still writing into `target_trace/`:

```bash
traincheck-onlinecheck -f target_trace -i invariants.json
```

If `traincheck-onlinecheck` fails with a missing `watchdog` package, install it in the same environment:

```bash
pip install watchdog
```

The easier path is offline checking. Wait for trace collection to finish, then run:

```bash
traincheck-check -f target_trace -i invariants.json
```

Use offline checking first when you are learning TrainCheck or reproducing an issue locally. Use live checking when you want violations while the training job is still running.

## What TrainCheck Writes

`traincheck-collect` writes:

- `trace_*.json` files for API traces.
- `proxy_log.json` when model-variable tracking is active.
- `env_dump.txt` with the collection arguments.
- an instrumented copy of the training script and execution logs.

`traincheck-infer` writes:

- `invariants.json` by default, or the path passed with `-o`.

`traincheck-check` and `traincheck-onlinecheck` write:

- `failed.log` for violated invariants.
- `passed.log` for triggered invariants that passed.
- `not_triggered.log` for invariants that never ran on the trace.
- `violations_summary.json` for machine-readable violation summaries.
- `report.html` for a browser-readable report.

## Choosing Reference and Target Runs

Use a reference run that should be correct. If you have a clean run of the same pipeline, use it. If you are debugging a new pipeline, start with an official example or a smaller training job that uses the same optimizer, precision mode, and distributed setup.

Use a target run for the script you want to check. It can be a modified version of the reference script, a larger training job, or a run that already looks suspicious.

Keep both runs short while you are iterating. TrainCheck needs representative behavior more than long training time.

## Common Adjustments

Use a config file when the command line gets long:

```bash
traincheck-collect --use-config --config traincheck.yml
```

A minimal config looks like this:

```yaml
pyscript: ./train.py
models_to_track:
  - model
modules_to_instr:
  - torch
output_dir: traincheck_trace
```

When using a config file, use underscores in config keys, such as `output_dir`, instead of CLI-style hyphens.

Use a shell script when your training command needs environment variables or launcher arguments:

```yaml
pyscript: ./train.py
shscript: ./run.sh
models_to_track:
  - model
```

Use `--copy-all-files` if the training script reads local files through relative paths.

## Current Limits

TrainCheck currently instruments PyTorch eager-mode execution. If your script uses `torch.compile`, pass `--use-torch-compile` so TrainCheck can keep compatibility behavior explicit.

Tracing adds overhead. Start with short runs, selective target tracing, and step sampling before scaling to a long job.
