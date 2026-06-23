# CLI Reference: Check Traces

Start with [Use TrainCheck](usage-guide.md) if you want the full workflow. This page explains `traincheck-onlinecheck` and `traincheck-check`.

TrainCheck has two checking modes:

- `traincheck-onlinecheck` checks traces while `traincheck-collect` is still writing them.
- `traincheck-check` checks completed trace files after collection finishes.

Use online checking when you want violations during a running job. Use offline checking when you want the easiest path or a reproducible local workflow.

## Live Checking

Start trace collection for the target run:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```

In another terminal, start the online checker:

```bash
traincheck-onlinecheck -f target_trace -i invariants.json
```

The online checker watches `target_trace/` and updates its report as new traces arrive.

If the command fails with a missing `watchdog` package, install it in the same environment:

```bash
pip install watchdog
```

Control the report refresh interval with:

```bash
traincheck-onlinecheck \
  -f target_trace \
  -i invariants.json \
  --report-interval-seconds 30
```

## Offline Checking

The offline path is simpler. First let `traincheck-collect` finish, then run:

```bash
traincheck-check -f target_trace -i invariants.json
```

Offline checking reads the completed trace folder and writes a results directory.

## Sampling and Checking

Sampling is configured during trace collection:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --sampling-interval 10 \
  --warm-up-steps 10 \
  --output-dir target_trace
```

Then run either checker normally:

```bash
traincheck-onlinecheck -f target_trace -i invariants.json
```

```bash
traincheck-check -f target_trace -i invariants.json
```

The checker does not decide which steps were traced. It checks the trace files that collection produced.

## Reports and Logs

Both checkers write:

- `failed.log`: violated invariants.
- `passed.log`: triggered invariants that passed.
- `not_triggered.log`: invariants that never ran on the trace.
- `violations_summary.json`: compact violation summaries.
- `report.html`: browser-readable summary.

The default output directory is timestamped. Use `-o` or `--output-dir` to choose a path:

```bash
traincheck-check \
  -f target_trace \
  -i invariants.json \
  --output-dir check_results
```

## W&B and MLflow

Log checker results to Weights & Biases:

```bash
traincheck-check \
  -f target_trace \
  -i invariants.json \
  --report-wandb \
  --wandb-project traincheck
```

Attach offline checker metrics to an existing W&B run:

```bash
traincheck-check \
  -f target_trace \
  -i invariants.json \
  --report-wandb \
  --wandb-run-id <run-id>
```

Log checker results to MLflow:

```bash
traincheck-check \
  -f target_trace \
  -i invariants.json \
  --report-mlflow \
  --mlflow-experiment traincheck
```

The online checker supports the same W&B and MLflow reporting flags.

## Useful Options

- `-f, --trace-folders`: trace directories produced by `traincheck-collect`.
- `-t, --traces`: individual trace files.
- `-i, --invariants`: invariant files produced by `traincheck-infer`.
- `-o, --output-dir`: results directory.
- `--no-html-report`: skip `report.html`.
- `--report-wandb`: log summary metrics and the HTML report to W&B.
- `--report-mlflow`: log summary metrics and the HTML report to MLflow.
- `--report-interval-seconds`: online checker report refresh interval.

Run the command help for the complete option list:

```bash
traincheck-check --help
traincheck-onlinecheck --help
```
