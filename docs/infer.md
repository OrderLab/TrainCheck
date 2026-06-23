# CLI Reference: Infer Invariants

Start with [Use TrainCheck](usage-guide.md) if you want the full workflow. This page explains the `traincheck-infer` command.

`traincheck-infer` reads traces from known-good runs and writes invariants. The checker uses those invariants to detect behavior that differs from the reference runs.

## Basic Usage

Infer invariants from one reference trace folder:

```bash
traincheck-infer -f reference_trace -o invariants.json
```

Infer from multiple reference trace folders:

```bash
traincheck-infer \
  -f reference_trace_1 reference_trace_2 reference_trace_3 \
  -o invariants.json
```

TrainCheck reads files named like `trace_*.json` and `proxy_log.json` from each folder.

## Choosing Input Traces

Choose traces from runs that should be correct. A short run is usually enough because training loops repeat the same API patterns many times.

Use multiple reference traces when the target pipeline uses behavior that one reference run does not cover, such as mixed precision, distributed training, gradient clipping, or a different optimizer.

## Useful Options

- `-f, --trace-folders`: trace directories produced by `traincheck-collect`.
- `-t, --traces`: individual trace files.
- `-o, --output`: invariant file path. The default is `invariants.json`.
- `--disable-relation`: skip specific invariant relation types.
- `--enable-relation`: infer only specific invariant relation types.
- `--disable-precond-sampling`: disable example sampling during precondition inference.
- `--precond-sampling-threshold`: set the precondition sampling threshold.
- `-b, --backend`: choose `pandas`, `polars`, or `dict` for trace processing.

Run the command help for the complete option list:

```bash
traincheck-infer --help
```

## Relation Filtering

Relation filtering is useful when a reference trace overfits to ordering details that do not matter for your target run.

For example, disable ordering-based relations:

```bash
traincheck-infer \
  -f reference_trace \
  -o invariants.json \
  --disable-relation FunctionLeadRelation FunctionCoverRelation
```

Enable only specific relation types:

```bash
traincheck-infer \
  -f reference_trace \
  -o invariants.json \
  --enable-relation APIContainRelation ConsistencyRelation
```

## Invariant File

The output file is JSON Lines: one invariant per line. Each invariant describes a relation that held in the reference trace, plus a precondition that says when the relation applies.

Example:

```json
{
  "text_description": "torch.optim.optimizer.Optimizer.zero_grad contains VarChangeEvent torch.nn.Parameter, pre_value: non_zero, post_value: None",
  "relation": "APIContainRelation",
  "params": [
    {
      "param_type": "APIParam",
      "api_full_name": "torch.optim.optimizer.Optimizer.zero_grad"
    },
    {
      "param_type": "VarTypeParam",
      "var_type": "torch.nn.Parameter",
      "attr_name": "grad",
      "pre_value": "non_zero",
      "post_value": null
    }
  ],
  "num_positive_examples": 200
}
```

This invariant says that `Optimizer.zero_grad()` normally clears parameter gradients in the observed context.

## Next Step

Collect a target trace with the invariant file:

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```
