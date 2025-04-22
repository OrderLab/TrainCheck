# `traincheck-collect`

`traincheck-collect` is the entry point of TrainCheck's workflow. It instruments your PyTorch training script to capture runtime behavior, generating detailed execution traces for later invariant inference and checking.

This utility dynamically wraps key PyTorch APIs and monitors model states, without requiring any modification to your original training code.

Use `traincheck-collect` whenever you need to:
- Generate traces from **reference pipelines** for invariant inference.
- Collect traces from **target pipelines** to check for silent issues using pre-inferred invariants.

---

## 🔧 Basic Usage

The minimal command to trace a training script with both API and model instrumentation.

```bash
traincheck-collect \
  --pyscript <your_entrypoint_script.py> \
  --models-to-track <model_variable_name1> <model_variable_name2> ...
```

Removing --models-to-track will disable variable instrumentation all together.
You can provide `-t --modules-to-instr` to specify libraries you want to instrument, defaulted to `torch`.
To provide additional arguments, you can provide a `--shscript` that invokes your entrypoint.

For

