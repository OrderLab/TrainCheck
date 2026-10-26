---
hide:
  - navigation
  - toc
---

<div class="hero" markdown="1">
  <img alt="TrainCheck" width="360" src="assets/images/traincheck_logo.png">
  <p><strong>Invariant Checking for AI Training</strong></p>
  <p>Learn normal training behavior from a healthy run, then catch silent bugs in a target run.</p>

  [Use TrainCheck](usage-guide.md){ .md-button .md-button--primary }
  [Install](installation-guide.md){ .md-button }
  [5-Min Tutorial](5-min-tutorial.md){ .md-button }
</div>

TrainCheck catches silent training bugs by tracing PyTorch API calls and model state changes. You give it a reference run that behaves correctly. TrainCheck infers invariants from that run, then checks a target run for violations.

## Start with This Workflow

### 1. Collect a Reference Trace

```bash
traincheck-collect \
  --pyscript reference.py \
  --models-to-track model \
  --output-dir reference_trace
```

### 2. Infer Invariants

```bash
traincheck-infer -f reference_trace -o invariants.json
```

### 3. Collect a Target Trace

```bash
traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --invariants invariants.json \
  --output-dir target_trace
```

### 4. Check the Target Run

Run the live checker while the target training job writes traces:

```bash
traincheck-onlinecheck -f target_trace -i invariants.json
```

The easier offline path is to check after trace collection finishes:

```bash
traincheck-check -f target_trace -i invariants.json
```

Both checkers write violation logs and a `report.html` summary.

## When to Use TrainCheck

- You changed a training pipeline and want to catch silent logic errors early.
- A run behaves strangely, but normal metrics do not explain why.
- You want to compare a target run against a healthy reference run or an official example.
- You need lower-overhead tracing for a long run; use selective collection with `--invariants` and step sampling.

## Documentation

- [Use TrainCheck](usage-guide.md)
- [Installation Guide](installation-guide.md)
- [5-Minute Tutorial](5-min-tutorial.md)
- [CLI Reference: Collect](instr.md)
- [CLI Reference: Infer](infer.md)
- [CLI Reference: Check](check.md)
- [Technical Documentation](technical-doc.md)
- [Performance Benchmarks](benchmarks.md)
