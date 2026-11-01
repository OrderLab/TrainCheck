# Distributed checkpoint continuation pilot

This is a synthetic development experiment for measuring whether optional
TrainCheck observations help an agent continue a training-engineering task.
It is not an official RSI benchmark result. It does not use invariant inference.

The solver receives the previous and current pipelines, existing checkpoints,
and measured training/validation curves. The prompt does not identify an error
or supply a target checker. The control can use ordinary source inspection,
logging, assertions, and custom diagnostics. The treatment additionally receives
the generic TrainCheck observation guide and instrumenter source.

## Components

- `task/`: a fixed small Transformer sequence-tagging workload, per-rank data and
  RNG streams, full-snapshot reference backend, and layer-sharded current backend.
  The current backend intentionally represents the initial task implementation;
  it is not a change to TrainCheck's production checkpoint behavior.
- `TASK.md` and `DECISION_SCHEMA.json`: the neutral continuation instructions and
  structured run/finish interface. Standard training results and explicitly
  requested diagnostic outputs have a declared, equal delivery policy.
- `decision.py`: strict validation of decisions and bounded requests.
- `run_worker.py`: per-rank output setup before candidate or TrainCheck imports.
- `cli_turn.py`: fresh/resumed native CLI execution, raw event preservation,
  monotonic receive timestamps, bounded process cleanup, and failure receipts.
- `accounting.py`: validated usage counters, per-turn differences for cumulative
  sessions, timing intervals, and allocated GPU accounting. Cached input and
  reasoning tokens are subsets, not additional tokens.
- `evaluate.py`: operator-only evidence validation. It never imports candidate
  code. It verifies source/artifact manifests and compares logically named model
  and optimizer state, parameter-group contracts, RNG/progress, and continuation
  trajectories on every rank. Unavailable evidence is an error, not a pass.

## Experiment boundaries

Freeze the task, model configuration, prompts, source hashes, evaluator, scoring
rubric, and budgets before starting fresh solver sessions. Preserve unsuccessful
qualification configurations. Give both arms the same existing checkpoints;
testing only a candidate's newly written checkpoints does not establish backward
compatibility.

Every run or remote diagnostic consumes the same request allowance. Record
failed requests, transport/setup time, measured queue waits, and allocated GPU
seconds. Retain raw wall time and report any queue-adjusted quantity separately.
Allocated GPU seconds include worker startup and the experimental state-export
harness; they are not measured GPU utilization or pure training time. Setup,
qualification, and private evaluation costs belong in separate totals.

Record first causal localization and the first submitted implementation that
later passes independent evaluation. A sufficient source-based explanation can
qualify as localization; a dynamic trace is not required. Turn-boundary usage is
authoritative, whereas a statement within a turn has only a usage interval.
First correct submission and the agent's final selected submission are different
milestones. Evaluate only after every solver has frozen its final selection;
private grading must not become development feedback.

Machine-specific transport/orchestration, raw sessions, traces, checkpoints, and
their manifests belong in the isolated experiment evidence directory. Solver
packets exclude this README, the private evaluator, operator protocol, and
regression tests. Packet read-scope rules are instructions, not an OS isolation
guarantee. The CLI uses its normal workspace write restrictions.

## Validation

Use the repository's configured hooks and explicitly type-check these reusable
tools and their tests: the current pre-commit mypy scope does not select `tools/`.
The focused suite is `tests/test_checkpoint_*.py`. It exercises malformed
decisions, cumulative-counter errors, subprocess timeouts/cleanup, rank output
separation, checkpoint corruption and compatibility, source/artifact integrity,
and full-state comparisons. GPU qualification is separate from local tests.

For operator scoring, run:

```sh
python -m tools.checkpoint_pilot.evaluate --manifest evaluation-input.json \
  --output evaluation-result.json
```

The manifest supplies the frozen contract and four `RunEvidence` records:
reference prefix/resume and candidate prefix/resume. For an existing-checkpoint
test, candidate prefix evidence describes the original supplied checkpoint,
while candidate resume evidence describes the submitted loader. Exit codes are
0 for passed, 1 for violated, and 2 for invalid or unavailable evidence.
