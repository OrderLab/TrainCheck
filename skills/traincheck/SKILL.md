---
name: traincheck
description: Diagnose PyTorch training behavior with TrainCheck reference traces, inferred invariants, and offline checks. Use when an agent needs evidence about a suspected training bug or the behavioral effect of an allowed training change.
---

Use TrainCheck to obtain evidence for the user's training task. A violated
invariant is a difference from its reference, not automatically a bug or a reason
to undo an intentional algorithm change. The task's evaluator remains the measure
of success.

## Establish the experiment

- Use the supplied TrainCheck source and Python environment. Keep the source on
  `PYTHONPATH`; the coverage observer below is a repository tool, not a wheel entry
  point. Do not install this skill globally during a treatment/control experiment.
- Use a legitimate public baseline, an authorized known-good run, or supplied
  reference artifacts. Never obtain a reference from hidden tests or a withheld
  solution. Record the reference's code, data, configuration, and framework
  versions. The Llama experiment's invariant file is not a universal rule set.
- Start with a short run that reaches the operation of interest. Use matching
  runtime/data/configuration for reference and candidate. Check a fresh healthy
  run before treating an alarm as a regression; object IDs, thread IDs, and CUDA
  stream handles can make learned rules process-specific.
- Keep benchmark code changes within the task's allowed edit scope. If training
  code cannot be changed, run instrumentation in an isolated diagnostic copy and
  submit only permitted artifacts. Instrumented training time is not an ordinary
  benchmark score.

## Collect, infer, and check

Example commands, using absolute paths and fresh output directories:

```sh
export PYTHONPATH="$TRAINCHECK_SOURCE"
"$PYTHON" -m traincheck.collect_trace --pyscript "$REFERENCE_SCRIPT" \
  --output-dir "$RUN/reference" --sampling-interval 1 --warm-up-steps 0
"$PYTHON" -m traincheck.infer_engine -f "$RUN/reference" \
  -o "$RUN/invariants.json" --num-workers 1 \
  --disable-relation FunctionLeadRelation FunctionCoverRelation
"$PYTHON" -m traincheck.collect_trace --pyscript "$TARGET_SCRIPT" \
  --invariants "$RUN/invariants.json" --output-dir "$RUN/target" \
  --sampling-interval 1 --warm-up-steps 0
"$PYTHON" -m traincheck.checker -f "$RUN/target" \
  -i "$RUN/invariants.json" --output-dir "$RUN/check" --num-workers 1
```

This example collects API evidence. For a task that needs parameter/gradient
updates, add `--models-to-track MODEL_VARIABLE --model-tracker-style subclass` to
both collection commands, then verify the generated policy and actual variable
records. A requested tracker alone does not prove those checks ran. Distributed
FSDP/DTensor model tracking and concurrent checking have separate compatibility
limits; use the offline path for the initial pilot. Do not switch to a different
tracker to make a failed comparison pass.

The two ordering relations are disabled in this bounded starting recipe to match
the validated offline workflow; record any chosen relation filter. Add a needed
family deliberately rather than silently changing rules after seeing a result.
For launchers or script arguments, inspect `collect_trace --help` and the supplied
task adapter. Do not guess a different training entry point.

Measure activation with the existing observer, using the complete trace file
list from that run, including variable traces when present:

```sh
"$PYTHON" "$TRAINCHECK_SOURCE/tools/fresh_validation/check_coverage.py" \
  --traces "$RUN/target"/trace_*.log --invariants "$RUN/invariants.json" \
  --output "$RUN/coverage.json"
"$PYTHON" "$TRAINCHECK_SKILL/scripts/summarize_check.py" \
  --coverage "$RUN/coverage.json" \
  --collection-policy "$RUN/target/instr_opts.json" \
  --output "$RUN/agent_report.json"
```

Inspect actual output filenames before invoking the observer; do not pass an
unexpanded glob or omit a `proxy_log.json` required by the chosen tracker. The
summary helper optionally accepts `--violations-summary` pointing to the direct
checker's `violations_summary.json`. That is a count cross-check, not a claim of
complete result equivalence. Complete evidence stays in the hashed source files.

## Interpret and act

- Exit code 0 from the checker means it ran, not that training is correct. Read
  `agent_report.json`: observed violations, active passes, dormant rules, unknown
  outcomes, skipped variable checks, and observer/checker failures are distinct.
  A report with no completed relation decisions provides no checking evidence.
- For an alarm, inspect its `per_invariant` entry in the original coverage file:
  the complete `checker_result`, invariant, trace records, and decision identities
  identify what differed. The compact summary lists at most ten problem entries;
  it preserves the source file and tells you how many remain.
- Keep literal-ID alarms visible and explain their portability limitation. Do
  not delete rules, rewrite IDs, or replace missing values to manufacture parity.
  Check whether the same alarm appears on a fresh healthy control.
- If the task intentionally changes an optimizer, dtype, model shape, or other
  learned behavior, re-evaluate whether the reference is applicable. Record that
  mismatch explicitly. Do not infer new rules from an unverified candidate and
  use those rules to certify the same candidate.
- Make a hypothesis from the evidence, apply a task-permitted fix, rerun the
  relevant check, and separately run the task's real evaluator. Report diagnosis,
  coverage, result, and total cost without equating fewer alarms with a better
  model. If no relevant rule activates, say so and use another diagnostic.

Keep raw traces, checkpoints, and `env_dump.txt` local to the experiment. The
collector records environment variables; share the selected diagnostic evidence,
not the entire output directory by default.
