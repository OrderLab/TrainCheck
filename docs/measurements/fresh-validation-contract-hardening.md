# Fresh validation: typing and report contracts

This hardens the reusable harness introduced in PR #148. The comparison base is
`27e67aafc1a526ed8e69e0a8e59ff30b8ff827cb`. Frozen predicates, training workloads,
and the original measured results are unchanged by this pass.

## Problem and change

All 36 original helper functions had incomplete type signatures. Worker/CLI
inputs, model factories, phase results, mutation reports, checker observation
state, and study comparisons now have explicit contracts. The eight helper
modules now contain 61 fully annotated function definitions, including boundary
validators. The new contract tests are annotated too.

JSON enters through `object` and is narrowed at required-field boundaries.
Missing fields, wrong types, out-of-range invariant indexes, incomplete phases,
and loss rows inconsistent with the claimed step count fail explicitly. The
single `cast` documents a checked string-keyed, read-only mapping boundary;
there are no new `Any` annotations or type-checker ignores.

The observer now requires each family's exact identity field. API containment
requires `parent_func_call_id`; the three single-call consistency/threshold
families require `func_call_id`; distinct-argument decisions require both event
records and their string call IDs. It never substitutes another family's field
or `None` when checker source changes. The proxy-skip flag is also required when
that precondition is requested.

`ObservationReport` uses optional detail fields while the mutable report is being
constructed. `finalize_report` enforces the conditional schema before returning
or aggregating it: completed checks require their checker payload and boolean
outcome; observed coverage requires counters, groups, decisions, source metadata,
and the family-specific call/pair fields. A captured observer failure remains
primary even if the checker catches it and later returns or raises a different
exception. The latter exception remains available as secondary error evidence.

The benchmark adapter validates dynamically loaded model classes and initializer
callbacks. Its helper-import bootstrap appends the harness path, preserving the
separately composed TrainCheck source selected by `PYTHONPATH`.

## Report compatibility

Coverage output now declares `schema_version: 2`. Existing completed-check fields,
checker payloads, raw identities, and decision records keep their meaning.
`coverage_status` distinguishes `observed`, `disabled`, `unsupported`, and `error`.
`observer_error` is separate from `checker_error`; either causes a nonzero CLI
result rather than a successful validation.

Aggregate active/dormant/identity-active totals are `null` when any coverage is
unobserved. `coverage_unobserved_invariants` and `observed_*` subset counts explain
that limit. Likewise, unknown checker outcomes produce a `null` violation total,
with `check_outcome_unknown_invariants` and `observed_violated_invariants` retained.
A direct checker run can have a known outcome while its coverage remains unknown.
These are deliberate schema corrections, not new performance measurements.

## Evidence

The [before/after fixtures](data/fresh-validation-contracts-before-after.json) run
both original and hardened observers against three controlled checker-source
changes: a renamed identity local, a `None` identity, and a checker that catches
the observer exception. Originally all three were labeled `completed` and emitted
one `func_call_id: null` decision. After hardening, all three report
`observer_error` and emit no fabricated decisions: 3/3 false completions become
0/3. A fourth regression covers a subsequent different checker exception.

Validation on Python 3.10.16, Torch 2.2.2, and Pandas 2.2.3:

- All 28 targeted tests pass: the 13 original checker/fault tests and 15 contract
  tests, including observer/direct parity, dormant rules, early failure,
  required identities, malformed reports, and composed-source import precedence.
- Pinned Mypy 1.9.0 checks all eight helper modules and the new test module with
  `--check-untyped-defs --disallow-untyped-defs --disallow-any-generics
  --warn-return-any --explicit-package-bases --follow-imports=silent
  --ignore-missing-imports`. Its `--python-executable` points at the installed
  TrainCheck runtime, so Torch annotations resolve. This is a stronger scoped
  check, not a claim of repository-wide `--strict` compliance.
- Configured Ruff 0.3.3, Black 24.3.0, and isort 5.10.1 hooks pass on the changed
  paths. The configured mypy hook does not select `tools/`; the explicit command
  above covers that gap. `git diff --check` passes.

Run the focused tests with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest tests.test_fresh_validation_coverage \
  tests.test_fresh_validation_fault tests.test_fresh_validation_contracts -q
```

The source-builder recipe remains pinned to its original, reproducible composition.
Full saved-trace replay and the newly composed PR stack require separate integration
validation. This change makes no new training-overhead or generalization claim.

## Updated production-stack integration

An isolated snapshot combines signed inference `ce25c12`, tracing `b64023b`,
and only the #145 production state-isolation patch from `aec748e`, plus the exact
signed harness source `f3804c0`. It does not change this PR's Git history or its
reproducible original source-builder recipe. The combined production passes 140
Linux regressions; all 28 harness regressions also pass against that source.
Stronger Mypy 1.9 checks pass all nine harness/test files against the updated
production interfaces with actual Torch type resolution.

All eight frozen saved-study checker runs (reference, same-seed control,
holdout, and controlled fault for each workload) complete successfully with
unchanged sources/inputs and the expected production module path. Every original
semantic field in **3,072 per-rule reports** matches, including decisions,
coverage, raw trace IDs, checker payloads, and summary counts. Additive schema-2
fields are allowed; meaningful fields and array order are never dropped. Retained
JSON NaN tokens compare as NaN rather than using Python's non-reflexive float
equality. Operational process IDs and changed harness/source provenance are
separate from semantic equality.

| Workload | Active rules in each run | Reference violations | Control | Holdout | Fault |
|---|---:|---:|---:|---:|---:|
| MNIST | 204 | 0 | 0 | 0 | 3 |
| Transformer | 564 | 0 | 2 | 2 | 4 |

The Transformer's two documented healthy false alarms remain visible. These are
saved-input checker replays, not new model training or a generalization claim.
[Compact integration evidence](data/fresh-validation-hardening-integration.json)
records exact source composition, runtime, counts, and comparison rules.
