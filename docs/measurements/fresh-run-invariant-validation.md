# Fresh-process invariant validation

This harness learns rules from one small training run, then checks the frozen
rules against separate OS processes: a same-seed control and a longer run with a
different initialization/training seed. Final measurements are pending. Earlier
MNIST pilot runs are diagnostics and are not final validation results.

The initial workloads are benchmark-derived MNIST and a small Transformer.
`benchmark_workload.py` extracts the named model definitions from a supplied
TrainCheck-Benchmarks checkout; it never executes the benchmark's top-level CLI
or downloads data. MNIST requires cached real IDX data. The Transformer uses the
checked-in WikiText prefix. See `tools/fresh_validation/ADAPTERS.md` for the original
adaptations. The fixed `data_steps=20` prepares the same corpus/vocabulary/model
shape for every run: reference/control train 10 steps with seed 42; holdout and its
uninstrumented baseline train 20 steps with seed 43. The first half of holdout inputs
overlaps reference inputs; this is not disjoint-dataset validation.

`prepare_source.py` creates an isolated snapshot from signed source revisions:
identity/cache stack 99d1690, tracing integrity 7d6c683, TorchScript 9ac61d1,
precondition-state fix 1b5aedf (only its two production hunks), and dynamic type-name
fix 2f034ae. The existing deterministic sampling/reservoir changes are retained.
The harness PR contains no production-code change. Earlier failed setup and
historical-compatibility-patch pilots remain separate from this final recipe.

Example from a checkout containing those commits, on Linux with an existing
compatible CPU PyTorch runtime and benchmark/data files:

```sh
python tools/fresh_validation/prepare_source.py --repo "$PWD" --output /path/to/new-source
python tools/fresh_validation/run_study.py \
  --source /path/to/new-source --python /path/to/runtime/bin/python \
  --benchmark-root /path/to/TrainCheck-Benchmarks --data-root /path/to/MNIST-cache \
  --output /path/to/new-study --workload mnist
# Use --workload transformer and omit --data-root for the language-model adapter.
```

Each phase starts a new process, with native threads limited to one, a 600-second
timeout, an 8 GiB process-group RSS limit, and a 256 MiB API-trace limit. Inference is
single-worker and disables FunctionLeadRelation/FunctionCoverRelation. These
bounds and observed phase times are operational controls, not performance data.

Coverage calls the real `inv.check(trace, check_relation_first=False)`. It counts
precondition group verifications separately from actual relation decisions using
AST-located line observations of the installed checker. API opportunities are
calls; DistinctArgument opportunities are within-step argument pairs. The checker
stops at the first violation per invariant, so unvisited remainders are explicit.
A `triggered` flag or zero violations alone does not establish active checking.
Missing tensor values, dormant rules, checker errors, unsupported coverage, and
requested-but-skipped proxy-variable checks are distinguished. Direct checking
without observers must produce exactly the same CheckerResult structures.

Raw process/thread IDs and frozen predicates are preserved. ID-bearing clause
kinds/values and activation are reported; their mere presence does not establish
that an ID caused dormancy. A workload with no ID-bearing rules leaves the
original TorchTitan rule set's portability unresolved.

A post-hoc sensitivity control selects a healthy, actually evaluated output-shape
rule and changes one recorded tensor's first shape dimension, retaining its type,
IDs and precondition inputs. Every other semantic field and every other raw line
must remain identical. The frozen checker must activate and violate the selected
rule. This is trace-level detector-path validation, not a reproduced training bug
or a general defect-coverage result.

`compare_study.py` verifies source/input immutability, fresh producer PIDs, exact
same-seed control losses and canonical parameter hashes, instrumented-versus-plain
holdout losses/hashes, observer/direct equality, and the targeted fault response.
Its `study_completed` status certifies these evidence gates. It does not assert
that every invariant passed the healthy runs; those violations are retained for
analysis. Per-family reports distinguish all active decisions from rules whose
requested checks were fully executed.

Focused counter tests:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest tests.test_fresh_validation_coverage -v
```
