# Fresh-process invariant validation

Fresh-process checking works for the tested MNIST rules, but exposes two object-ID
false alarms in the small Transformer rules. This is a reproducible offline
validation harness and a bounded result, not a generalization or performance claim.
No production code changes are included.

| Workload | Learned rules | Reference active / violations | Fresh same-seed control active / violations | Longer changed-seed holdout active / violations | Dormant rules | Requested proxy checks skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST, 1,199,882 parameters | 204 | 204 / 0 | 204 / 0 | 204 / 0 | 0 | 0 |
| Transformer, 36,197 parameters | 564 | 564 / 0 | 564 / 2 | 564 / 2 | 0 | 0 |

Every active rule completed at least one actual relation decision. That does not
mean every candidate opportunity was evaluated: for example, MNIST holdout IO
rules have 3,880 opportunities, 3,880 precondition passes, and 3,760 comparisons
because missing values skip 120 comparisons. Rules stop on their first violation.
There were no checker errors or unsupported-coverage results. The direct checker
without observers produced identical CheckerResult structures on both holdouts.

For both workloads, the same-seed fresh control's complete loss trajectory and
initial/final canonical parameter hashes exactly match the reference. The
instrumented 20-step holdout likewise exactly matches its uninstrumented baseline.
These comparisons apply to these CPU runs and are not a claim of universal
instrumentation transparency.

The two Transformer alarms are unconditional APIContain rules asserting that
`clip_grad_norm_` contains `Module.named_parameters` / `Module._named_members` with
particular arguments. Both child calls are present in the failed parent's strict
time interval, on the same process/thread. Their recorded model arguments differ
only in `_TRAINCHECK_src_mask_ID`: the reference's memory-object ID was embedded
literally into both learned child selectors. Fresh processes have different IDs.
Thus both rules fail even on the same-seed control despite identical losses and
parameters. The evidence retains the frozen selectors and raw child arguments;
no IDs or learned rules were rewritten. This is an object-identity portability
failure, distinct from thread-ID precondition portability.

The post-hoc single-field controls activate a passing frozen rule and violate it
at the edited call. MNIST changes one recorded output dtype from `torch.float32`
to `torch.float64`: one output rule plus two IO rules fail (3 new violations).
Transformer changes one output shape from `[16, 2, 32]` to `[17, 2, 32]`: two rules
newly fail, in addition to the 2 existing healthy alarms (4 total). These are
trace-metadata sensitivity controls, not injected training bugs. They establish
neither broad defect coverage nor the consistency of an alternative tensor execution.

The [compact evidence](data/fresh-run-invariant-validation.json) records source
commits/hashes, data hashes, actual runtime, paired parameter hashes/losses, raw
process/thread IDs, fault edits, family totals, and explicit failed-attempt history.
The [MNIST ledger](data/fresh-run-invariant-validation-mnist.jsonl) and
[Transformer ledger](data/fresh-run-invariant-validation-transformer.jsonl) retain
every rule's opportunities, group-level precondition counts, completed decisions,
visited/unvisited counts, violations, and zeros for all four checked traces.
Their run arrays use the evidence JSON's `coverage_columns`; null means not
applicable/unobserved. `fully_validated_active` means an active rule without a
requested proxy-check skip; it does not promise all candidates were checked.

Final runs use signed harness `cad62cc98ef5fb7b313325443caa33f715ed141e`, Python
3.11.4, PyTorch `2.16.0.dev20261012+cpu`, pandas 2.3.3, and NumPy 2.1.0. Thirteen
focused counter/mutation tests pass locally and on that Linux runtime; 29 composed
production-stack tests also pass there. Both complete study commands exited 0,
with all input/source immutability and comparison gates satisfied. The
`study_completed` status means those gates passed; it does not erase the two
healthy Transformer violations.

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

Each phase starts a new process, with native threads limited to one. A monitor
polls every 250 ms and stops the process group after a 600-second timeout,
8 GiB process-group RSS threshold, or 256 MiB API-trace threshold. These are
sampled monitor thresholds, not OS hard resource limits. Inference is
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
original TorchTitan rule set's portability unresolved. Neither new workload has
any process/thread-ID-bearing inferred clause. The frozen `identity_bearing_*`
counters measure only PID/TID precondition clauses, not object IDs in API arguments. In particular, these results do not
validate the original 836-rule corpus's three literal-thread-ID APIContain rules
or its fourth nonliteral thread-inequality rule.

A post-hoc sensitivity control selects a healthy, actually evaluated output-shape
rule, or an output-dtype rule when no eligible shape rule exists. It changes one
recorded shape dimension or float32/float64 dtype value, retaining its JSON type,
IDs and precondition inputs. Every other semantic field and every other raw line
must remain identical. The frozen checker must activate and violate the selected
rule. This is trace-level detector-path validation, not a reproduced training bug
or a general defect-coverage result. Dtype mutation changes recorded metadata only;
the tensor type-name key remains unchanged, so it does not represent a complete
alternative tensor execution. The original 74b3adf MNIST run completed its healthy
phases but stopped at the shape-only selector; that diagnostic remains separate
from fresh runs with the revised selector.

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
  python -m unittest tests.test_fresh_validation_coverage tests.test_fresh_validation_fault -v
```
