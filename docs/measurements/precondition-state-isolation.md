# Isolate pruning state between independent precondition solves

`find_precondition_from_single_group` used a mutable default set for
`_pruned_clauses`. A clause rejected as uninformative for one solve remained in
that set and was removed from later independent solves in the same process.
This can suppress a valid precondition based on prior process history, even
without sampling or RNG changes.

The fix uses a `None` default and allocates a fresh set only when no set is
provided. Recursive calls still receive the same explicit set, and callers that
supply a set retain its update/reuse behavior. No recursion or pruning rule changes.

## Minimal reproduction

Each synthetic example has one record with the same process/thread/function.
Solve A has phase `train` in both positive and negative examples; its signal is
`True` for positives and `False` for negatives. A correctly learns
`signal == True` and prunes `phase == train` as irrelevant to A.

Independent solve B has positive phase `train` and negative phase `eval`.
It needs precisely the phase clause that A pruned. Before the fix, running B
after A loses that clause and returns no precondition. After the fix, B returns
`phase == train`.

| Sequence in a fresh process | Before: preconditions per call | After |
|---|---|---|
| B | 1 | 1 |
| A, B | 1, **0** | 1, **1** |
| B, A | 1, 1 | 1, 1 |
| B, B | 1, 1 | 1, 1 |

All examples use the actual `TracePandas`, clause construction, pruning, and
verification code. Python RNG state stays unchanged in every sequence. These are
fresh-process versus warm-process histories, not a multi-worker-count matrix.
Because workers retain Python module state across tasks, this defect can make a
solve depend on which earlier tasks ran in that process, independently of RNG.

## Observed changes in a complete configured replay

A separate correctness replay applied only this PR's two production hunks to the
identity-cache candidate (`504e04d`), preserving that candidate's earlier RNG fix.
The standalone source and base of this PR are unchanged. Independent inspection
of all 76 production Python files confirmed that only `precondition.py` changed,
exactly as intended. All 17 combined tests passed (11 cache tests and six state
isolation tests).

Both unprofiled runs completed on the same 573,494-row, single-rank trace, with
the same Linux runtime (Python 3.11.4, PyTorch 2.16.0.dev20261012+cpu, pandas 2.3.3,
NumPy 2.1.0), one worker, and one native thread. `FunctionLeadRelation` and
`FunctionCoverRelation` were disabled. Input/source hashes and ordered context
states remained unchanged.

| Output | Cache only | Cache + state isolation |
|---|---:|---:|
| Emitted invariants | 835 | 836 |
| Failed hypotheses | 105 | 104 |

All 940 hypothesis identities match after excluding preconditions and accounting
for the failed-record wrapper's example-count fields. One formerly failed
`APIContainRelation` is now emitted; none moved from emitted to failed. Of the
835 previously emitted records, 833 remain structurally identical and **two gain
additional constraints**. These two differences persist after normalizing clause,
disjunction, and constant-value order; they are not ordering noise.

| API containment predicate | Observed change |
|---|---|
| `Linear.forward` calls `checkpoint._internal_assert` | Previously failed; now emitted with six autocast/context/thread clauses. |
| `RMSNorm.forward` calls `checkpoint._internal_assert` | Retains its step clause and gains six autocast/context/thread clauses (one to seven). |
| `get_rng_state` calls `current_device` | Retains both CUDA-device clauses and gains a thread clause (two to three). |

The two existing predicates become more restrictive through added conjuncts.
All three changed/new predicates include concrete trace-specific thread IDs.
Their applicability to future checking runs is unvalidated; these results do not
establish generalization or ground-truth correctness. This single replay pair is
not a speed or memory comparison. [Compact replay evidence](precondition-state-isolation-replay-20261025.json)
records source composition, checks, output hashes, and the observed structures
without publishing literal thread IDs or raw traces.

## Validation and scope

Six focused regression tests had five failures before the fix and all pass
afterward. They cover independent/reversed/repeated solves, fresh versus warm
processes, caller-supplied sets, and recursion. The recursive fixture splits two
positive groups into two valid preconditions; siblings share one set, while a
new root solve receives a different set. Repository-pinned lint, type, and format
hooks pass.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest tests.test_precondition_state_isolation -v
```

This is an independent correctness fix. Recovering previously suppressed clauses
can add preconditions or otherwise change inference outputs. No unchanged
large-trace output, global order independence, scheduling determinism, or
performance improvement is claimed. The identity-cache performance comparison
uses its own unchanged candidates.

[Compact reproduction evidence](precondition-state-isolation-20261025.json)
records the fixture, before/after counts, source hashes, test results, and limits.

## Engineering contract checks

The regression fixture now has explicit record, example, solve, and recursion
observer types. The observer requires the actual recursive call parameters rather
than accepting arbitrary keyword arguments, and fresh-process comparisons check
the entire expected serialized precondition. The production fix is unchanged.

Pinned Mypy 1.9 reports **14 errors before and zero after** on the fixture with
`--check-untyped-defs --disallow-untyped-defs --disallow-any-generics`,
`--ignore-missing-imports`, and `--follow-imports=silent`. The last option limits
diagnostics to this fixture; it does not certify strict typing of imported legacy
modules. All six regressions pass on Python 3.10.16 / PyTorch 2.2.2 / pandas 2.2.3.
The configured hooks also pass on the production function and fixture; the hooks
exclude tests from most checks, so the explicit fixture check is necessary.

These checks add no new performance measurement. The earlier replay results above
remain tied to their recorded source revisions.
