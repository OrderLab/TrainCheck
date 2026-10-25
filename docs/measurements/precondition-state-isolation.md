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
