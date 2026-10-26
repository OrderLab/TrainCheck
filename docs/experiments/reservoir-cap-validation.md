# Example reservoir cap validation

Iterable construction and stage/cross-trace merging previously appended retained examples directly. This bypassed `MAX_EXAMPLE_LIST_SIZE` and left the represented population count stale. The fix uses bounded insertion and merges according to the original stream populations, preserving duplicate occurrences and exact concatenation order when complete inputs fit.

Results below were recorded on 2026-10-13. The pending source and saved evidence hashes were independently rechecked on 2026-10-25. The baseline is `9eb4f320d3db34d372145261704b805b88e39bda`; combined validation includes the deterministic sampling change in #139.

## Controlled fixture

The fixture sets capacity to 2, merges two streams containing ten occurrences each, and separately constructs a reservoir from five occurrences.

| Measurement | Before | After |
|---|---:|---:|
| Retained after merge | 4 | 2 |
| `num_seen` after merge | 10 | 20 |
| Retained from five-item iterable | 5 | 2 |
| `num_seen` for iterable | 5 | 5 |

These measure reservoir correctness. They do not measure whole-process memory savings.

## Regression and replay checks

Run the portable regression suite from the repository root:

```bash
python -m unittest discover -s tests -p test_example_reservoir.py -v
```

All 21 reservoir tests passed on Linux with Python 3.10.16, PyTorch 2.2.2+cu121, pandas 2.3.3, and NumPy 1.26.4. Coverage includes cap/count accounting, duplicates, atomic rejection, continuation RNG state, stage/cross-trace call sites, actual inference with 1/2/4 workers under fork and spawn, and fixed-seed marginal-inclusion sweeps. With #139, all 21 reservoir and 11 sampling tests passed; its sampling fixture explicitly disables the cap to exercise precondition downsampling separately.

Nine complete saved-trace inference replays used one pass per workload and worker count, with native thread counts set to 1 and `FunctionLeadRelation`/`FunctionCoverRelation` disabled. Every invariant and failed-hypothesis JSON multiset matched across workers and against sampling-only outputs; multiplicity and array order were retained in comparisons.

| Workload | Workers tested | Invariants | Failed hypotheses |
|---|---|---:|---:|
| MNIST | 1, 2, 4 | 193 | 21 |
| Small transformer | 1, 2, 4 | 518 | 10 |
| DCGAN | 1, 2, 4 | 512 | 86 |

Input and source hashes remained unchanged. Compact counts, canonical output hashes, and source/evidence hashes are recorded in [reservoir-cap-validation.json](reservoir-cap-validation.json). Its combined-source manifest and test hashes identify the validated combination with #139, including the uncapped sampling fixture. Raw saved traces are external to this note; the synthetic regression suite runs directly from the checkout.

Publication validation on 2026-10-25 reran all 32 combined tests successfully on macOS. Production Python ASTs match the previously validated combined source; publication changes are formatting and narrow lint comments for intentional exact-integer checks.

## Scope

The merge RNG must advance independently of source selection and must not reset across nested merges. Existing equal-seed source reservoirs can still be correlated: the guarantee is marginal occurrence inclusion, not a uniform joint sample across correlated reservoirs.

Optional containment generalization is skipped when an input has already been sampled, because discarded set overlap cannot be reconstructed. Complete inputs retain exact set deduplication before capping the known union. Consequently, generalized hypotheses can change above the cap. Temporary unions, total hypothesis counts, and total process memory are outside this per-reservoir bound.
