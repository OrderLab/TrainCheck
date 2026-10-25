# Cache process and thread identities during precondition inference

`get_meta_vars` checks process and thread membership for each request. Previously,
each check repeated `dropna().unique().tolist()` over the full event column.
The change caches those exact pandas results lazily per trace and returns a fresh
list to each caller. Null filtering, scalar types, encounter order, and independent
process/thread membership remain unchanged. Frame replacement or length changes
invalidate both identity caches.

## Isolated measurement

On a deterministic 573,494-row synthetic trace, the median time for construction
plus 1,000 `get_meta_vars` calls fell from **9.489831 s to 0.012004 s** across three
alternating repetitions. Each trial starts with a fresh trace and includes the
first cache fill. The baseline restores only the two original identity methods;
both variants use the same constructor and other methods.

| Repetition | Legacy | Cache |
|---|---:|---:|
| 1 (legacy first) | 9.256223 s | 0.014135 s |
| 2 (cache first) | 9.489831 s | 0.010777 s |
| 3 (legacy first) | 9.548995 s | 0.012004 s |

The shared input has eight process IDs, sixteen thread IDs, deterministic missing
values, and no context managers. All six ordered outputs match: 750 empty
dictionaries and 250 `None` values; input hashes remain unchanged. Input generation,
garbage collection, and result hashing are excluded. The run used a macOS arm64
workstation, Python 3.10.16, pandas 2.2.3, NumPy 1.26.4, and native thread settings
of one. No CPU affinity or exclusive machine access was imposed.

This is an identity-lookup microbenchmark, not completed invariant inference or
training overhead. Paired full-inference results remain pending. Earlier profiling
attempts terminated with SIGSEGV at scheduled diagnostic timers and produced no
completed profile/output; the candidate stopped before the cache phase. Those
attempts provide no valid performance comparison.

## Correctness and contract

All 43 local context/call/identity regression tests and all 11 identity tests in the
matching Linux runtime pass. Independent review adds 78 differential comparisons
across 13 dtype/value cases, six pickle protocols, and two spawned-worker tasks.
The checks cover nulls, integer precision, scalar types, encounter order, missing
columns, independent process/thread membership, fresh return lists, stage isolation,
frame/length invalidation, serialization, and actual precondition inference.

Identity values, dtypes, and row order must remain stable within a frame. Same-size
in-place edits, column replacement, and reordering require a new `TracePandas`.
The cache stores O(P + T) values; returned lists and membership checks still cost
O(P + T). Its source reference is not a copied DataFrame, but can retain the old
frame until invalidation. No RSS/PSS, peak-memory, or multiprocessing memory-scaling
improvement is claimed.

From the checkout, reproduce the isolated measurement with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.benchmarks.identity_cache --rows 573494 --calls 1000 --repetitions 3

python -m unittest tests.test_context_index tests.test_call_index tests.test_identity_cache
```

Use `--rows 1000 --calls 20` for a smoke test. The script needs no external trace;
it records the checkout commit and source hash and checks output equality.
[Compact evidence and provenance](inference-identity-cache-20261025.json) retain the
measured source commit, runtime, inputs, per-trial timings, and limitations.
