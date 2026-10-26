# Cache process and thread identities during precondition inference

`get_meta_vars` checks process and thread membership for each request. Previously,
each check repeated `dropna().unique().tolist()` over the full event column.
The change caches those exact pandas results lazily per trace and returns a fresh
list to each caller. Null filtering, scalar types, encounter order, and independent
process/thread membership remain unchanged. Frame replacement or length changes
invalidate both identity caches.

## Complete saved-trace inference

On labmach, one sequential unprofiled pair reduced total wall time from
**1,385.656 s to 98.551 s (14.06x)**. Both runs completed successfully. All **835
invariant records and 105 saved failed-hypothesis records** match as complete
structural JSON multisets, preserving duplicate occurrences and array order.
Object key order and record line order are ignored. No records were missing or
extra. Both full/stage context files are byte-identical (711 states each), and
all source and input hashes remained unchanged.

| Measurement | Baseline | Identity cache |
|---|---:|---:|
| Total child wall time | 1,385.656 s | 98.551 s |
| Trace loading/construction | 10.503 s | 10.699 s |
| Reported inference | 1,371.032 s | 83.717 s |
| Generation and pruning (part of inference) | 619.580 s | 72.759 s |
| Final preconditions and sort (part of inference) | 751.403 s | 10.913 s |
| Peak sampled child-session PSS | 2.127 GiB | 2.130 GiB |

The workload is the two complete trace files from the saved 6,163,712-parameter,
10-step reference training run: 573,494 events and 286,335,289 raw bytes, sequence
length 256, batch size 1, seed 42. No training was rerun. Both variants use the
context/call indexes from PR #141; this comparison isolates the identity cache,
with only publication-time imports/type annotations, comments/docstrings, and
formatting also differing.
The independent pruning-state correctness fix in PR #145 is excluded from both
variants and evaluated separately.

Both runs used pandas, one worker, native thread settings of one, no GPU,
`PYTHONHASHSEED=0`, and the same six enabled relations. `FunctionLeadRelation`
and `FunctionCoverRelation` were disabled. Both had a 1,800-second cutoff and a
12-GiB sampled child-session PSS limit. The runtime was Python 3.11.4, Torch
2.16.0.dev20261012+cpu, pandas 2.3.3, and NumPy 2.1.0. Torch's source revision
matches the original CUDA wheel, but the original training used Python 3.12.3
and CUDA 13.0. The eligible API count is 41 in both logs; the original name set
was not recorded, so exact original-environment coverage is not established.

This is one baseline-first pair without CPU affinity or exclusive host access;
it is not a repeated-run speedup estimate. Total wall time includes child startup,
imports, JSON serialization, and monitor polling (about one-second resolution),
but excludes initial/final integrity hashing. The load timer includes harness
context serialization. The inference timer ends before invariant JSON writes;
its two large phases above are log-marker intervals, not additional time. Memory
is sampled every second for the concurrent child session, excludes the monitor,
and depends on shared pages; no memory reduction is established.

The result concerns offline inference over this saved trace and enabled relation
set. It does not establish training overhead, distributed scaling, or correctness
of the inferred invariants against ground truth. Earlier attempts combining
cProfile and periodic traceback dumps exited with SIGSEGV and are excluded. Both
were disabled for this timed pair; its success alone does not identify the crash
cause.

[Full-trace measurements and provenance](inference-identity-cache-full-trace-20261025.json)
include phase times, output digests, trace/source hashes, runtime, and archive
identity. The retained evidence archive contains both complete output sets,
context files, stdout logs, memory samples, and invocation/provenance manifests.

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

This isolates identity lookup. The separate complete inference comparison above
measures the saved training trace.

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
frame until invalidation. No memory or multiprocessing memory-scaling improvement is claimed.

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
