# Index function-call time windows

Repeated calls to `query_func_call_events_within_time` previously filtered every
trace row by type, time, process, and thread before grouping the selected calls.
The new lazy index uses binary search within each process/thread pair, restores
the selected rows to their original order, and retains the existing pandas
`groupby`, record construction, and assertions. A conservative fast path handles
ordinary int64 identities with int64 or float64 time; all unsupported dtypes,
query scalar types, and comparison cases use the original pandas mask.

## Complete saved-trace inference

Three alternating sequential pairs on labmach reduced median total wall time from
**101.796 s to 76.748 s (1.326x, 24.61% less time)**. Both variants include the
identity cache and the independent pruning-state correction from PR #145. The
only production difference between the measured snapshots is the time-window
index in `trace_pandas.py`.

| Pair | Execution order | Baseline wall | Candidate wall | Baseline peak PSS | Candidate peak PSS |
|---|---|---:|---:|---:|---:|
| 1 | Baseline, candidate | 102.655 s | 76.748 s | 2.167665 GiB | 2.168241 GiB |
| 2 | Candidate, baseline | 101.796 s | 76.788 s | 2.129158 GiB | 2.168367 GiB |
| 3 | Baseline, candidate | 101.610 s | 76.658 s | 2.129099 GiB | 2.168411 GiB |

All six runs completed and produced the same **836 invariant records and 104
saved failed-hypothesis records** as the corrected reference. Complete structural
JSON multisets match, preserving duplicate multiplicity and array order. The raw
output files also match byte for byte across the six runs. Full/stage context
files match the corrected reference byte for byte, and all source/input hashes
remained unchanged. Independent review verified all 295 archived files, source
snapshots, complete outputs, contexts, and calculations. Trace integrity checks
ran before and after inference on the remote host; the raw 286-MB input is not
in the downloaded archive and was not independently rehashed locally. This is
an incremental measurement over the corrected cached
baseline; it should not be multiplied by the earlier single-pair identity-cache
speedup as an independently measured combined result.

| Median measurement | Baseline | Time-window index |
|---|---:|---:|
| Total child wall time | 101.796 s | 76.748 s |
| Trace loading/construction | 10.642 s | 10.646 s |
| Reported inference | 86.385 s | 61.853 s |
| Generation and pruning (part of inference) | 73.854 s | 49.225 s |
| Final preconditions and sort (part of inference) | 12.484 s | 12.469 s |
| Peak sampled child-session PSS | 2.129 GiB | 2.168 GiB |

The median sampled memory peak increased by **40.15 MiB**. The first baseline
peak was higher than the other two and close to its paired candidate peak;
per-run values above retain that variation. Each trial uses a fresh process and
trace, including the first lazy index construction. The index stores
O(N) timestamp/position values and requires temporary build storage. It is a
runtime/memory tradeoff, and its first-query cost matters for short-lived traces.

The input is both complete trace files from the saved 6,163,712-parameter,
10-step reference run: 573,494 events and 286,335,289 raw bytes, sequence length
256, batch size 1, seed 42. No training was rerun. A separate probe confirmed that
all 24 sampled real-call queries hit the new fast path, with int64 time/PID/TID
columns and built-in integer bounds/identities. This is sampled eligibility
validation, not a hit count for every query in the timed inference runs.

Both variants use pandas, one worker, native thread settings of one, no GPU,
`PYTHONHASHSEED=0`, and six enabled relations; `FunctionLeadRelation` and
`FunctionCoverRelation` remain disabled. Every run has a 1,800-second cutoff and
12-GiB sampled child-session PSS budget. The runtime is Python 3.11.4,
Torch 2.16.0.dev20261012+cpu, pandas 2.3.3, and NumPy 2.1.0. Torch's source revision
matches the original CUDA wheel, but training used Python 3.12.3 and CUDA 13.0.
The eligible API count is 41; the original name set was not retained, so matching
the count does not establish exact original-environment API coverage.

The host was shared, with no CPU affinity or exclusive reservation. Other owned
experiment jobs were paused during the six timed runs; unrelated background
work was left untouched. Before-run one-minute load averages ranged from 1.29
to 2.25, and sampled whole-host CPU utilization from 3.3% to 4.6%. These snapshots
do not bound interference during each run. The result covers one saved workload
and relation set, not training overhead, distributed scaling, or invariant
ground-truth correctness.

Total wall time includes child startup, imports, output writes, and monitor
polling (about one-second resolution), but excludes integrity hashing. The load
timer includes harness context serialization. The inference timer ends before
invariant JSON writes. Its two phases in the table are log-marker intervals,
not additional time. Memory is sampled once per second for the concurrent child
session, excludes the monitor, and depends on shared pages. The memory budget is
a monitor threshold, not an OS hard limit. cProfile and periodic traceback dumps
were disabled; fatal-error traceback handling remained enabled.

[Compact measurements and provenance](inference-time-range-index-20261025.json)
retain every run's timing, memory peak, background-load snapshots, source/input
hashes, output digests, and archive identity. Raw traces and output records are
retained as study artifacts rather than published in this repository.

## Correctness and cache contract

The 18 new differential tests compare the indexed path with the unchanged pandas
fallback. Together with context/call/identity tests, 61 tests pass locally. The
matching Linux runtime passes those tests plus six common pruning-state tests
(67 total). An independent oracle extracted from the exact parent method checks
144 dtype/offset cases and 6,480 queries with no differences in each of two
pandas/NumPy environments: 2.2.3/1.26.4 and 2.3.3/2.1.0.

The tests cover strict boundaries, tied/reversed bounds, signed integer extremes,
mixed integer/float precision above 2**53, NaNs/infinities/signed zero, malformed
bounds and scalar subclasses, null/extension/categorical dtypes, empty frames,
unsorted rows and duplicate labels, categorical/null call IDs, incomplete or
exception records, fresh returned objects, and cache invalidation. Empty
selections still run the original grouping logic, including its assertions.

Time, event type, process/thread identity values and dtypes, and row order must
remain stable within a cached frame. Replacement or length changes invalidate
the cache; same-size in-place changes require a new `TracePandas`. The cache holds
a source-frame reference rather than copying the DataFrame, but can retain an old
frame until invalidation. Existing pandas group ordering, category/dropna
behavior, within-group row order, assertion messages, and fresh record/event
objects remain unchanged.

## Reproduction

The portable synthetic benchmark needs no saved trace:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.benchmarks.time_range_index --rows 200000 --queries 1000 --repetitions 3

python -m unittest tests.test_context_index tests.test_call_index \
  tests.test_identity_cache tests.test_time_range_index
```

On the matching Linux runtime, its median construction-plus-query time fell from
**2.662321 s to 0.340217 s (7.825x)**. Ordered outputs match in all six trials, and
the input remains unchanged. Candidate first-query median was **0.076523 s**
versus **0.003568 s** for legacy filtering, exposing the lazy build cost. Input
generation, garbage collection, and result hashing are excluded; fresh record
construction is included. This synthetic result is separate from full inference.

For the saved-trace measurement, reconstruct the baseline from commit
`99d16905296d83495570fdcafa91370b242d2528` plus only the pruning-state correction
hunks from `1b5aedffbeefd4cdd9d30036dbb0a833521526cf`. The candidate adds the production
change at `24e7cb0f7a573ff6225a388eff189d9e4af5e8b5`; the benchmark was checkpointed
at `39152aded8b97ed83988e3292bb89c6e3adcf750`. Match the recorded per-file hashes.
With the retained `inference_replay_fatal_only.py` and `inference_scaling.py`
harness files, invoke each fresh variant as follows, using the two trace files
identified by digest in the evidence JSON:

```sh
"$INFERENCE_PYTHON" "$STUDY/inference_replay_fatal_only.py" \
  --mode infer --source "$SOURCE_SNAPSHOT" --traces "$TRACE_A" "$TRACE_B" \
  --output "$NEW_OUTPUT_DIRECTORY" --timeout 1800 --memory-limit-gib 12
```

Run baseline/candidate, candidate/baseline, baseline/candidate sequentially, with
no `--profile` flag. The harness fixes workers/native threads/hash seed and CPU
mode, verifies source/input hashes, and saves phase/memory/context evidence. The
controller compares complete outputs. `run_pairs.py`, exact invocations, both
immutable source snapshots, full outputs, and the verified harness/helper hashes are
retained in the evidence archive. Reproducing this full-trace result requires
access to the saved trace and retained harness; the synthetic benchmark and
regression tests are self-contained in the checkout.

## Engineering hardening verification

The inference stack now types task payloads and RNG scopes, validates required
worker settings before any worker/global mutation, and types each benchmark's
inputs and measurement output. Identity APIs describe their actual schemaless
pandas scalar values instead of promising integers. The time index uses numeric
containers only after the existing conservative dtype checks. Legacy context
comparison retains one documented pandas scalar `Any` boundary; no values are
coerced to satisfy typing and the cache stability contract is unchanged.

All 97 affected inference tests pass, and the configured Ruff/Mypy/Black/isort
hooks pass for changed files. Pinned Mypy 1.9 also passes the sampling and worker
modules, three benchmark tools, and worker-contract tests with
`--check-untyped-defs --disallow-untyped-defs --disallow-any-generics
--follow-imports=silent --ignore-missing-imports`. New worker-contract tests
produce 18 failures before runtime validation and pass afterward; missing,
unknown, and malformed fields are rejected before state changes.

A new bounded Linux replay of signed inference head `ce25c12`, with the same
separate #145 production patch applied, completed on the original 573,494-event
trace. **All 836 invariants, 104 failed-hypothesis records, and both context files
are byte-for-byte identical to the corrected reference.** Source and input hashes
remain unchanged. This one verification run took 76.794 seconds; it is not a new
paired performance experiment, and the earlier performance claims above remain
tied to their original measured revisions.

The separately composed production source (inference, tracing, and pruning-state
fixes) passes whole-source Mypy on 77 files, Ruff, and isort. Whole-source Black
still reports three unchanged legacy files (`config/config.py`,
`instrumentor/proxy_wrapper/hash.py`, and `instrumentor/proxy_wrapper/subclass.py`);
the original base reports six. Scoped strict checks do not certify fully strict
typing of every legacy body.

[Compact hardening evidence](inference-hardening-validation-20261025.json)
records exact composition, runtime, output hashes, and limits.
