# TracePandas indexing measurements

The context index removes two full-frame scans per context instance. The call-ID index removes repeated full-frame scans for pre/post lookups, including lookups made before containment sampling. The optimizations preserve selection coverage and record-validation behavior.

The adjacent `trace-pandas-indexes-20261025.json` records sanitized measurements, source/input hashes, environment versions, state equality, profiler call counts, and nine successful small-workload replay results. The complete saved Llama trace contains 573,494 events from ten training steps of a 6,163,712-parameter model.

| Measurement | Before | After |
|---|---:|---:|
| Preparation, original → context index | 170.855 s | 11.714 s |
| Context-index work within preparation | 159.439 s | 0.880 s |
| Pre-record profiling, context → combined | 187.914 s / 6,891 calls | 1.034 s / 6,891 calls |
| Post-record profiling, context → combined | 289.472 s / 13,498 calls | 3.888 s / 13,498 calls |
| Diagnostic wall time to the same error | 546.50 s | 71.98 s |

Preparation is one paired measurement on the same Linux host using one native thread. Total preparation includes state serialization and excludes process startup; context-index timers exclude serialization. Both preparation variants produce the same 711 ordered states in the full trace and stage trace. The profiler's cumulative times overlap across callers and must not be added indiscriminately.

The large-trace diagnostics both fail at `torch._prims_common.is_contiguous` argument binding: replay PyTorch 2.2.2 expects one argument while the PyTorch 2.16.0.dev20261012 trace contains two. These results demonstrate reduced search cost, not completed large-trace inference throughput or online-checker overhead. The replay environment also resolves fewer eligible containment APIs than the collection environment (35 versus 41), so a final reference run needs compatible framework dependencies.

Nine small-workload runs complete successfully: original/context/combined variants for MNIST, small Transformer, and DCGAN. Full invariant and failed-hypothesis JSON match under structural multiset comparison, retaining multiplicity and array order. Per-workload counts are 193/21, 518/10, and 512/86.

## Reproduce local correctness and synthetic measurements

From a checkout with TrainCheck dependencies installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_context_index tests.test_call_index
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.benchmarks.trace_pandas_indexes --output measurements.json
```

The synthetic benchmark constructs its inputs and needs no external trace or cluster. It alternates variant order, checks exact results, and reports every repetition and the median. Its context case times construction plus state serialization; its call case times construction and all lookups, including the first lazy-index build. Its sizes/repetition count are configurable; these synthetic timings are separate from the saved-trace measurements above.

The oracle fixture contains only the exact original context/pre/post/stage methods from commit `9eb4f320d3db34d372145261704b805b88e39bda`; supporting types and the constructor come from the package under test. Provenance includes the original file and method hashes. The fixture's `TracePandas` binding ensures stage splitting creates legacy instances, and the tests assert this explicitly.

The published source also corrects the inherited call-event cache annotation to include `IncompleteFuncCallEvent`; narrow lint comments preserve exact-string guards. These typing/comment changes do not alter lookup behavior.

## Cache contract

Event rows, their order, call IDs, and event types must remain stable after initialization. Frame replacement or a length change rebuilds the call index. Same-size in-place edits or row reordering can leave stale positions and require a new `TracePandas` instance. The tested inference path does not perform those mutations. This change does not introduce general cache invalidation for arbitrary dataframe mutations.
