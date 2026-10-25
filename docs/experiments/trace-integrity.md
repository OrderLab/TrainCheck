# Recorder re-entry and buffer integrity

Recorder-owned argument conversion, tensor metadata inspection, and JSON encoding can call instrumented APIs again. The wrapper previously did not honor the conversion helper's local suppression flag, so recorder work could generate more records or recurse. Independently, the producer and periodic flusher accessed the same buffer without synchronization: overlapping flushes could duplicate a batch, and a producer append could be cleared before delivery.

The recorder now uses an exception-safe, thread-local suppression depth around its own work. It releases suppression before executing application code, preserving nested application calls and generator execution. A class context manager preserves frozen framework exceptions. The buffer snapshots mutable values synchronously, then locks append, queue handoff, and clear as one ordered operation. Buffers still belong to individual producer threads for trace routing.

## Controlled impact

These tests deliberately inject a conversion helper or force a thread interleaving; they do not estimate how often an ordinary training job triggers either defect.

| Controlled trigger | Buggy | Fixed |
|---|---:|---:|
| 100 application calls with a wrapped serialization helper | 400 records, including 200 recorder-owned records | 200 records, zero recorder-owned records |
| Two overlapping flushes; 50 trials × 100 records | 10,000 emitted, including 5,000 duplicates | 5,000 emitted, zero duplicates |
| Append during paused flush; 50 trials, 5,050 expected records | 50 records missing | Zero records missing |

The finite re-entry case preserves application return values and raises no `RecursionError`: self-tracing can inflate a trace without an obvious recursion failure. The buffer race can cause loss as well as duplication; valid pre/post pairing alone cannot detect a completely missing pair.

## Ordinary workload control

All 45 runs completed: MNIST CNN, a small Transformer, and reduced-width DCGAN × five conditions × seeds 42/43/44. Each run used five training steps, batch size two, one GPU, and one native CPU thread. Conditions were no instrumentation, the buggy snapshot, guard only, buffer lock only, and both fixes.

None of the 36 instrumented runs triggered either defect. Each instrumented condition had the same record count and raw byte size at every seed:

| Workload | Records per run | Raw bytes per run | Buggy → fixed size change |
|---|---:|---:|---:|
| MNIST CNN | 654 | 472,863 | 0% |
| Small Transformer | 2,006 | 1,564,970 | 0% |
| DCGAN | 4,596 | 3,401,128 | 0% |

All recorded losses and canonical final model-state hashes matched their corresponding uninstrumented baselines. These are adapted micro-workloads with ordinary local tensors on PyTorch 2.2.2+cu121, not full published benchmark reproductions or DTensor coverage. The observations support these correctness repairs; they do not establish a general performance improvement or invariant-checking overhead result.

## Validation and provenance

Run the eleven portable regression tests from the checkout:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_trace_integrity -v
```

The suite covers recorder re-entry, JSON emission re-entry, real tensor snapshots, nested application calls, generators, original and frozen exceptions, thread-local suppression, overlapping flushes, and concurrent producer ordering.

[Sanitized results](trace-integrity-results.json) retain controlled counts, workload settings, base/source revisions, and evidence-file hashes. Measurements used TrainCheck base `9eb4f320d3db34d372145261704b805b88e39bda` and adapted benchmark source `ae6601978217bc0f06b9a78479cf00f0bc7242c9`; all instrumented conditions shared the same dynamic-type-name compatibility patch. No private paths, raw training traces, or checkpoint data are included.
