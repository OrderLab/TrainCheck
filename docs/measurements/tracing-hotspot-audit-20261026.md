# Bounded tracing hotspot audit

The first optimization target is object-metadata conversion. Repeated process/thread
lookups and context retrieval are already cheap in these runs. This investigation
changed no production code and does not claim a tracing speedup.

## Unprofiled observation

Two alternating fresh-process pairs per workload ran three CPU training steps:
plain then traced, followed by traced then plain. Both modes use the same model,
data, training math, and RNG resets. A fifth, separately profiled traced run was
used only for attribution. All ten cases completed successfully.

| Workload | Parameters | Plain three-step wall time, two runs | Traced three-step wall time, two runs | API records per traced run | Raw API bytes |
|---|---:|---:|---:|---:|---:|
| MNIST | 1,199,882 | 0.043102 / 0.029823 s | 0.331755 / 0.332118 s | 520 | 351,073 |
| Tiny Transformer | 21,052 | 0.008537 / 0.004783 s | 0.878434 / 0.822045 s | 1,338 | 1,014,134 |

The ratios of the two medians are 9.10x and 127.66x. These very small denominators,
first-step effects, and two repeats make them bounded diagnostic observations,
not stable overhead estimates for longer training. The timer covers training
bodies, `start_step`, and the wrapper-disable assignment. It excludes imports,
instrumentation setup, data preparation, loss extraction, model hashing, and
process shutdown. No GPU or distributed workload ran.

Tracing used normal torch API coverage (`use_full_instr=False`), unfiltered
argument/return metadata, interval 1, zero warmup, and descriptors disabled.
There were 20,998 installed wrappers, but only 260 and 669 recorded calls in the
two workloads. There was no proxy/model tracking or tensor-value collection.
These are trace-collection measurements, not selective invariant-checking overhead.

## Separate profile

Each cProfile run covered the main training thread during the three step bodies.
Its total recorded time was **0.912044 s for MNIST** and **2.474558 s for the
Transformer**, substantially above ordinary traced wall time. cProfile magnifies
Python-heavy paths; these shares must not be applied directly to unprofiled time.
Background file writing, setup, `start_step`, and process shutdown are outside
the profile. Cumulative rows overlap and must not be added.

| Component | MNIST calls | MNIST share of profiled time | Transformer calls | Transformer share |
|---|---:|---:|---:|---:|
| `convert_var_to_dict`, cumulative | 404 | 94.15% | 1,709 | 97.54% |
| Dumper `safe_getattr`, cumulative and inside conversion | 172,768 | 54.53% | 477,225 | 56.29% |
| All `warnings.py` functions, summed exclusive time | — | 28.06% | — | 28.89% |
| `get_process_thread_id`, cumulative | 260 | 0.020% | 669 | 0.018% |
| `get_meta_vars`, cumulative | 260 | 0.0025% | 669 | 0.0022% |
| Final trace `serialize`, cumulative | 520 | 0.403% | 1,338 | 0.397% |

`get_process_thread_id` caches identities in thread-local state. `get_meta_vars`
returns the existing metadata dictionary; it does not reconstruct context per
call. Conversely, metadata conversion repeatedly enumerates object attributes
and invokes `safe_getattr`, which enters/exits a warning context and installs an
ignore filter for every attribute. The warning-module shares above count its
exclusive function time once, avoiding nested cumulative double counting.

Serialization and snapshot construction execute on the producer; file writes
already use a background queue consumer. The small JSON share here does not
measure file-writer cost, full tensor dumps, GPU synchronization, or larger traces.
It is evidence against prioritizing PID/TID caching or a JSON-library replacement
for this particular metadata-only CPU workload.

## Invocation counts explain the workload difference

The unprofiled medians below cover three CPU training steps and only two runs per
mode. Added time is median traced time minus median plain time; dividing it by
recorded calls is an aggregate normalization, not a measured latency for each API.
Invocation/attribute counts come from the separate profile and its trace; timing
rows use ordinary-run medians.

| Metric | MNIST | Transformer |
|---|---:|---:|
| Plain training | 36.463 ms | 6.660 ms |
| Traced training | 331.936 ms | 850.239 ms |
| Added tracing time | 295.474 ms | 843.579 ms |
| Added time per step | 98.491 ms | 281.193 ms |
| Recorded Python calls | 260 | 669 |
| Calls during the forward region | 66 | 285 |
| Attribute reads via `safe_getattr` (separate profile) | 172,768 | 477,225 |
| Added time divided by recorded calls | 1.136 ms | 1.261 ms |

Transformer has 2.57x the recorded calls, 2.76x the attribute reads, and 2.85x the
absolute added time. Its plain computation is also 5.47x faster in this particular
small configuration. Both numerator and denominator explain the much larger
slowdown ratio; this is not evidence of an order-of-magnitude increase in cost per
intercepted call. Slower computation can dilute percentage overhead without
reducing absolute tracing time. This relationship is conditional: changing device
or execution path can also change tracing costs and the set of calls.

The raw trace contains nested Python API calls, including framework bookkeeping.
`Module.named_modules` accounts for 252/669 Transformer calls (37.7%), reflecting
repeated parameter/module traversal. The forward region has more intercepted
calls, but not all additional events are mathematical operations. Region counts
use exact pre/post call IDs and step metadata; forward is after zero-grad and
before NLL. They are not additive timings or counts of device kernels.

Repeated serialized tensor/Parameter observations number 286 versus 752; module
observations number 24 versus 366. Transformer additionally serializes 282
generator observations and 243 sets. The sets produce empty metadata dictionaries,
yet still traverse the generic inspection path. These are repeated snapshots,
not unique objects. Avoiding redundant work on known types is a candidate;
dropping these records or fields would be a separate coverage change.

## Disjoint profile breakdown and meaning of serialization

The following partition assigns every function's exclusive time exactly once.
Its denominator is the separate profiled main-thread training time, not ordinary
training wall time or measured added overhead. Each column sums to 100% before
rounding. See the [detailed evidence](data/tracing-hotspot-breakdown-20261026.json)
for source profile/trace hashes, counts, methods, and exact values.

| Exclusive-time category | MNIST | Transformer |
|---|---:|---:|
| Conversion loop and attribute helper's own Python code | 36.94% | 38.23% |
| Warning filters and their direct helper calls | 36.47% | 37.50% |
| Type checks across callers, excluding warning-owned time | 13.86% | 14.37% |
| Attribute lookup/name filtering across callers, excluding warning-owned time | 6.56% | 6.96% |
| JSON encoding and JSON-compatibility probes | 0.61% | 0.67% |
| Everything else in the profiled thread | 5.57% | 2.27% |

The broader warning category includes `warnings.py` exclusive time plus direct
helper time assigned by pstats caller edges, such as list/filter operations. The
earlier 28.06%/28.89% figures include only functions defined in `warnings.py`.
Warning-to-warning edges are not added again. Other categories subtract the
warning-owned helper portion before assigning remaining exclusive time. Generic
type/attribute operations can have non-conversion callers, so these are an
explicit code-cost partition rather than exact semantic stage measurements.

Here, metadata conversion means constructing a snapshot dictionary from a live
object. A tensor snapshot records dtype, shape, gradient/device flags and object
identities, with no tensor elements. The implementation enumerates public
attributes with `dir`, reads them with `getattr`, and classifies the returned
values; many inspected methods are subsequently discarded. This is part of
serialization in the broader sense. The earlier ~0.4% row covers only encoding an
already constructed record with `serialize`, not this snapshot construction.
There is also an earlier `json.dumps` compatibility probe for each value; the
partition above includes its JSON-function time.

Warning management does not mean printing warnings. Every `safe_getattr` enters
`warnings.catch_warnings`, installs `simplefilter("ignore")`, reads one attribute,
and restores warning state. This is repeated even when no warning is raised.
Hoisting or bypassing this work needs the getter/warning/concurrency contract
checks described below.

## Hashing is an unmeasured path in this audit

Both profiles contain no calls to `dump_tensor`, `tensor_hash`, or `tensor_stats`.
API metadata conversion passes `include_tensor_data=False`, and model/proxy
tracking is disabled. The harness's initial/final model SHA-256 checks run outside
timed training for correctness. Hashing is also not universally one hash per API:
value/proxy tracking hashes eligible tensor-valued attributes according to its
tracking configuration. Therefore this audit does not estimate the cost
of tensor hashing, full tensor dumps, or statistics during training.

The frozen production code also has an important routing detail:
`dumper.dump_tensor` passes `with_cuda=False` to `tensor_hash` for both CPU and CUDA
tensors. The selected hash path converts floating values to int64, uses
`.cpu().numpy().astype(np.int64)`, and performs a column loop in Python/NumPy.
CUDA training alone consequently does not select the CUDA hash implementation;
on CUDA tensors this path also includes a device-to-host transfer. This is a
source finding, not a measured time allocation. MPS hashing has an explicit
unsupported-fast-path TODO in the dumper.

A hash-enabled follow-up must measure the actual tensor-tracking path separately:
number and bytes of hashed snapshots, input conversion, transfer/synchronization,
hash arithmetic, metadata construction, JSON encoding, and writer drain. CPU versus
CUDA hash choices must preserve hash results and mutation/snapshot behavior; no
backend or algorithm change was made here. The original metadata-only profile
cannot establish that hashing or full serialization is cheap.

## Correctness and evidence

Every run's complete loss trajectory, canonical initial/final model-state hash,
and full state-shape map exactly match the plain reference. Canonical state hashes
cover sorted parameter/buffer names, dtypes, shapes, and tensor bytes. The two
profiled runs also pass these comparisons.

All six downloaded API trace files were independently rehashed and parsed:
**5,574 records total**, with exactly one pre and one post for every recorded
call ID. Counts and byte sizes match the run manifests; there are no duplicate,
incomplete, or exception record groups. This does not prove universal API coverage.
Fresh-process raw traces contain different timestamps and identities; they were
not normalized, and byte equality between runs is not claimed. All 77 production
Python files and both harness files match their pre-run hashes after download;
each traced run's imported module digest matches the frozen source.

Source: `205a687eef30e498bd2d9a9951652b7cc05bec68`, tree
`2177e2fa826bb4623c3c9cdbb3bdcb61fafbedab`. Runtime: labmach, Intel Core i9-13900K,
32 logical CPUs, Python 3.11.4, PyTorch `2.16.0.dev20261012+cpu`, float32, one
intra-op/inter-op/native thread, `PYTHONHASHSEED=0`, initialization seed 42,
training seed 43, batch size 2. The shared host had no affinity or exclusive
reservation. The other owned inference experiment started only after this audit
finished. Sampled main-process RSS was approximately 329 MiB plain / 536–546 MiB
traced for MNIST and 321 MiB / 517 MiB for Transformer; this includes imports and
setup and does not isolate trace-buffer memory. Per-case limits were 90 seconds,
4 GiB sampled main-process RSS, and 256 MiB API trace; none was reached.

The existing adapter extracts exact benchmark model definitions without executing
their top-level scripts. MNIST uses the first six cached training images and
labels, NLL, and Adadelta. Transformer uses two 32-dimensional encoder layers,
64-dimensional feedforward, length-16 sequences, NLL, clipped manual SGD, and a
98-token WikiText prefix with a 60-token vocabulary. This is smaller than the
separate fresh-validation study's Transformer. Dataset, source, adapter, state,
and raw trace digests are in the [compact evidence](data/tracing-hotspot-audit-20261026.json).

The complete 80,977,920-byte evidence archive has SHA-256
`31ad9746dc9b5318b3f49d9119c86606dce48194ac92701695960bb537a84036`.
It retains exact source, adapters/controller, commands, per-step timings/losses,
raw traces, profiles, and logs. On labmach it is represented by the directory
`/home/yuxuan/experiments/codex-tracing-overhead-20261026-205a687`; the downloaded
archive and verifier are under the local project's
`outputs/release-20261026/tracing-audit/`. Raw traces are not committed.

To repeat on that host, create a new experiment directory, copy the retained
`audit_runner.py` and `workload.py` there, extract `source.tar` into its `source/`
subdirectory, and run:

```sh
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python \
  "$NEW_AUDIT/audit_runner.py" --root "$NEW_AUDIT"
```

The controller records exact child commands and refuses to overwrite individual
run directories. It reads the existing benchmark/data paths shown in the evidence.

## Proposed follow-up, not an implemented fix

First investigate batching warning-context setup and avoiding redundant metadata
classification. The measured hot loop is specific enough to justify a focused
prototype; there is no measured candidate speedup yet.

Hoisting warning suppression is not automatically semantics-preserving: custom
attribute getters may change warning filters, warn or raise, and concurrent
conversions can interact with process-global warning state. Preserve the current
per-getter behavior for unsupported/dynamic objects unless equivalence is proven.
Likewise, caching attribute names must account for instance-specific `__dir__`,
properties, newly added attributes, tensor mutation, and tensor subclasses.
Do not cache mutable values or move snapshots after the application call.

Any candidate needs exact conversion/trace contract comparisons, warning-state and
exception tests, concurrency checks, model/RNG parity, and new unprofiled paired
runs. No reduction in recorded APIs, fields, or snapshot timing is proposed.
