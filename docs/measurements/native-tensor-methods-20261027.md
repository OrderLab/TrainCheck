# Historical CPU measurement: skipping tensor method metadata

**Superseded candidate:** these numbers measure `77ea19b`. Later lifetime tests
found that skipping a bound method changes references observable by subsequent
properties. The revised implementation retains real native binding and the
original classifier, bypassing only the guarded getter. Its fresh measurements
must be reported separately; the figures below are not claims about that revision.

TrainCheck's metadata converter reads hundreds of tensor methods and then discards
their bound callable values. Each read also enters and exits a warning-suppression
context. The candidate recognizes a narrowly verified ordinary method binding and
skips both operations. Recorded metadata fields, JSON encoding, tensor hashing,
and warning handling for properties remain unchanged.

## Measured results

These are ordinary, unprofiled CPU runs on labmach. They are not H100 results or
inference timings. Each primary workload ran three training steps in eight fresh
processes, ordered baseline/candidate/candidate/baseline twice. Four measurements
per revision contribute to each median.

| Workload and timing scope | Baseline median | Candidate median | Time reduction |
| --- | ---: | ---: | ---: |
| MNIST, API-traced training body | 276.949 ms | 217.152 ms | 21.59% |
| Small Transformer, API-traced training body | 676.312 ms | 516.164 ms | 23.68% |
| 2,000 complete tensor metadata conversions | 1.626168 s | 1.158417 s | 28.76% |

Reduction is `(baseline - candidate) / baseline`; training-body speedups are
1.275x and 1.310x. Training timings include `start_step` and the training body,
excluding imports, setup, metrics/state evidence, queue drainage and shutdown.
Component timings measure `convert_var_to_dict(..., include_tensor_data=False)`;
they exclude tensor hashing and are not a fraction of training time.

Raw three-step training-body times in milliseconds, listed by pair number:

| Pair | MNIST baseline | MNIST candidate | Transformer baseline | Transformer candidate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 277.677 | 217.590 | 679.095 | 515.303 |
| 2 | 276.435 | 214.336 | 675.916 | 519.917 |
| 3 | 276.054 | 216.714 | 675.490 | 517.025 |
| 4 | 277.462 | 217.629 | 676.709 | 509.738 |

An untimed component diagnostic counted **1,192 to 60 guarded attribute reads**
for two full conversions, a **94.97% reduction**. Both versions retained the same
28 field names per object. All eight timed component runs emitted 56,000 fields.
The separate diagnostic found no emitted warnings and preserved warning-filter
contents/identity, the warning handler, Python/NumPy/Torch RNG and tensor values.
This count does not establish what fraction of the H100 workload is eligible.

Single-pair secondary runs with **both API and subclass model tracking** enabled
measured MNIST 659.491 to 642.414 ms (2.59% reduction), and Transformer 758.467 to
620.978 ms (18.13%). Each retained 64 VAR records; API counts were 520 and 1,340.
These are preliminary single-pair results, not repeated performance estimates.

Fallbacks have a cost. A separate eight-repeat helper benchmark measured 84,900
eligible method reads at 78.556 to 48.702 ms, but 450 property reads at 0.400 to
0.690 ms. Property-heavy or subclass-heavy workloads may benefit less or regress.
The primary training results above include this fallback cost. There is no claim
that warning management has been eliminated, or that full instrumentation now
has low overhead. Against single plain controls, API-traced medians still cost
9.16x to 7.18x for MNIST and 152.14x to 116.11x for the tiny Transformer; these
ratios use one plain run per workload and are not repeated plain comparisons.

## Sources, workload and environment

- Baseline: `6dccb47d0a8bc192fbe7cbbea700b66106bc2381` (main).
- Measured candidate: signed `77ea19be77578fb68c654ea861f05a1a7c6267dc`.
- No production dependencies on the open CUDA-hashing, selective-policy,
  checking-workload or coverage PRs. The optional H100 experiment composes those
  existing study changes separately.
- Lab host: `yuxuan-Precision-3660`, Linux 6.1.0-1027-oem x86-64, Python 3.11.4,
  Torch `2.16.0.dev20261012+cpu`, NumPy 2.1.0. One Torch, interop, OpenMP, MKL and
  OpenBLAS thread. CUDA hidden; deterministic algorithms enabled; float32.
- Primary mode: API tracing, no model tracker, `use_full_instr=False`, every
  training step, no warmup, production tensor-hash configuration. No profiler
  or diagnostic wrappers during ordinary timings.
- MNIST: exact benchmark `Net` class, 1,199,882 parameters, first six cached
  training examples, batch two, original normalization, NLL/Adadelta(lr=1).
- Transformer: exact benchmark `PositionalEncoding`/`TransformerModel` classes,
  21,052 parameters, two layers, width 32, feed-forward width 64, two heads,
  sequence 16, batch two, 98-token WikiText prefix and 60-token vocabulary;
  NLL/backward/clipped manual SGD. This is a bounded adaptation, not the full
  RNN benchmark script or the 6.2M-parameter TorchTitan Llama experiment.
- Construction/data seed 42; training RNG reset to 43 after setup. Both variants
  use the unchanged `tools/fresh_validation/benchmark_workload.py` adapter.
  Every run records exact model-definition and dataset SHA-256 values.
- Plain controls import TrainCheck and its global Module setter hook but install
  neither API wrappers nor tracked parameters. They are state/loss controls,
  not an independently installed framework-only benchmark.

The benchmark-directory Git HEAD was unavailable, so manifests identify the
actual source files and extracted class definitions by content hashes. Source
snapshots were hashed before and after every process and remained unchanged.
The runs have 90-second, 4 GiB observed-RSS and 256 MiB trace caps. Overall memory
overhead was not isolated; no memory improvement is claimed.

## Behavior and compatibility

All 16 primary runs exactly match the recorded plain-control initial/final model
state, gradients, optimizer state (where present), stored module values including
the Transformer's cached mask, Python/NumPy/Torch RNG states, and loss trajectory.
Tensor evidence includes names, dtypes, shapes and SHA-256 of bytes; raw checkpoint
tensors were not archived. Both secondary combined pairs also match their controls.

Every primary pair has the same API sequence/count map and complete PRE/POST
groups: 520 MNIST and 1,338 Transformer records. Recursive raw comparisons retain
every field and report differences only in timestamps, process/thread/call/object
IDs and `_TRAINCHECK_*_ID` values. Cross-process raw traces are **not byte-equal**;
IDs were not deleted or remapped to claim equality. These results do not prove
cross-run object-identity equivalence or frozen-invariant checker parity.
Same-object regression tests separately exercise recorded identities and order.
Raw API trace sizes also match in all eight runs per workload: 351,073 bytes for
MNIST and 1,014,134 bytes for the Transformer.

The fast path accepts exact Tensor/Parameter instances only, with empty native
dictionaries and ordinary lookup. It checks live class namespaces, descriptor
ownership, inheritance identity, and retained-type configuration at each decision.
Only original MRO identities and live namespace proxies are cached; object values
and mutable lookup results are not. Changed inheritance disables the shortcut,
including after bases are restored. Properties, subclasses, shadowed attributes,
custom lookup/configuration and `_TRAINCHECK` fields use the existing path. A
getter may mutate later attributes or warning filters without bypassing restoration.

Validation:

- 17 new contract tests; 106 focused tests pass on Python 3.10.16 / Torch 2.2.2.
- 24 metadata tests pass on the matching lab Python/Torch runtime.
- Full local suite: 326 passed, two failures independently reproduced on unchanged
  main: `test_traincheck_internal_attr_hidden` and `test_inject_eval_loop`.
- Changed-file pinned pre-commit passes. All four new Python files pass stricter
  scoped mypy with untyped-body, untyped-definition and generic checks.
- All-files Ruff/mypy/isort pass. All-files Black finds existing trailing
  whitespace in `config/config.py` and `instrumentor/proxy_wrapper/hash.py` on
  both revisions; hook-created unrelated edits were restored and excluded.

## Evidence and reproduction

Durable local evidence: `outputs/metadata-fastpath-20261027/` in the canonical
checkout. Lab evidence: `/home/yuxuan/experiments/metadata-fastpath-20261027/`.
Large traces and machine-specific experiment scripts are intentionally outside
Git. `PROTOCOL.md`, `training_benchmark.py`, `run_comparison.py`, and
`compare_training.py` record the exact bounded training recipe. The downloaded
`training-evidence/` contains every run's manifests, state evidence, timings,
raw traces and strict comparison reports; `component/` contains all component
timings and separate diagnostics.

| Artifact | SHA-256 |
| --- | --- |
| Baseline source archive | `7aae9422ca4b026c6f4c5056e53c198cb4e2b1acc323fbbdd1a651efdd27e09e` |
| Candidate source archive | `c2e394d2043da3a5a305312f0147c236b2311a99e071bcf20343db63ba8ab4d9` |
| Final training worker | `8a8b351a8baf0a98215461855cc72bfded7dbc94a667f27af789170a2deedda6` |
| Downloaded training evidence archive | `d406868f9c24d598f5d91350f0eb3360ed2cb3cf9713fb20b0e7f0e6a7898381` |

The committed component benchmark can run against either frozen source tree:

```sh
PYTHONPATH="$baseline_source" PYTHONDONTWRITEBYTECODE=1 "$python" \
  "$candidate_source/tools/benchmarks/tensor_metadata.py" \
  --iterations 1000 --output baseline.json
PYTHONPATH="$candidate_source" PYTHONDONTWRITEBYTECODE=1 "$python" \
  "$candidate_source/tools/benchmarks/tensor_metadata.py" \
  --iterations 1000 --output candidate.json
```

Check the emitted `dumper_path` and SHA-256. Repeat in ABBA order with fresh output
paths. The initial local pilot without explicit `PYTHONPATH` loaded an editable
installation from another checkout; it is retained as `local-component-wrong-import.json`
and excluded from every reported comparison.

For the full CPU recipe on labmach, use the saved harness in a fresh study copy
with frozen sources and the same prepared plain controls:

```sh
python=/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python
"$python" "$study/run_comparison.py" --root "$study" \
  --candidate-commit 77ea19be77578fb68c654ea861f05a1a7c6267dc
```

The subsequent [H100 selective-Llama comparison](native-tensor-methods-h100-20261027.md)
measures three ordinary AB/BA pairs and a holdout independently of these CPU runs.
It also records a Python 3.12 import compatibility fix and separate eligibility
counts. The CPU figures above retain their original measured revision and scope;
they are not extrapolated to the H100 workload.
