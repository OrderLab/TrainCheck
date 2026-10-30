# Instrumenter hot-path optimization, 2026-10-28

This study compares signed implementation
`161313a93512b07410fa2b1072613333a5a2e492` with
`bccc56fab83d525198286f1855b9dd7f6866135f` (PR #162). PR #163 depends on
#162. The work targets metadata collection; it changes neither the trace format
nor the point at which mutable values are serialized into an owned snapshot.

## Implementation and contracts

1. Native Tensor/Parameter metadata lookup caches two lookup-hook facts for a
   contiguous immutable suffix of the class MRO. CPython's native `type.__flags__`
   descriptor supplies
   `Py_TPFLAGS_IMMUTABLETYPE`; class names or package names never establish
   immutability. All descriptors, mutable class namespaces and receiver
   dictionaries remain live,
   and every binding still checks the current MRO, lookup hooks and descriptor
   owner. Real method bindings and the existing classifier preserve observable
   temporary lifetimes. Only booleans and existing instrumenter-owned namespace
   views are retained; no descriptor or attribute-name references are added.
   Other runtimes or an absent capability retain live
   lookup; malformed present metadata is an internal error.
2. Exact None, bool, float, str and signed-64-bit integers avoid a throwaway
   `json.dumps` compatibility probe. The existing compile/FakeTensor gate still
   runs first. Subclasses, containers and other objects retain their original
   path. Larger integers keep the probe, including Python's configurable
   integer-to-string limit and its synchronous errors. Returned objects and
   runtime type labels are unchanged.

No metadata values or mutable class decisions are cached. No tensor fields,
identity fields, warnings from application getters, trace records or checks are
dropped. Final `orjson`/stdlib JSON fallback, argument observation order,
serialization timing, producer buffering and writer behavior are unchanged.

## Ordinary labmach measurements

| Timed body | Baseline median | Candidate median | Less time |
|---|---:|---:|---:|
| 800 add/mm calls | 1.516760 s | 1.479685 s | 2.44% |
| MNIST, 10 training steps | 0.667927 s | 0.646459 s | 3.21% |
| Transformer, 10 training steps | 1.771281 s | 1.725027 s | 2.61% |

Training uses the existing benchmark adapters, batch size 2, Transformer
sequence length 16, and ten matched data/training steps. These are complete
short training loops, not full dataset epochs. Traced training bodies include
forward, loss, backward and optimizer work; setup, evidence capture, final
queue wait and shutdown remain outside the body. MNIST trial ranges overlap;
every Transformer candidate trial is faster than every baseline in this set.
All before/after numbers in this table use the corrected revision.

MNIST has 1,199,882 parameters and uses the first 20 cached training examples,
no shuffling, NLL loss and Adadelta. The small Transformer has **28,267
parameters**, two layers, width 32, two attention heads and feed-forward width
64. Its WikiText-2 prefix has 322 tokens and a 171-token vocabulary. The prefix
SHA256 is `c2122261adb584adc0e9cd4672758896de1260e7dae0ed732e8f07394bce9712`;
the underlying text SHA256 is
`9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`.
This differs from the earlier three-step study's smaller vocabulary/model;
the fresh baseline and candidate both use the same ten-step data and model.
The training harness SHA256 is
`8a8b351a8baf0a98215461855cc72bfded7dbc94a667f27af789170a2deedda6`,
and the frozen adapter SHA256 is
`5b6852940601d663c9aedad3e096c869fea1631d2038d21956e883ae6e4c7a31`.

The corrected candidate reduces the median timed operation loop from
**1.516760 to 1.479685 seconds, 2.44% less time (1.025x)**. There are four fresh
processes per revision in A1/B1/B2/A2/A3/B3/B4/A4 order. Baseline samples are
1.521204, 1.518426, 1.515093 and 1.513667 seconds; candidate samples are
1.544820, 1.481346, 1.475127 and 1.478023 seconds. One candidate run is slower
than baseline, so the result must not be described as an improvement in every
trial. Percentages divide saved time by baseline median time.

Each trial makes 800 calls: 200 repetitions of `torch.add` and `torch.mm` with
exact Tensor/Parameter arguments, using 16x16 float32 inputs. All 800 outputs
remain retained during the loop in every variant. Full API argument and return
metadata produce 1,600 records per trial. Model tracking is disabled. Timing
includes the identical output-list retention and excludes setup, state evidence,
final queue wait and process shutdown.

Every traced run matches the fresh plain control's complete recorded input,
output and Python/NumPy/Torch RNG evidence. Four complete ordered raw-record
comparisons retain every field and differ only in original process-specific
identities and timestamps; raw byte equality is not claimed. Source inventories
are verified before and after the study. Recorded final queue-wait time is not
total I/O cost: writing also overlaps the operation loop and evidence capture.

The unchanged lab runtime is Python 3.11.4, Torch 2.16.0.dev20261012+cpu,
NumPy 2.1.0 and orjson 3.13.0. Runs use one pinned performance core (CPU8) and
one native/Torch thread. There is no GPU, selective-instrumentation policy,
online checker or distributed execution in this comparison.
The single plain control establishes state/output parity, not a stable
uninstrumented timing baseline; no tracing-overhead multiplier is inferred.

Across all 24 instrumented trials, complete recorded state/loss/RNG comparisons
against the three fresh plain controls pass. Twelve ordered raw comparisons
retain all IDs/timestamps and find no other field, shape, type, array or exception
differences. Per trial, MNIST emits 1,584 records/1,066,058 bytes; Transformer
4,446 records/3,370,348 bytes; add/mm 1,600 records/1,863,600 bytes. Each has zero
VAR records because model tracking is disabled in the timed series.

| Workload | Observed peak RSS median, baseline to candidate | Final queue-wait median, baseline to candidate |
|---|---:|---:|
| Add/mm (process group) | 508.160 to 507.867 MiB | 0.023853 to 0.327049 ms |
| MNIST (training process) | 542.705 to 548.924 MiB | 0.208640 to 0.193022 ms |
| Transformer (training process) | 518.225 to 518.201 MiB | 0.283403 to 0.248324 ms |

These are operational observations, not isolated memory costs attributable to
the optimization. In particular MNIST's observed RSS is higher; no process-memory
saving is claimed. All min/median/max values and 24 individual samples are in
`measurements-summary.json`. Queue wait occurs after state evidence capture,
when background writes have already progressed; it is not the total I/O cost.

## Separate model-tracker correctness smoke

Six fresh three-step runs cover a plain control and baseline/candidate combined
API + subclass model tracking for each model. All four combined cases match
their plain control's complete recorded initial/final model, optimizer, gradient,
stored-value, loss and RNG evidence. Parameter identity and optimizer references
are checked within each run. Complete ordered API/VAR comparisons retain all
fields and find only original ID/timestamp differences.

Each MNIST combined run emits 520 API and 64 VAR records. Each Transformer run
emits 1,340 API and 64 VAR records. The three-step Transformer smoke uses 21,052
parameters and its shorter data prefix, distinct from the 28,267-parameter
ten-step timed workload. No performance result is inferred from these smoke runs.

Observer coverage is explicitly limited: MNIST observes all eight parameters
before/after each step, with eight updates at steps 2 and 3. Transformer observes
only `model.input_emb.weight` before/after each step; all 29 parameter names
appear in update records at steps 2 and 3. Coverage matches the baseline but is
not complete Transformer pre/post coverage. No invariant checker ran in this
study, and trace integrity is not a claim of invariant activation or success.

## Component measurements

These local CPU measurements are separate from training. Percentages divide
saved elapsed time by the baseline elapsed time in the same row.

| Component | Baseline | Candidate | Less time |
|---|---:|---:|---:|
| 120,000 exact scalar conversions, first series | 166.130 ms | 107.213 ms | 35.46% |
| Same scalar workload, second series | 167.685 ms | 105.716 ms | 36.96% |

Scalar measurements use Python 3.10.16/Torch 2.2.2 and six fixed values repeated
20,000 times, four runs per
variant in A1/B1/B2/A2/A3/B3/B4/A4 order. The baseline function comes from the
frozen baseline source; the candidate is the actual scalar implementation in
`f9b221f`. Its entire dumper module is unchanged in corrected `161313a`, SHA256
`60e626d77dcbcbdc98317dd061a0bc09cde638b3d0bf70a96a3da510f46fc7de`.
These component measurements do not validate the superseded descriptor cache.
The two repetitions are labeled by the accompanying replay trace files, but the
scalar workload is the same synthetic workload in both; they are not MNIST or
Transformer training timings. No profiling hooks run inside these measurements.

An earlier descriptor-map cache passed the ordinary workload comparisons but
failed a new regression: a custom getter observed additional descriptor/name
references through `sys.getrefcount`. That implementation and its performance
results remain historical evidence. The corrected cache records only lookup-hook
facts and existing namespace views. The regression fails against the earlier
implementation and passes after correction.

The corrected cache's fixed containers occupy **344 bytes per process** in the
local runtime, measured with `sys.getsizeof`. This excludes the NamedTuple
class/code objects and allocator overhead; it is not peak RSS. The helper's
SHA256 is `374383debe3dcdb76d13e6b901c25c9441021a4c1cf5df8f1d9bb5f8f520b3d9`.

Two actual-source local native-metadata batches each have nine alternating-order
repeats. Full Tensor conversion is 893.6 to 887.9 us (0.64% less) in the first
batch and 869.6 to 836.1 us (3.85% less) in the second. The first batch has clear
scheduling outliers; one bounded repeat was performed and both complete batches
are retained. This does not support one stable general converter percentage.
The ordinary pinned lab comparisons provide the primary pipeline measurements.

## Binary capture and other rejected experiments

Replay uses one saved 520-record MNIST trace, reconstructing positional integer
keys while retaining every decoded field. The trace SHA256 is
`5f941cfb23dee11054af901056fbbac489f440ba32ce005f859c8476323dac59`.
These are exclusively owned decoded fixtures, not arbitrary live objects.
Forty passes process 20,800 records; each comparison uses four runs per variant.

| Replay experiment | Current serializer | Experiment | Result |
|---|---:|---:|---|
| Pickle capture alone, without validation | 58.912 ms | 31.892 ms | 45.86% less producer work in this optimistic fixture |
| Python validation followed by pickle | 58.773 ms | 124.984 ms | 112.65% more producer work |
| Pickle capture, decode, then existing JSON serialization | 58.773 ms | 143.648 ms | 144.41% more total sequential work |

Pickle payloads occupy 286,757 bytes versus 350,553 UTF-8 JSON bytes, excluding
framing. Exact final JSON strings match after round-trip. This does **not** prove
a correct live-object snapshot: custom pickle hooks can run application code,
and Python validation followed by pickling leaves a concurrent mutation gap.
IPC, scheduling, copies, backpressure, worker memory and drain costs are excluded.
Parallel execution might hide some worker work, but was not measured here.
The binary approach remains exploratory, with no production change or training
speedup claim.

The same reconstructed MNIST corpus sends 219/520 records (42.12%) through
stdlib fallback because positional argument keys are integers. A callback-free
prefix guard before bypassing the failed `orjson` attempt was **128.56% slower**
than the existing serializer. Simply enabling non-string keys changes codec
semantics, including nonfinite values, so it was not adopted. UTF-8 byte batching
showed only 0.09% difference in that replay and was not adopted. Direct native
slot dispatch and cached mappingproxy getter methods also failed to provide a
meaningful gain. Preserved prototypes and raw results distinguish these rejected
options from the final implementation.

## Validation

- Python 3.10: 70 focused metadata/scalar/trace-integrity regressions pass.
- Python 3.12: 43 metadata tests and 12 scalar tests pass.
- Full Python 3.10 suite: 357 pass, with two failures reproduced in an isolated
  unchanged baseline archive: `test_traincheck_internal_attr_hidden` and
  `test_inject_eval_loop`. This is not a claim that the full suite is green.
- Configured changed-file pre-commit hooks pass. New tests also pass explicit
  Ruff/Black/isort checks. Strict scoped mypy checks the metadata helper and both
  new test modules; ordinary repository mypy checks the changed dumper and its
  callers. Existing untyped metadata delegates remain outside strict coverage.
- Independent cross-review found no actionable defect. Signed-commit verification
  succeeded with the configured signing key.
- All 329 timed-evidence files and 89 combined-evidence files passed checksum
  verification after download. All study processes have finished.

## Evidence and reproduction

Durable raw evidence lives under
`outputs/instrumenter-optimization-20261028/` in the canonical TrainCheck
checkout, outside Git. Relevant files include `CHECKPOINT.md`,
`native-lookup-lifetime-repair/REPORT.md`, its two actual-source timing datasets,
`native-lookup-lifetime-repair/final-cache-size.json`, `mnist-final-component-probe.json`,
`transformer-final-component-probe.json`, `binary-capture-probe-checked.json`,
and the test logs. Scripts, full timing repetitions, source hashes and limitations
are retained beside those results. Exploratory older results are preserved and
are not substituted for the final-candidate numbers above.

The isolated labmach harness is under `labmach/`; its README documents the frozen
source inventories, existing runtime/data, exact invocations, resource bounds
and separate `--stage micro` / `--stage training` gates. The candidate study is
`/home/yuxuan/experiments/instrumenter-optimization-20261028/161313a93512`.
Its preparation archive SHA256 is
`e6d8929a4bff0e1d5101f38cbb2c7f4ba8583241f0f8c31b4f1a6fefc40f93d3`.
Local final study artifacts are under `labmach/studies/161313a93512/`.

| Artifact | SHA256 |
|---|---|
| `timed-evidence.tar.gz` (23,645,408 bytes) | `1fe4188c382816424e21496327a018a0610caaa8eeaa93d91ecd9f59eaf5471b` |
| `combined-evidence.tar.gz` (3,850,102 bytes) | `e356b9393168c00758f20c7382c57fe877d0798d7851d7e56dd0018f6b396747` |
| `verified-analysis.json` | `1320d25c89f540c8a041d756b928ab3f4d649f4b5789b76f5bb02f31b26f62c2` |
| `measurements-summary.json` | `48302b7201c607ffbbc0b8c56b12fab550816b85ce4ebe7e4c3106bafbd2e867` |
| `verified-combined-analysis.json` | `4a0614b078383279aa286eaf0efa988edff18027167cbecae223f36a123f4dba` |

No H100 experiment is part of this development pass.
