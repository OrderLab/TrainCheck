# Native tensor bindings without repeated warning contexts

The final implementation in signed commit `37f22e58954da403105c2c5510b7c7b6b3162ee6`
reduces ordinary traced training time on the measured CPU and H100 workloads.
It preserves native method binding and the existing metadata classifier, while
avoiding the warning-managed getter for verified native tensor methods. JSON
encoding, tensor hashing, instrumentation selection and the checker are unchanged.

## Fresh ordinary measurements

| Workload / timing scope | Baseline | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| CPU MNIST, complete three-step API-traced body | 286.400 ms | 222.665 ms | 22.25% |
| CPU small Transformer, complete three-step API-traced body | 682.324 ms | 527.587 ms | 22.68% |
| CPU, 2,000 complete metadata conversions | 1.607038 s | 1.190048 s | 25.95% |
| H100 Llama, median of steady-step run medians | 999.855 ms | 842.885 ms | 15.699% |

Percentages divide saved time by baseline time. CPU measurements have four fresh
processes per variant, ordered ABBA twice. Component conversion is separate from
training, starts after one warmup conversion per object and excludes tensor
hashing; its percentage is not a fraction of total training time. CPU training uses full API collection with model tracking disabled.

H100 job **55326071** uses the same 6,163,712-parameter TorchTitan Llama debug
model, native FSDP, batch one, sequence 256, and one H100. It collects selectively
using the same 722 frozen invariants and 244-API policy. There is no model tracker
or concurrent online checker. Each fresh process trains ten seed-42 steps; steady
statistics use synchronized steps 3–10. Every measured step remains in the evidence.

| H100 pair / order | Baseline median step | Candidate median step | Time reduction |
| --- | ---: | ---: | ---: |
| 1 / AB | 999.855 ms | 842.430 ms | 15.745% |
| 2 / BA | 1,000.320 ms | 842.885 ms | 15.738% |
| 3 / AB | 998.851 ms | 843.650 ms | 15.538% |

The **median paired reduction is 15.738% (1.1868x)**; the table's ratio of run
medians is a separate aggregate, 15.699%. Including cold steps 1–2, median complete
ten-step training-body time is **12.579847 to 10.990844 seconds, 12.631% less**.
Timings exclude
imports, setup, state evidence, outer loop housekeeping, queue drain and shutdown.
The separate seed-43, 20-step holdout has medians of 1,002.488 to 842.039 ms over
steps 3–20 (16.005% less); this is one validation pair, not a fourth primary pair.

The single H100 plain-control run has a 44.456 ms steady median: baseline/candidate
are still 22.49x/18.96x that time. CPU single plain controls give 8.60x/6.69x for
MNIST and 48.05x/37.15x for the tiny Transformer. Plain controls import TrainCheck
but install neither API wrappers nor tracked parameters. These ratios have only
one plain repetition, and the workloads are small; they are not general overhead
estimates. No multi-GPU scaling or concurrent-checker improvement is established.

## What changed and what remains

Ordinary metadata conversion visits hundreds of Tensor methods. Their values
usually contribute no output fields, yet generic lookup enters and exits a warning
context for each one. The shortcut verifies exact Tensor/Parameter type, ordinary
attribute lookup, native descriptor ownership, unchanged inheritance and an empty
native instance dictionary. It then calls the descriptor type's native binding
operation. Properties, custom lookup, subclasses, instance shadowing, foreign
descriptors and TrainCheck-specific fields retain the original guarded getter.

Only original MRO identities and live namespace proxies are cached. Current
namespaces and native dictionary storage are rechecked for each attribute, so
mutable values and descriptor results are not cached. A changed MRO disables the
shortcut, including after bases are restored. The original classifier handles
live primitive-type configuration exactly once. Native allocation failure retains
NOT_FOUND/backoff behavior; internal assertions remain visible.

The implementation deliberately retains the real bound object. A prior temporary
value and its type must be released at their ordinary points, while the bound
method must keep its receiver/function alive during a later property. Four
regressions demonstrated that the earlier skip-based candidate changed these
behaviors. All four fail on verbatim `287b3a2` sources on Python 3.10 and 3.12 and
pass with the final binding implementation. A fifth regression verifies that a
function's own `__get__` attribute cannot override Python's native descriptor
protocol. Historical CPU/H100 reports remain separately labeled; their faster
skip-based measurements are not reused as results for the final implementation.

This does not combine warning contexts around arbitrary properties, cache tensor
metadata, change serialization, or introduce asynchronous dumping. The CPU
component diagnostic reduces guarded getters from 1,192 to 60 per two conversions,
while retaining all 28 field names per object. It preserves warning-filter
contents/identity, the warning handler, tensor values and Python/NumPy/Torch RNG.
Those are operation counts, not percentages of runtime attributed to warnings.

A separate H100 diagnostic, job **55326074**, counted steps 3–10:

| Scope | Binding helper calls | Native bindings without guarded getter | Remaining guarded getters |
| --- | ---: | ---: | ---: |
| All threads | 1,940,048 | 1,698,000 (87.52%) | 242,048 |
| Window owner | 1,243,312 | 1,100,304 (88.50%) | 143,008 |
| Other trace-producing thread | 696,736 | 597,696 (85.79%) | 99,040 |

Exact Tensor receivers account for all native bindings: 1,698,000 of 1,788,000
calls (94.97%). Exact Parameter receivers are unobserved, not a measured zero
eligibility rate. The 152,048 other-receiver calls all use the fallback. Every
successful shortcut still constructs the real method and runs classification.
The count percentages do not attribute a fraction of total runtime to warnings.

The collector preserves returned objects and exceptions, verifies bindings before
installation and at restoration, and records zero exceptions escaping the binding
helper or guarded getter. Internal NOT_FOUND handling remains unchanged and is
not a separate exception counter. Its counted window takes 1.39489x its ordinary
control's duration; diagnostic
durations do not enter the performance table. Both corpora have 416,364 rows,
208,182 complete calls and 142,621,017 bytes with zero raw audit issues. Every
per-step/API/event-kind count and per-step byte total matches, as do the complete
recorded state and policy. This proves volume/integrity parity, not raw byte equality.

## State, traces and checking

All 16 CPU traced runs match fresh plain controls' recorded initial/final model,
gradient, optimizer and stored module state, RNG and losses. Every CPU MNIST run
has 520 API records / 351,073 bytes; every Transformer run has 1,338 records /
1,014,134 bytes. Every call has one PRE and one POST, with no VAR records. All
eight paired raw comparisons preserve complete ordered event sequences, fields,
arrays and duplicates. Remaining differences are identities and timestamps;
raw equality is false and cross-process identity-graph equivalence is not claimed.

All nine H100 phases pass independent source, environment, command, policy,
complete recorded state and loss checks. Seed-42 phases match the plain control;
the seed-43 candidate matches its selective baseline. GPU tensor evidence includes
names, shapes, dtypes and byte hashes, rather than archived full optimizer-state
checkpoints. No identity, tensor value or array was removed to establish state
parity.

Raw comparisons of the first seed-42 selective pair and the seed-43 holdout
uniquely match both complete ordered per-thread streams in each pair. Each variant
has 416,364 records / 142,621,017 bytes in the first selective pair and 715,004 /
244,304,737 in the holdout. These full field comparisons cover those two pairs;
raw/parser integrity audits separately cover all eight selective runs. No fields, array elements or
duplicates are removed. Differences remain in identities/timestamps, constructor
`object_id` and CUDA `cuda_stream`/`native_handle` values; raw equality is false.
Constructor snapshots store literal `id(receiver)` in both source archives. All
800/1,600 stream-metadata observations preserve alias equality and a bijective
observed handle correspondence. No structural/type/shape/array/exception payload
differences occur, including the same two step-one `_BypassDispatchCache` exceptions.
No global cross-process identity graph or inter-thread ordering equality is claimed.

All eight raw/parser audits and eight checker phases completed successfully for
this exact revision. The first seed-42 selective pair and seed-43 holdout each
execute the same 722 frozen rules. Every cohort has **719 active rules: 718 passes
and one violation; three rules are dormant**. There are no checker errors, unknown
outcomes, unobserved coverage or variable-check skips. Decision counts match across
variants: 114,373 per ten-step corpus and 218,713 per holdout corpus. No variable
precondition is requested, consistent with the API-only scope; this study does
not establish model-tracker or variable-checking coverage.

Within each corpus, all 722 complete `checker_result` values match structurally
between observed and independent direct checking. The comparator preserves scalar
types, fields, identities and array order; five legacy NaN literals match only at
the same paths. Cross-variant rule outcomes and decision counts match, while
complete checker results match for 721/722 rules. Rule 614 retains fresh-run
identity/timestamp differences in its failure evidence; those fields are not
removed or remapped to claim equality.

The shared violation is rule 614, `cuda.stream() always calls
StreamContext.__init__()`: its frozen child argument requires CUDA handle
385452624. Actual handles are 386050160/386062656 in the ten-step pair and
386059824/386070112 in the holdout. Both handle aliases agree, all other required
fields match, and no fields are missing or extra. Dormant APIContain rules 216/276
require reference worker ID 23438301021888; rule 717 requires unequal threads and
has zero decisions. Rules were not reinferred or filtered. This is unchanged
frozen-rule behavior, not evidence of zero violations or universal rule activation.

Offline validation ran on two separate labmach CPU cores, one native thread per
worker and an 8-GiB process-group cap per lane. Its scheduling and operational
timings are separate from inference-performance claims. Source and frozen input
inventories remained unchanged. All 98 downloaded report files were verified.

## Sources, environment and validation

- CPU baseline: `6dccb47d0a8bc192fbe7cbbea700b66106bc2381`.
- H100 baseline: composed `8cc73ab35590caa1a5d7286b16462d045b6d2da0`; candidate
  overlays only `dumper.py` and `metadata_methods.py` from signed `37f22e5`.
  CUDA hashing, selective policy and workload code are identical across variants.
  The production PR targets main and does not depend on the study's other changes.
- TorchTitan: `2210414902d84d1219fd63ab8efeb24375a2a1a1`.
- Frozen invariant SHA-256:
  `bad55c222f4eb3406bdfaec527ab1a75eaf66cdbe8f06c42d7f0775f4ee9224b`;
  compiled policy SHA-256:
  `df4c91296774b2dfd3af09d0a25e21e25f26ff1de93cb97ab959fa77379c3211`.
- Labmach CPU: Python 3.11.4, PyTorch `2.16.0.dev20261012+cpu`, NumPy 2.1.0;
  one Torch/native thread, CUDA hidden. MNIST is the existing 1,199,882-parameter
  Net with six cached normalized examples, NLL and Adadelta. The Transformer is
  21,052 parameters: two width-32 layers, feed-forward width 64, two heads,
  sequence 16, batch two, fixed WikiText prefix/vocabulary, NLL and clipped SGD.
  Seeds, data and benchmark-definition hashes are retained in every run manifest.
- Expanse: H100 80GB HBM3, Python 3.12.3, PyTorch
  `2.16.0.dev20261012+cu130`, CUDA 13.0. Eight CPUs / 48 GiB allocated; GPU work
  runs on compute nodes. Ordinary job completed in 4:36. Batch peak memory covers
  the entire job, not individual variants; no memory-cost improvement is claimed.
- Final local full suite: **335 passed, two failures** also reproduced on the
  measured main baseline (`test_traincheck_internal_attr_hidden` and
  `test_inject_eval_loop`). All 26 focused metadata tests pass on Python 3.10 and
  3.12; the exact composed source passes 33 metadata/classification tests on both.
  The matching labmach runtime also passes all 33.
- Pinned changed-file pre-commit, explicit test lint/format checks and strict
  scoped mypy pass. The legacy dumper retains its configured mypy coverage rather
  than claiming a repository-wide strictness migration. Earlier all-files checking
  found only the documented, unchanged Black whitespace failures in `config.py`
  and `proxy_wrapper/hash.py`; those unrelated edits remain excluded.

## Durable evidence and reproduction

Local artifact root: `outputs/metadata-fastpath-20261027/` in the canonical
checkout. CPU evidence is in `cpu-binding/`; new H100 packages and results are in
`gpu-binding/` and `gpu-binding-diagnostic/`. Final checking evidence is in
`validation-binding/job-55326071/validation-summary.json` and
`focused-attribution.json`; raw comparison reports are under ordinary job 55326071's
`trace-comparison/`. Exact commands, frozen source
inventories, raw traces, timings, diagnostic scripts and failure logs remain
outside Git. Historical studies are preserved separately. No credentials are
included in source or evidence packages.

| Artifact | SHA-256 |
| --- | --- |
| CPU evidence archive | `7418a17bd9bf05c36119ac56eb23abb28bd5280e552afafbe2a6df39931e8edd` |
| CPU candidate source archive | `61ff85dce4b0e94633e6fc932d6bc051744fe9c17d8625cf83a5d44c7aae5381` |
| H100 candidate source archive | `517c5afa213ec1693d42aa04f9afd61676dd1680f94b2f73734f0759dc3e68b5` |
| H100 ordinary results, job 55326071 | `3754fa9de95c586a4338ff49a5daec3ccac67a4d40689cfb07bf09c6e66a403b` |
| H100 diagnostic results, job 55326074 | `d6ddd8bb9852209d6f12f64e23f7f68a56d3e3d1f5535645b13b79644aa90df4` |
| Complete fresh validation reports | `4323fff37a0085d75de432645025918b3b262ec56b316f2bb19d002690288abd` |

From the artifact root, rerun saved analyzers:

```sh
python analyze_cpu_binding.py --root cpu-binding/results --archives cpu-binding \
  --output cpu-binding/verified-analysis.json
python analyze_binding_gpu_results.py \
  --ordinary gpu-binding/job-55326071/results/ordinary \
  --output gpu-binding/job-55326071/ordinary-analysis.json
```

For new experiments, use fresh output directories and the frozen packages' README
commands. CPU runs use `cpu-binding/run_study.py --root STUDY --candidate-commit
37f22e58954da403105c2c5510b7c7b6b3162ee6`; H100 submission verifies package checksums
before `sbatch comparison.sbatch`. Source inventories are verified before and after
execution. CPU processes have 90-second / 4-GiB / 256-MiB-trace bounds; H100 phases
and jobs have explicit time/trace/allocation limits. No limit was reached.
