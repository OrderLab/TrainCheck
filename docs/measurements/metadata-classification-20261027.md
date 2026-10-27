# Final matched metadata-classification measurements

With the reviewed correctness changes on both sides, the metadata fast path
reduces median three-step CPU training time by **9.96% on MNIST** and **18.43%
on the small Transformer**. Both API and subclass model tracing are enabled
under ordinary collection (`use_full_instr=False`, retaining normal exclusions).
Every one of the ten matched pairs improves. Hashing is unchanged.

These final numbers supersede the initial study for the current implementation.
The initial evidence remains in the parent output directory and in repository
checkpoint `bcf9bb82c3c11812336d1199dc699aa4ee6473e2`. Do not compare timings
across the studies to attribute additional improvements.

## Exact sources and experiment

| Variant | Commit | Tree |
|---|---|---|
| Baseline: all current correctness/default changes, metadata fast path removed | `7a40649d2ce55c1a92380e26cd0f74b61c05824b` | `c84687d977b748a5080f034ace78c78b45cec172` |
| Candidate: all current changes including metadata fast path | `035f7f1976c5d84bd01f5d96bb095fb07c9cc059` | `89fdd92ebc6a57e5878d27cfb1b18dc69ee28a0b` |

All **78 production Python files per variant** match the respective exact Git
archive byte for byte. The only production difference is
`traincheck/instrumentor/dumper.py`. Both sides include the parameter identity,
alias/ownership lifecycle, default-tracker and constructor-snapshot fixes.

The optimization recognizes exact built-in callable types before repeatedly
checking them against framework tensor/module classes. Attribute enumeration,
getter evaluation, warning scope/restoration, supported metadata fields and the
generic fallback remain present. No hash algorithm, backend, dump format,
sampling or observation-depth change is part of the comparison.

Conditions: labmach Intel Core i9-13900K; Python 3.11.4; PyTorch
`2.16.0.dev20261012+cpu`; one PyTorch/native thread; CPU only; float32; batch 2;
three **cold** steps. Initialization seeds are 42, training seeds 43 for Python,
NumPy and PyTorch. No affinity or exclusive host reservation was used.

- MNIST uses the exact benchmark `Net` (1,199,882 parameters), the first six
  cached MNIST training images, NLL loss and Adadelta with learning rate 1.
- Transformer uses exact benchmark model definitions with bounded dimensions:
  embedding 32, two layers, two heads, feedforward 64, sequence length 16 and
  vocabulary 60 from the WikiText-2 prefix. Gradient clipping and the original
  manual SGD updates remain present.
- Model registration occurs before optimizer construction. Both-tracer runs use
  explicit subclass tracking, depth-3 static observers, argument scanning,
  hash-format tensor dumps on CPU, sampling interval 1, warmup 0, and disabled
  API descriptors. This does not measure `--use-full-instr`, Proxy/VarSampler
  modes, CUDA, MPS or distributed training.

The timer includes `start_step`, the training body and disabling instrumentation
on exit. It excludes imports/setup, loss extraction, model/RNG verification and
final writer shutdown. RNG capture runs **after the timer with wrappers disabled**
and adds no API trace records. Every case uses a fresh process and has a 90-second,
4-GiB sampled main-process RSS, 256-MiB trace limit; no case hit a limit.

## Ordinary measurements

Five pairs per workload alternate baseline/candidate then candidate/baseline,
ending baseline/candidate. Profiles are separate from these measurements.
Percent reduction uses the baseline median **training-body time** as denominator,
not added overhead or whole-process time.

| Workload | Baseline median (range), ms | Candidate median (range), ms | Reduction | Ratio of medians |
|---|---:|---:|---:|---:|
| MNIST | 667.776 (653.230–675.525) | 601.247 (577.793–626.689) | **66.529 ms / 9.96%** | 1.111× |
| Transformer | 959.440 (945.047–1,022.942) | 782.596 (765.700–810.611) | **176.844 ms / 18.43%** | 1.226× |

| Workload | Baseline runs, ms | Candidate runs, ms | Paired reductions, ms |
|---|---|---|---|
| MNIST | 674.327, 667.776, 653.230, 666.759, 675.525 | 626.689, 577.793, 587.515, 601.247, 626.016 | 47.637, 89.983, 65.715, 65.512, 49.509 |
| Transformer | 945.047, 967.448, 951.148, 959.440, 1,022.942 | 797.606, 768.657, 782.596, 810.611, 765.700 | 147.440, 198.792, 168.552, 148.829, 257.243 |

Paired reductions range from 7.06%–13.48% for MNIST and 15.51%–25.15% for
Transformer. No outliers were discarded. Five short pairs on a shared machine
do not establish a confidence interval, steady-state scaling or GPU performance.

## Profiles and remaining work

One separate profile per variant/workload has total time **1.372028 → 1.122652 s**
for MNIST and **2.719891 → 2.135597 s** for Transformer. Every percentage below
uses that variant's **profiled total**, not its ordinary median. Cumulative rows
overlap; they cannot be added into an exclusive cost partition.

| Cumulative component | MNIST baseline → candidate, ms (% of profile total) | Transformer baseline → candidate, ms (% of profile total) |
|---|---|---|
| Metadata conversion, `convert_var_to_dict` | 1,307.722 (95.31%) → 1,061.525 (94.56%) | 2,641.869 (97.13%) → 2,056.602 (96.30%) |
| Built-in `isinstance` | 169.932 (12.39%) → 20.078 (1.79%) | 415.405 (15.27%) → 53.038 (2.48%) |
| Attribute lookup, dumper `safe_getattr` | 624.366 (45.51%) → 598.322 (53.30%) | 1,513.444 (55.64%) → 1,472.196 (68.94%) |
| Warning `simplefilter`, nested in attribute lookup | 228.109 (16.63%) → 219.396 (19.54%) | 553.408 (20.35%) → 540.450 (25.31%) |
| API argument conversion | 599.447 (43.69%) → 461.566 (41.11%) | 1,513.930 (55.66%) → 1,184.774 (55.48%) |
| API return conversion | 293.872 (21.42%) → 223.348 (19.90%) | 950.236 (34.94%) → 742.331 (34.76%) |
| `ProxyParameter.dump_trace` | 423.777 (30.89%) → 387.950 (34.56%) | 208.495 (7.67%) → 161.196 (7.55%) |
| `tensor_hash`, nested in snapshots | 221.088 (16.11%) → 230.689 (20.55%) | 5.592 (0.21%) → 5.207 (0.24%) |
| Final JSON `serialize` | 4.231 (0.31%) → 5.508 (0.49%) | 10.609 (0.39%) → 10.576 (0.50%) |

`isinstance` calls fall from **1,503,819 to 464,813** for MNIST and from
**3,622,987 to 1,154,741** for Transformer. Parameter `__instancecheck__` calls
fall from 203,335 to 2,817 and 495,303 to 8,937 respectively; Transformer also
has one separate instance-check entry in each profile.

Attribute lookups and warning scopes stay at **211,616 for MNIST** and **516,009
for Transformer**. Their repeated setup remains a concrete next instrumentation
bottleneck, subject to preserving getter side effects and warning restoration.
Final JSON encoding is small in these profiles. Hashing was not optimized:
both variants use the existing CPU implementation, with unchanged call counts.
The small profile hash-time differences are not a hash-speedup claim.
Metadata conversion and JSON encoding still run synchronously in the producer;
the background threads handle buffered file writing. The small JSON row does
not imply that all preparation of a trace record is cheap or asynchronous.

## Correctness, coverage and resources

All **30 cases** passed: six candidate correctness cases (Transformer plain,
API, VAR and combined; MNIST plain and combined), twenty ordinary paired cases
and four profiles. The six correctness cases complete before measurement starts.
Every paired/profiled run is checked directly against the corresponding plain
candidate run: exact initial/final model-state digests and shapes, every loss
and step ID, and Python/NumPy/PyTorch RNG state before training and after each
step. Only elapsed seconds are excluded from loss-row comparison.

All retained raw trace checksums, sizes, counts and API pre/post multiplicities
were reread locally. Every tracked parameter has changing data hashes across
the three steps. Counts and coverage signatures agree across ordinary/profiled
baseline and candidate runs:

| Workload | API records / complete pairs | VAR records / parameters | Tensor data+gradient hash fields | Raw API+VAR bytes |
|---|---:|---:|---:|---:|
| MNIST | 520 / 260 | 64 / 8 | 112 | 419,429 |
| Transformer | 1,340 / 670 | 64 / 29 | 64 | 1,088,456 |

Profile calls to both `tensor_hash` and `hash_tensor_cpu` exactly match 112 and
64 recorded hash fields. MNIST has 48 gradient hashes in addition to its 64 data
hashes. Transformer snapshots have `grad=None` at the existing observation points:
they capture changing parameter values, but **do not establish gradient snapshots
or pre/post observations for every manual SGD update**. We did not replace manual
SGD or add synthetic observations to improve those counts. Final model/loss/RNG
checks separately establish behavior parity for the completed training steps.

Logical hashed input volume, counting repeated snapshots, is 67,193,392 bytes for
MNIST and 214,496 bytes for Transformer; this is not physical memory traffic or
a time share. Raw identities and timestamps remain intact. There is no claim of
normalized raw trace equality across independent processes.

| Ordinary median | MNIST baseline → candidate | Transformer baseline → candidate |
|---|---:|---:|
| Setup | 2.576 → 2.636 s | 2.551 → 2.528 s |
| Whole process | 6.816 → 6.915 s | 6.719 → 6.816 s |
| Post-metrics exit | 2.272 → 2.234 s | 1.900 → 2.156 s |
| Sampled main-process peak RSS | 556.816 → 561.523 MiB | 515.715 → 515.727 MiB |

Post-metrics exit includes background-writer drain, Python shutdown and polling
delay. It is not a pure I/O timer. These short jobs do **not** show a whole-process
speedup; startup/shutdown and host variation outweigh the training-body savings.
Full per-run distributions are retained in [`data/metadata-classification-20261027.json`](data/metadata-classification-20261027.json).

## Artifacts and reproduction

Local directory:
`/Users/matt/gitrepos/traincheck/outputs/instrumenter-performance-20261027/final-matched`.
Remote directory:
`/home/yuxuan/experiments/codex-instrumenter-performance-20261027-final-matched`.

The source archives are exact Git exports. Their isolated workload adapters add
the same typed model-registration callback before optimizer construction; no
production source is patched in place. Data and original benchmark definitions
come from the existing read-only caches recorded in each manifest.

With a fresh copy of the retained experiment structure (run directories must not
already exist):

```bash
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python correctness_driver.py --root "$EXPERIMENT_ROOT/candidate"
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python run_experiment.py --root "$EXPERIMENT_ROOT" --phase paired
python analyze_experiment.py --root "$LOCAL_EVIDENCE_ROOT"
```

The runner records every child command and sets the isolated source path, native
thread limits, CPU visibility and deterministic seed configuration. It aborts on
execution, coverage, model/loss/RNG or resource-limit failure. The analyzer verifies
source archives, executed harness hashes, raw records, plain references, unchanged
production-file scope and profile hash activation. No installations or shared
source/environment edits were used.

The benchmark copy has no available Git-head identity. Exact source/definition
and data hashes are therefore retained in each manifest and [`data/metadata-classification-20261027.json`](data/metadata-classification-20261027.json):

| Input/environment artifact | SHA-256 |
|---|---|
| MNIST benchmark source | `4478c5de92610a1e03d61c356801d51ffca749f99dac00f234d6b6ea2bcb2aa7` |
| Transformer benchmark source | `e4451a33e9b16953004bc639d736d14f42a2b21f2939ccabe4cff10d3c40b74c` |
| MNIST image IDX file (first six samples used) | `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db` |
| MNIST label IDX file | `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5` |
| WikiText-2 training file | `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f` |
| Used 98-token prefix | `9b0b504c73ee103a9128f2072536e99b2696039399bd57669477f030db10eb35` |
| Runtime package inventory | `f9b751a6b4b74230992a297eec30164e8d78fcd6a2d177726075f381bfae3c37` |
| CPU inventory | `ab09fc242f18a43f3e25a1bb6d8aaf6b2940f71efa4c2800c82f825f13bdaaeb` |

| Artifact | SHA-256 |
|---|---|
| Baseline source archive | `efc665f1b6914964f3ac93ac76a3e00791b5cf07c310818c65e7711496260537` |
| Candidate source archive | `793087ee08bbfb10ed60712d7d22da2153809236c26eb113d78592695754fe4e` |
| Raw evidence archive, 25,722,591 bytes | `875be9098fab644afed9798cd6005f66af6567719e6be1765d84e9bd74725584` |
| Executed runner | `21984b309b688e95de62d37b59e6e5f7c9a1685ef5293e54dfaacd60e9305875` |
| Executed paired driver | `5a0d4c4a02ea397bce646a482c2657bb80a1cd28bb5fa6e7cd0127071c6fda67` |
| Executed correctness driver | `752968c1c2bd4b646f2c30dc83788597ff5567d41b3759bebc2b9c658795d6fe` |

Configured Ruff/Black/isort checks pass for the four harness/analysis modules.
Scoped mypy passes with `--check-untyped-defs --disallow-untyped-defs
--disallow-any-generics --follow-imports=skip --ignore-missing-imports`; dependency
typing is not fully checked by that invocation. The exact composed candidate also passes pinned production hooks, configured
mypy on all 78 production files, and strict installed-Torch checking of eight
new contract/test modules. The complete suite passes 246/247 tests; the sole
`test_inject_eval_loop` failure also occurs on unchanged main. Separate real CLI
runs complete ordinary, selective and full-flag collection, with exact training
parity and two active passing invariant decisions each. Detailed correctness
evidence is retained in `outputs/instrumenter-final-validation-20261027` and
`outputs/instrumenter-final-cli-20261027`, and described in PRs #154, #155 and #157.
Raw traces, profiles and checked harnesses remain in the experiment directories;
only the reviewed report and compact measurement evidence are committed here.


## Asynchronous snapshot work

File writes already use background threads on the subclass path. Moving only
final JSON encoding off the producer offers little headroom in this study.
The larger synchronous work inspects live objects and prepares their metadata.
Moving those getters to a later worker after a shallow copy can observe changed
values and change when errors occur. A future asynchronous tensor snapshot path
needs owned snapshots, producer-stream ordering, bounded bytes/items, ordering
per destination, and explicit worker-error and drain behavior. This change adds
no asynchronous snapshot conversion. Per-attribute warning setup is the next
measured implementation target, provided getter and warning semantics remain
intact. Selective-checking overhead needs its own timing study.
