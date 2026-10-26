# Transformer tracker correctness and metadata fast-path measurements

The fixed subclass tracker completes the previously failing Transformer workload.
In five alternating pairs with **both API and model tracing active**, the metadata
classification candidate reduced median three-step CPU training time by **8.67%
for MNIST** and **18.90% for the small Transformer**. These are reductions from the
baseline's training-body time with both tracers enabled under ordinary collection
and its normal API exclusions (`use_full_instr=False`). They are not whole-process
speedups or comparisons against the earlier PR #153 measurements.

## Exact comparison and scope

| Variant | Commit | Tree |
|---|---|---|
| Baseline: subclass assignment fix | `88b6df2401e187ca6bd81bf0a83bc7c5d9a5cd54` | `5415a3b0e5eb08d119dffa91631d76866e6e1d11` |
| Candidate: same fix plus callable metadata fast path | `3232887a8bdb6dbf6abcab233ead43a9697f0c18` | `cb74fdd3a8185deb0b3d31c0d20d2a7eac0b8097` |

All 77 executed production Python files per variant match the respective exact
Git archive byte for byte. The only differing production file is
`traincheck/instrumentor/dumper.py`. Hash algorithms, hash backend, tensor dump
format, sampling, observer depth, and tracing coverage remain unchanged.

The performance change recognizes exact built-in callable types before testing
them against framework tensor/module classes. It retains attribute discovery and
warning handling, and retains the generic path for arbitrary callable objects and
subclasses. This study does not change or optimize hashing.

## Workload and timing contract

- labmach, Intel Core i9-13900K, Python 3.11.4, PyTorch
  `2.16.0.dev20261012+cpu`; one PyTorch/native thread, CPU only, deterministic
  algorithms, batch 2, float32, three cold training steps.
- MNIST: exact benchmark `Net`, 1,199,882 parameters, first six cached MNIST
  training images, NLL loss and Adadelta with learning rate 1.
- Transformer: exact benchmark model definitions with bounded dimensions
  (embedding 32, two layers, two heads, feedforward 64, sequence length 16,
  vocabulary 60 from the WikiText-2 prefix). Original gradient clipping and
  **manual SGD parameter updates remain present**.
- Initialization seeds 42; training seeds 43 for Python, NumPy and PyTorch.
- Explicit subclass parameter tracking before optimizer construction; normal
  depth-3 static observers; parameter hash format enabled, CPU backend;
  `scan_proxy_in_args=True`; API descriptors disabled; sampling interval 1,
  warmup 0; `use_full_instr=False`, retaining normal API exclusions. This does
  not exercise `--use-full-instr`, default Proxy, or VarSampler mode.
- Ordinary timing includes `start_step`, the training body, and disabling
  instrumentation on exit. It excludes imports/setup, loss extraction, RNG/state
  verification and final background-writer drain. **RNG snapshots are outside
  the timer and after `DISABLE_WRAPPER=True`**, so they add no API trace records.
  Separate cProfile runs cover the training body, not setup or diagnostics.
- Every case uses a fresh process. Five pairs per workload alternate
  baseline/candidate, candidate/baseline, baseline/candidate, candidate/baseline,
  baseline/candidate. The shared host was not reserved or pinned to a CPU.
- Limits: 90 seconds, 4 GiB sampled main-process RSS, 256 MiB trace per case.
  No case hit a limit. Native threads were restricted to one; other users' work
  and shared environments were not changed.

## Correctness smoke

The fixed baseline passed plain, API-only, VAR-only and combined Transformer
three-step runs. Initial/final model state, every loss value, and Python/NumPy/
PyTorch CPU RNG state before training and after each step matched exactly.

| Mode | Single smoke time (ms) | API records | VAR records | Tracked parameters |
|---|---:|---:|---:|---:|
| Plain | 15.418 | 0 | 0 | 0 |
| API | 877.959 | 1,338 | 0 | 0 |
| VAR | 88.366 | 0 | 64 | 29 |
| Combined | 940.872 | 1,340 | 64 | 29 |

These smoke times establish execution, not a repeated ablation estimate.
VAR-only keeps Instrumentor's no-dump wrappers. All modes import TrainCheck's
global module-assignment hook; plain installs no API wrappers or parameter
proxies. Modes with VAR tracking also install observers and argument scanning,
so four-mode differences are not a physically exclusive cost partition.

## Ordinary paired measurements

| Workload | Baseline median (range), ms | Candidate median (range), ms | Reduction from baseline median | Ratio of medians |
|---|---:|---:|---:|---:|
| MNIST | 868.830 (831.441–943.226) | 793.489 (722.298–849.878) | 75.341 ms / **8.67%** | 1.095× |
| Transformer | 966.440 (957.389–1,028.738) | 783.771 (775.097–791.462) | 182.669 ms / **18.90%** | 1.233× |

All individual ordinary runs, in repetition order:

| Workload | Baseline times (ms) | Candidate times (ms) | Paired reductions (ms) |
|---|---|---|---|
| MNIST | 943.226, 868.830, 831.441, 903.219, 833.264 | 814.235, 722.298, 793.489, 792.977, 849.878 | 128.991, 146.531, 37.952, 110.242, **−16.615** |
| Transformer | 958.257, 957.389, 966.440, 971.936, 1,028.738 | 775.097, 791.462, 784.180, 777.745, 783.771 | 183.160, 165.927, 182.261, 194.191, 244.967 |

MNIST improves in four of five pairs; its paired reductions range from −1.99%
to 16.87%. Transformer improves in all five pairs (17.33%–23.81%). These are
bounded cold-step diagnostics on a shared host, not confidence intervals or
steady-state/GPU/distributed scaling claims. No run was discarded as an outlier.

## Profiles: what changed and what remains

Each variant has one separate cProfile run per workload. Total profiled time is
1.599957 → 1.322230 seconds for MNIST and 2.834436 → 2.173404 seconds for
Transformer. Percentages below use **that variant's profiled total**. Cumulative
rows overlap; do not add them or apply them to the ordinary timing denominator.

| Profile component | MNIST baseline → candidate, ms (% of respective total) | Transformer baseline → candidate, ms (% of respective total) |
|---|---|---|
| All `convert_var_to_dict`, cumulative | 1,537.323 (96.09%) → 1,257.303 (95.09%) | 2,748.502 (96.97%) → 2,091.410 (96.23%) |
| Built-in `isinstance`, cumulative | 171.536 (10.72%) → 20.463 (1.55%) | 427.563 (15.08%) → 52.964 (2.44%) |
| Dumper `safe_getattr`, cumulative | 626.707 (39.17%) → 605.309 (45.78%) | 1,579.442 (55.72%) → 1,498.406 (68.94%) |
| Warning `simplefilter`, cumulative, nested in `safe_getattr` | 227.450 (14.22%) → 219.941 (16.63%) | 579.481 (20.44%) → 552.660 (25.43%) |
| API argument conversion, cumulative | 600.868 (37.55%) → 466.125 (35.25%) | 1,578.695 (55.70%) → 1,207.365 (55.55%) |
| API return conversion, cumulative | 291.809 (18.24%) → 225.436 (17.05%) | 987.921 (34.85%) → 752.378 (34.62%) |
| `ProxyParameter.dump_trace`, cumulative | 653.538 (40.85%) → 574.950 (43.48%) | 213.353 (7.53%) → 162.592 (7.48%) |
| `tensor_hash`, cumulative, nested in snapshots | 437.785 (27.36%) → 414.738 (31.37%) | 5.762 (0.20%) → 5.312 (0.24%) |
| Final `serialize`, cumulative | 4.350 (0.27%) → 4.386 (0.33%) | 12.093 (0.43%) → 11.245 (0.52%) |

The classifier evidence is concrete: `isinstance` calls fall from 1,503,363 to
464,357 for MNIST and 3,622,491 to 1,154,245 for Transformer. Parameter
`__instancecheck__` calls fall from 203,335 to 2,817 and 495,303 to 8,937,
respectively (Transformer also has one other instance-check entry in both
variants). The number of attribute lookups/warning scopes is unchanged:
211,616 for MNIST and 516,009 for Transformer. Thus repeated per-attribute
lookup/warning setup remains an actionable instrumentation bottleneck; this
candidate does not remove it. Final JSON encoding is a small part of these
profiles. Hash call counts remain 112 and 64; hash timing variation is not
attributed to a hash optimization.

## Activation, data integrity and resource accounting

All **28 original cases** (four smoke, twenty ordinary, four profile) completed
successfully. The paired driver initially compared each workload against its
first baseline combined run. A supplementary check explicitly anchors all 24
ordinary/profile cases to plain training: the existing Transformer plain smoke
and a new isolated MNIST plain run using the exact baseline source. Every initial
and final state digest/shape, loss, step ID, and per-step Python/NumPy/PyTorch RNG
snapshot matches. Timing fields alone are excluded from this behavioral comparison.
The supplemental run supplies a correctness reference, not another speedup estimate.
All saved raw trace SHA-256s, sizes, counts and API pre/post pair multiplicities
were independently reread and validated locally.
Every tracked parameter has at least two distinct data hashes across the three
steps. The five ordinary runs and separate profiles have matching coverage
signatures across variants:

| Workload | API records / complete pairs | VAR records / parameters | Data+gradient hash fields | Raw API+VAR bytes |
|---|---:|---:|---:|---:|
| MNIST | 520 / 260 | 64 / 8 | 112 | 419,429 |
| Transformer | 1,340 / 670 | 64 / 29 | 64 | 1,088,456 |

Transformer snapshots have `grad=None` and capture changed parameter values at
the current observers' assignment/zero-grad observation points. They do **not**
establish pre/post snapshots for each manual SGD update, or gradient snapshot
coverage. The final state check independently verifies that all three training
steps and their updates matched. We did not replace manual SGD or add synthetic
observers to improve coverage. MNIST's Adadelta path has data and gradient hashes.

Logical hashed input volume, counting repeated snapshots, is 67,193,392 bytes
for MNIST and 214,496 bytes for Transformer. This is not physical memory traffic
or a time partition. No cross-process raw trace equality is claimed: identities
and timestamps remain intact in the retained raw records.

| Ordinary median | MNIST baseline → candidate | Transformer baseline → candidate |
|---|---:|---:|
| Setup | 2.702 → 2.709 s | 2.603 → 2.609 s |
| Whole process | 7.018 → 7.019 s | 7.018 → 6.817 s |
| Post-metrics process exit | 2.078 → 2.131 s | 2.039 → 2.040 s |
| Sampled main-process peak RSS | 561.492 → 561.285 MiB | 515.441 → 515.648 MiB |

Post-metrics exit includes writer drain, Python shutdown and up to one polling
interval; it is **not a pure I/O drain timer**. The small training-body savings
do not materially reduce whole-process MNIST time in this short workload. Full
per-run distributions for setup/process/exit/RSS/bytes are in [`data/metadata-classification-20261027.json`](data/metadata-classification-20261027.json).

## Reproduction and artifacts

Remote root:
`/home/yuxuan/experiments/codex-instrumenter-performance-20261027-fixed`.
Local root:
`/Users/matt/gitrepos/traincheck/outputs/instrumenter-performance-20261027`.
Use a fresh copy of the experiment directory for reruns because run directories
are deliberately created exclusively. `baseline/source.tar` and
`candidate/source.tar` retain the exact source archives; the only experimental
adapter change in `source/tools/fresh_validation/benchmark_workload.py` is the
typed registration callback before optimizer construction, identical in both
variants. Retained harnesses are `component_runner.py` and `run_experiment.py`;
`analyze_experiment.py` rereads and validates the evidence.

```bash
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python run_experiment.py --root "$EXPERIMENT_ROOT" --phase smoke
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python run_experiment.py --root "$EXPERIMENT_ROOT" --phase paired
python analyze_experiment.py --root "$LOCAL_EVIDENCE_ROOT"
```

The runner sets `PYTHONPATH` to each isolated source tree, restricts native
threads, disables bytecode writes and CUDA visibility, and records exact child
commands. It aborts on execution, coverage, loss, model-state or RNG divergence.
No installation or shared source/environment edit was needed.

| Artifact | SHA-256 |
|---|---|
| Baseline source archive | `ac1a830f490bef1af6b1b15ba49d116ff28a8de2e57f51dd80afc714f4e8d909` |
| Candidate source archive | `b945a9d4c2b43def8481660bc3fc6ae45dbfbd34e3a5e8a0c848e5ca379276e6` |
| Raw evidence archive, 24,826,924 bytes | `1de069eb0dd5d8dd4ce60e758a4d3dae5716fa69bda1f3e30570b983f4e69773` |
| Executed runner | `d17315872c4e412d3a2be075fa733eebb9c7d4f8189dff82d53dc6bae92c0818` |
| Executed driver | `3a89d821238902362b110abbc9d66bb2b87c1b1cef09144bf7407a5400cd72f4` |

The raw archive retains every log, metrics/manifest file, trace, profile and
environment inventory. Large/raw data remains in outputs and the isolated
remote directory, not in commits.

Scoped mypy passes for all three new harness/analysis modules with
`--check-untyped-defs --disallow-untyped-defs --disallow-any-generics` and
`--follow-imports=skip --ignore-missing-imports`. The latter two options mean
dependency typing is not fully checked. Production test/check results belong
to the focused implementation PRs; this report records the actual workload
validation only.

## Asynchronous work: what this measurement supports

The subclass API/VAR file writers already run in background threads. Producers
still inspect live values and build immutable serialized records before enqueueing
so later application mutations cannot change earlier observations. Final JSON
encoding contributes 0.27–0.52% of the measured profiles; moving just that encoding
to a worker has little headroom here. The larger synchronous work is metadata
inspection and conversion, including per-attribute warning scopes.

Moving arbitrary object inspection to a worker after a shallow copy would change
which values, getters and exceptions are observed. A future asynchronous tensor
snapshot path needs owned copies, producer-stream ordering for CUDA, bounded byte
and item queues, per-destination ordering, and explicit flush/error behavior.
It would need separate contract tests and an explicit decision about delayed
serialization errors. This PR implements the measured synchronous classification
fast path only; it does not add asynchronous snapshot conversion.

The separate CLI default validation also reproduced a pre-existing startup
recursion under `--use-full-instr` in `torch._ops.__getattr__` on unchanged main
and the subclass candidate. Those broader-flag runs were stopped at 45 seconds;
they are not included in the successful ordinary-collection measurements above.

Plain-reference validation is included in the measurement JSON under
`plain_reference_validation`. Its raw supplementary archive is retained as
`plain-parity.tar.gz` (12,709 bytes), SHA-256
`6093c207f75f30331248220838613369a50d6fa3a4f6f534e0075ce7f8f4d4c1`,
with its checker alongside the other local evidence. The original 28-case
archive is unchanged.
