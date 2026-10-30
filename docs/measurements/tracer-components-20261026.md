# API and model-tracer overhead: bounded CPU measurements

The previous tracing audit had **API tracing only**. This follow-up activates the
`subclass` model tracer and default tensor-hash dumping, and separates four real
configurations. It changes no production code. MNIST completes with exact training
state/loss parity; Transformer model tracing exposes a pre-existing failure, so no
Transformer model-tracing overhead is reported.

## Ordinary training timings

Each fresh process trains the exact benchmark MNIST `Net` (1,199,882 parameters)
for three CPU steps on the first six cached MNIST images, batch 2, fp32, Adadelta,
initialization/training seeds 42/43, and one native thread. Three repeats use orders
plain/API/VAR/both, both/VAR/API/plain, and VAR/plain/both/API. These are cold
three-step diagnostics, not steady-state or large-scale training estimates.

| Mode | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Plain | 29.668 ms | 29.061 ms | 31.311 ms | 29.668 ms |
| API only | 343.709 ms | 338.689 ms | 335.543 ms | 338.689 ms |
| Model/VAR only, hash enabled | 372.319 ms | 414.599 ms | 474.165 ms | 414.599 ms |
| API + model/VAR, hash enabled | 707.320 ms | 727.127 ms | 752.502 ms | 727.127 ms |

The timer includes the training body, `start_step`, and wrapper-disable assignment.
It excludes imports, instrumentation/model setup, loss extraction, full model-state
verification, and final background-writer drain. The plain process imports
TrainCheck, including its global Module `__setattr__` hook, but installs neither
API wrappers nor proxied parameters; this shared import is a stated baseline limit.

Subtracting medians gives **697.459 ms added time** with both tracers. The ablation
accounting is:

| Term | Added time | Share of 697.459 ms added time |
|---|---:|---:|
| API only minus plain | 309.021 ms | 44.31% |
| VAR only minus plain | 384.932 ms | 55.19% |
| Interaction: both - API - VAR + plain | 3.506 ms | 0.50% |

The marginal cost of adding model tracking to API tracing is **388.438 ms**.
These are differences between measured configuration medians, **not an exclusive
physical stage timer**. The interaction includes shared work, configuration
differences, and timing noise. The VAR-only range is material; three repeats do
not resolve a 3.5 ms interaction reliably, and interaction may be negative in
another matched run. Percentages above use added time, not total 727.127 ms.

Sampled peak main-process RSS, including imports/setup rather than isolated
trace-buffer memory: plain: 442.8–464.9 MiB; api: 536.1–549.1 MiB; var: 548.5–552.2 MiB; combined: 563.6–563.8 MiB.

## What actually activated

Model parameters are converted to `ProxyParameter` **before optimizer creation**.
The normal static observer lookup tables are installed at depth 3, with hidden
functions/modules excluded and `observe_then_unproxy=False`. Every VAR-enabled
run registers all eight model parameters and emits **64 VAR records**, each with
a tensor-data hash; 48 also contain a gradient hash. All eight parameters have
pre/post-Adadelta observations with changed data hashes, so this is active model
tracing rather than configuration-only registration. There are **112 tensor hash
fields** over three steps, representing 67,193,392 logical input bytes (64.081 MiB).
That is logical tensor size, not allocated memory or transfer volume.
The large `fc1.weight` tensor (128 x 9,216) accounts for 66,060,288 bytes, or
98.31% of that logical volume. This identifies a concrete hash workload; it does
not mean that tensor accounts for 98.31% of hash time. The CPU implementation's
column loop also makes the last dimension relevant, not only the total bytes.

API-enabled runs emit **520 API records** (260 complete pre/post pairs). Plain and
VAR-only emit no API records; plain and API-only emit no VAR records. Counts and
complete raw call pairs are independently checked after downloading the traces.

These numbers apply to `subclass` mode. The CLI defaults to `proxy`, whose
`json_dumper` writes synchronously; that distinct mode has not been measured here.
The subclass mode uses the buffered VAR writer.

The modes are real activation ablations, not identical wrapper topology:

- VAR modes install static observers and enable `scan_proxy_in_args`; API-only
  omits them, matching normal collection configuration.
- VAR-only passes `funcs_to_instr=[]` to `Instrumentor`, retaining production
  no-dump wrappers that take a faster path than API record construction.
- The original model dump and observer paths remain untouched; no new suppression,
  sampling, field omission, or hash-backend change is applied for this measurement.

## Separate combined profile

A separate three-step combined cProfile run totals **1.495160 s**, versus the
ordinary combined median 0.727127 s. It preserves the complete loss trajectory and
initial/final model state. These cumulative rows overlap and must not be added;
their percentages are of **profiled main-thread time**, not ordinary added time.

| Entry point | Calls | Cumulative time | Share of profile |
|---|---:|---:|---:|
| API argument conversion | 260 | 590.920 ms | 39.52% |
| API return conversion | 260 | 285.949 ms | 19.12% |
| `ProxyParameter.dump_trace` | 64 | 568.882 ms | 38.05% |
| `tensor_hash` (nested inside variable snapshots) | 112 | 362.960 ms | 24.28% |
| `hash_tensor_cpu` (nested inside `tensor_hash`) | 112 | 321.312 ms | 21.49% |
| Final record JSON `serialize`, API + VAR | 584 | 4.557 ms | 0.305% |

The caller graph identifies all 112 hash calls beneath `dump_tensor`. The variable
entry has 48 observer-triggered calls plus 16 attribute-update-triggered calls.
Hashing is a substantial part of the now-active variable path. API argument/return
conversion together accounts for 58.65% cumulative profiled time, but this sum and
the variable entry are presented as diagnostic entry points; the aggregated
profile is not used to claim a disjoint semantic stage partition.

## Transformer failure instead of an overhead number

Plain and API-only Transformer runs complete. The first VAR-only run fails before
finishing its first step when `TransformerModel.forward` assigns `self.src_mask =
mask`. The model tracer's `subclass_setattr_hook` accepts either a Tensor or Module
and forwards the value to `proxy_parameter`; that helper assumes an `nn.Module`
and calls `named_parameters` on the plain Tensor:

```text
subclass.py:286 wrapped_setattr -> subclass_setattr_hook
subclass.py:263 subclass_setattr_hook -> proxy_parameter(value, ...)
subclass.py:231 proxy_parameter -> module.named_parameters(recurse=False)
AttributeError: 'Tensor' object has no attribute 'named_parameters'
```

The matrix stops at this failure; it does not silently change tracker style, omit
the mask assignment, or claim a successful Transformer VAR/combined measurement.
The failed case retains its log, manifest, partial state and explicit error.

The [compact evidence](data/tracer-components-20261026.json) retains all
per-run timings, activation checks, profile rows, and the explicit failed case.

## Provenance and validation

- Frozen source: release commit `9ba90ac84a1b4efd7e9f65b48b36bbbebf958253`, tree
  `6fbebbeae8f1c4327d39452361325b9c716a4068` (`v0.2.0`). All 77 production
  Python files byte-match the release archive. Only the isolated workload adapter
  receives a typed model-registration callback immediately after construction.
- Host/runtime: labmach, Intel Core i9-13900K, Python 3.11.4, PyTorch
  `2.16.0.dev20261012+cpu`, one native thread, no CPU affinity/exclusive reservation.
- All 12 ordinary MNIST cases and the separate profile have exactly matching full
  loss trajectories, canonical initial/final model hashes, names, dtypes and
  shapes. Raw trace digests/counts/pre-post multiplicity and active hash fields
  were rechecked locally. No IDs or trace contents were normalized to claim raw
  trace equality across processes.
- Each case is capped at 90 s, 4 GiB main-process RSS and 256 MiB raw trace. No
  successful MNIST case hits a cap. No GPU experiment runs in this matrix.
- `component_runner.py`, `run_matrix.py`, and `analyze_components.py` pass mypy
  with `--check-untyped-defs --disallow-untyped-defs --disallow-any-generics`
  (`--follow-imports=skip --ignore-missing-imports` limits dependency coverage).
- Retained data: `component-summary.json` / `evidence.json`; exact scripts,
  commands, manifests, profiles and raw traces under local
  `outputs/tracing-components-20261026/components/` and remote
  `/home/yuxuan/experiments/codex-tracing-components-20261026-v020`.
- Archive `evidence.tar.gz`: 12,483,879 bytes; SHA-256
  `d9dad870e9b26d2b076d09d6f56f6037cdb4948d4af0e58f26e7aa2105f594ff`.

Recompute the validated summary locally with:

```sh
python analyze_components.py --root /path/to/retained/components
```

The matrix commands and changed adapter are retained for exact reproduction.
Existing run directories are never overwritten. A new run should use a fresh
experiment root. The Transformer failure should be reproduced/fixed separately
before extending its overhead claims.
