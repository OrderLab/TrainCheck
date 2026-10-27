# Native tensor method optimization on H100

Ordinary job **55325956** measures the metadata optimization on the native
TorchTitan Llama debug workload. The median paired steady-step reduction is
**17.779% (1.2162x)**. Including cold steps, the complete ten-step training body
takes **12.568243 to 10.764749 seconds, 14.350% less time**.

This is **selective API trace collection**, with no concurrent online checker
and no model tracker. The one-GPU experiment retains native FSDP but does not
establish multi-GPU scaling. CPU results are reported separately in
[the original measurement](native-tensor-methods-20261027.md).

## Ordinary training measurements

Each fresh process trains the same 6,163,712-parameter model with seed 42 for ten
steps, batch one and sequence length 256. Each table entry is the median of
synchronized native steps 3–10. Pair order was fixed before measurement.

| Pair / order | Baseline | Candidate | Time reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 1 / AB | 1,000.882 ms | 821.738 ms | 17.899% | 1.2180x |
| 2 / BA | 997.812 ms | 821.305 ms | 17.689% | 1.2149x |
| 3 / AB | 998.726 ms | 821.159 ms | 17.779% | 1.2162x |

Percentages use the baseline duration as denominator. The median paired reduction
is 17.779%; the separate ratio of run medians is 998.726 to 821.305 ms, or 17.765%.
The full-body 14.350% reduction includes steps 1–2. Body timings exclude imports,
initialization, state evidence, outer loop housekeeping, queue drain and shutdown.
Every raw step/phase duration is retained; no profiled run contributes to this table.

A single plain-control run has a 44.178 ms steady-step median. Relative to it,
the baseline/candidate cost is **22.61x / 18.59x**, or **+2,160.68% / +1,759.07%**.
These plain ratios have only one control repetition. The optimization helps, but
selective collection remains expensive on this small model.

A separate seed-43, 20-step baseline/candidate holdout has medians over steps 3–20
of 996.772 to 822.673 ms (17.466% less). This is one validation pair, excluded from
the three-pair performance estimate.

## Avoided work, measured separately

Diagnostic job **55325959** counts actual helper decisions during steps 3–10.
Its wrappers retain no tensor references and preserve the original return values
and exceptions. Counters cover both calling threads and reconcile with getters.

| Scope | Helper calls | Discarded method bindings skipped | Remaining guarded reads |
| --- | ---: | ---: | ---: |
| All threads | 1,940,048 | 1,698,000 (87.52%) | 242,048 |
| Window owner | 1,243,312 | 1,100,304 (88.50%) | 143,008 |
| Other trace-producing thread | 696,736 | 597,696 (85.79%) | 99,040 |

Exact Tensor receivers account for all 1,698,000 skips, from 1,788,000 helper calls
(94.97% eligible). Other receivers have 152,048 calls and no skips. Exact Parameter
calls are **not observed**; this is not a measured zero eligibility percentage.

The prior accepted owner-thread profile counted 1,243,312 `safe_getattr` calls.
The matching denominator here is the new **owner-thread** helper count, not the
1,940,048 all-thread count. These are avoided getter/warning-context counts, not
percentages of total runtime. The remaining 242,048 reads keep the original
per-attribute warning restoration. No global warning suppression was introduced.

The diagnostic perturbs its window by 1.38557x (9.10619 versus 6.57215 seconds)
relative to its ordinary candidate control; those durations are excluded from
performance claims. Both corpora have 416,364 rows, 208,182 complete calls,
142,621,017 bytes and zero raw audit issues. Every per-step/API/event-kind count
and per-step byte total matches. This establishes volume/integrity parity, not
byte equality of identity-bearing records. The collector checks wrapper identity
at restoration but did not record pre-install binding identity; that runtime
assertion remains unobserved rather than being inferred as passed.

## Application and checking validation

Independent analysis verifies all nine phase commands, successful exits, ordered
step counts, finite timings, source stability, environment and selective policy.
Every seed-42 run exactly matches the plain control's complete recorded initial
and final model/gradient/buffer/RNG fingerprints, losses, model configuration,
parameter count and parameter types. The seed-43 candidate matches its selective
baseline on the same fields. Tensor evidence records names, shapes, dtypes and
byte hashes; it is not an archived full optimizer-state checkpoint. No recorded
IDs, arrays, hashes or values were normalized to obtain state parity.

All eight selective phases use the same 722 frozen invariants and identical
compiled policy bytes. Raw trace audits and observed/direct checking of the
baseline/candidate reference and holdout corpora are running separately on
labmach. Invariant activation and checker parity are **pending**, not implied by
the timing/state or diagnostic-volume checks above.

## Revision, environment and reproduction

- Baseline: composed `8cc73ab35590caa1a5d7286b16462d045b6d2da0`.
- Candidate: only `dumper.py` and new `metadata_methods.py` from signed
  `287b3a2cb97f28e8f7e5b21210781d9c65541f2d`, overlaid on that baseline.
  CUDA hashing, selective policy and workload source remain identical.
- TorchTitan: `2210414902d84d1219fd63ab8efeb24375a2a1a1`.
- Expanse, one NVIDIA H100 80GB HBM3; Python 3.12.3,
  Torch `2.16.0.dev20261012+cu130`, CUDA 13.0. Eight allocated CPUs, 48 GiB RAM,
  Torch/BLAS threads fixed by the saved launcher. Jobs run on allocated compute.
- Ordinary job completed in 4:35; diagnostic in 0:56. Resource accounting is
  saved. Peak batch memory covers entire jobs, not isolated metadata overhead;
  no per-variant memory improvement is claimed.

The first job, **55325946**, failed before candidate training because CPython 3.12
exposes `type.__mro__` as a native getset descriptor rather than the member
descriptor used in 3.10/3.11. The fix accepts both explicit native types while
retaining validation and rejecting application properties. Failure was reproduced
locally before fixing. Afterwards 67 affected tests pass on Python 3.12, 108
focused tests on Python 3.10, and strict scoped typing/pinned checks pass. The full
local suite has 328 passes and the same two failures reproduced on unchanged main.
The failed job is preserved separately and excluded from all timing estimates.

Local artifacts are under `outputs/metadata-fastpath-20261027/`. Lab validation
uses its isolated `validation-job-55325956` directory. Full source inventories,
frozen inputs, commands, raw traces, per-step timings, failure logs and validators
remain outside Git; the public report contains no trace or credential payloads.

| Artifact | SHA-256 |
| --- | --- |
| Composed baseline source archive | `e65d03f567850a94a3fd1c1e5f1870bb97363694803484534854df4ea497e37d` |
| Corrected candidate source archive | `c32333538188e69e8b856e65cb7e4a2bf5090b1d177762e0f1448e1a062a4fa8` |
| Ordinary results archive | `06503d2b2ad0887ffe23950455d8d6f52372a32d10b7392d985f024f394cb311` |
| Diagnostic results archive | `f9369d3d448806554d1e1f07f2df576ba65d688bdfc88fb91495bd29068bf2df` |

From the local artifact directory:

```sh
python analyze_gpu_results.py \
  --ordinary gpu/job-55325956/results/ordinary \
  --output gpu/job-55325956/ordinary-analysis.json
python gpu-diagnostic/validate_hit_counts_v2.py \
  --results gpu/job-55325959/results \
  --output gpu/job-55325959/hit-count-validation.json
python gpu-diagnostic/summarize_hit_counts.py \
  --validation gpu/job-55325959/hit-count-validation.json \
  --counter gpu/job-55325959/results/diagnostic/counters/hit-counts.json \
  --prior-profile ../checking-overhead-20261027/final-component-profile-analysis.json \
  --output gpu/job-55325959/hit-count-analysis.json
```

The ordinary analyzer has 11 regression tests, including failed/missing phases,
negative speedups, policy/hash changes and malformed/nonfinite data. The counter
and post-download validators test concurrency, exceptions, restoration, empty
windows and provenance failures. New reusable helpers pass strict typing and the
repository's pinned formatting/lint checks. Exact submission commands are saved
in `gpu/README.md` and `gpu-diagnostic/README.md`; they verify packages and enforce
time/trace limits before running the isolated phases.
