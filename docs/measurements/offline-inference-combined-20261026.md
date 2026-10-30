# Direct comparison of complete offline inference

One matched, unprofiled pair on the retained TorchTitan Llama debug training
trace reduced complete offline-inference wall time from **1,439.538 s to 76.686 s**: **18.77x faster**, or
**94.67% less wall time**. This directly measures the combined result against the
pre-identity-cache baseline. It does not multiply the earlier PR speedups.

The denominator is the entire offline-inference process: startup/imports, trace
loading and preparation, hypothesis generation, precondition inference, output
writing, and process exit. It excludes training, trace collection, and invariant
checking. The result supports an approximately 19x improvement for this saved
trace and configuration, not a general training-pipeline speedup.

## Measured result

| Measurement | Corrected baseline | Final main | Change |
|---|---:|---:|---:|
| Complete offline wall time | 1,439.538 s | 76.686 s | 18.77x; 94.67% reduction |
| Sampled peak process-tree PSS | 2.126628 GiB | 2.165641 GiB | +39.95 MiB; +1.83% |
| Invariants written | 836 | 836 | Full file byte-identical |
| Failed hypotheses written | 104 | 104 | Full file byte-identical |

Both runs completed with exit code zero, without reaching the 2,400-second or
12-GiB sampled-PSS limit. Memory was sampled once per second; this is a sampled
peak rather than an exact allocation high-water mark. The monitor's own memory
is outside the measured child process tree. Its polling can add up to roughly
one second to the measured process lifetime. Source/input integrity hashing is
outside the wall timer. The retained harness serializes context snapshots inside
each child process; this shared diagnostic cost is included in complete wall
time and the relevant loading/inference timers.

Unprofiled log timestamps show where the elapsed time moved:

| Phase | Baseline time | Share of baseline total | Final-main time | Share of final-main total |
|---|---:|---:|---:|---:|
| Trace loading, reported timer | 10.777 s | 0.75% | 10.745 s | 14.01% |
| Hypothesis generation and pruning | 641.082 s | 44.53% | 49.536 s | 64.60% |
| Precondition inference and output sorting | 782.884 s | 54.38% | 12.512 s | 16.32% |

Each percentage uses that column's complete wall time as its denominator. These
phase markers do not cover every startup, shutdown, and output operation and
should not be treated as an exhaustive partition or profiler attribution. The
trace-loading cost is nearly unchanged because both variants already contain
the context/call indexes from #141.

## Sources and correctness

The baseline is the exact frozen `call-index-src` snapshot used before #144's
identity-cache optimization, with only #145's two-hunk pruning-state correction
applied. This archived snapshot predates publication-only type/comment changes
and is identified by file hashes, not presented as a single Git revision. Its
original provenance manifest SHA-256 is
`7e8e267085ece31844a820752004af6a1c5bb293a3a2733990b039d72fcd872a`.
The baseline already contains the earlier deterministic sampling, reservoir,
and context/call-index work; their gains are not included in this comparison.

The candidate is main commit `205a687eef30e498bd2d9a9951652b7cc05bec68`, tree
`2177e2fa826bb4623c3c9cdbb3bdcb61fafbedab`. The corrected `precondition.py` is
byte-identical in both variants, and all six pruning-state regressions pass on
each. This holds the independent correctness fix constant while comparing the
remaining inference optimizations and hardening against final main.

All **836 invariants**, **104 failed hypotheses**, and both complete context
snapshots (**711 states each**) match byte for byte. No records, IDs, predicates,
array contents, duplicates, or output ordering were discarded to obtain parity.
The two input traces and all source hashes remain unchanged before/after each
run. The retained harness/helper hashes are identical to the earlier studies.

Independent download verification rehashed all 330 archived evidence files,
matched all 118 candidate Python files against the specified Git revision,
and confirmed that the only change to the original 76-file baseline is the
pruning-state fix. Both run provenance manifests match their archived sources.
All raw-output comparisons were repeated locally on the downloaded files.
The [compact evidence](data/offline-inference-combined-20261026.json) records
the exact times, memory, digests, settings, and validation results.

## Workload and controls

The same two original traces contain **573,494 events / 286,335,289 raw bytes**
from ten training steps of TorchTitan `llama3_debugmodel` (6,163,712 parameters),
sequence length 256, batch size 1, seed 42, run `20261012T163111Z-88c4d357`.
This is TorchTitan's small Llama debug configuration, not the separately named
TinyLlama model/project. The original collection manifest identifies that
configuration and TorchTitan revision
`2210414902d84d1219fd63ab8efeb24375a2a1a1`; its SHA-256 is recorded in the
compact evidence. The trace files are read unchanged; no training is rerun here.

Both fresh processes use pandas, one inference worker, native thread limits of
one, `PYTHONHASHSEED=0`, and CPU execution. Six relations are enabled:
`APIContainRelation`, `ConsistencyRelation`, `ConsistentOutputRelation`,
`ConsistentInputOutputRelation`, `DistinctArgumentRelation`, and
`ThresholdRelation`. `FunctionLeadRelation` and `FunctionCoverRelation` are
disabled, matching the earlier studies. Each run sees 41 eligible APIs.

Runtime: labmach, Intel Core i9-13900K with 32 logical CPUs, Linux 6.1.0-1027-oem,
Python 3.11.4, PyTorch `2.16.0.dev20261012+cpu` at Torch revision
`8f9045d72f71e7aa9e16b6d7a055ec5bc1f720b5`, pandas 2.3.3, NumPy 2.1.0.
Original collection used Python 3.12.3 and a CUDA build at the same Torch source
revision. This comparison controls the replay environment between variants,
not between original collection and replay.

This is **one baseline-first pair**, without repeats, randomized order, CPU
affinity, or an exclusive host reservation; no confidence interval is claimed.
One unrelated Python process was using approximately one CPU core at both
pre-run snapshots. The separate tracing audit finished before this pair began.
The result does not establish distributed scaling, GPU performance, portability
of these invariants to fresh runs, or overhead of selective checking.

## Retained evidence and reproduction

The complete evidence archive is 5,479,198 bytes, SHA-256
`776ee54a40fa883d167c3cccef3a40105a82f396f5235d8d566a93112c097823`.
It contains both source snapshots, the pruning patch, exact commands, unchanged
replay harness/helper, controller, runtime/resource snapshots, per-second memory
samples, full output/context files, and logs. Raw input traces remain at their
existing paths and are identified by full digests in the evidence.

Remote experiment directory:
`/home/yuxuan/experiments/codex-release-inference-20261026-205a687`.
The downloaded archive, independently checked results, and source manifest are
under the local project's `outputs/release-20261026/inference-pair/`.
No raw traces or machine-specific experiment files are committed.

To repeat on labmach, create a new directory and copy the retained
`baseline-src/`, `candidate-src/`, `source-manifest.json`, `run_pair.py`,
`inference_replay_fatal_only.py`, and `inference_scaling.py` into it. Keep the
original trace paths from the manifest available, then run:

```sh
/home/yuxuan/experiments/codex-inference-runtime-20261025/venv/bin/python \
  "$NEW_STUDY/run_pair.py"
```

The controller runs baseline then candidate with the same limits and refuses
to overwrite measurement directories. `comparison.json` records direct raw
byte equality in addition to the inherited harness's structural summaries.
