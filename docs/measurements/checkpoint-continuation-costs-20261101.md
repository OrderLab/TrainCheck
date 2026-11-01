# Distributed checkpoint continuation: agent efficiency pilot

**This case did not demonstrate a TrainCheck debugging-efficiency benefit.** All four agents localized the same checkpoint binding problem from source and submitted a correct repair before their first remote request. Both treatment agents chose ordinary Python diagnostics; neither imported or activated TrainCheck. This measures optional tool availability, and provides no estimate of the effect or overhead of active TrainCheck use.

All four final selections and first submissions pass independent full-state and continuation checks: the original two-rank checkpoint, two fresh two-rank seeds, and a fresh single-rank compatibility case. There are 16 passing contract reports, independently replayed locally. This is a small synthetic development pilot, not an official RSI benchmark result or a general causal estimate.

## Measured milestones

| Trial | Supported localization (s) | First valid submitted code (s) | Tokens through that submission | GPU-s before that submission |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 26.79 | 102.55 | 380,339 | 0 |
| Optional TrainCheck 1 | 38.01 | 111.12 | 334,186 | 0 |
| Control 2 | 34.35 | 99.39 | 304,915 | 0 |
| Optional TrainCheck 2 | 29.49 | 104.53 | 263,008 | 0 |

Localization is the first received visible statement with a sufficient causal argument, adjudicated after all agents froze. Static source evidence qualifies. All four statements explicitly derived shard order `0,2,1,3` being bound to model order `0,1,2,3`. No later runtime result was retroactively credited.

Times start at trial launch and are observed upper bounds: message receipt for localization, and CLI completion for the earliest submitted snapshot later passing every independent contract. Snapshot-copy timing is retained separately. These are not measurements of private thought or the exact instant of an edit. Native token counters are exact at completed-turn boundaries; localization occurred within turn 1, so its token count is only bracketed between zero and that turn’s total. We do not interpolate it.

### Tokens through the first valid submission

| Trial | Input | Cached input, included in input | Uncached input | Output, including reasoning |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 374,612 | 335,360 | 39,252 | 5,727 |
| Optional TrainCheck 1 | 328,810 | 278,784 | 50,026 | 5,376 |
| Control 2 | 299,253 | 251,904 | 47,349 | 5,662 |
| Optional TrainCheck 2 | 257,047 | 217,600 | 39,447 | 5,961 |

The milestone token total is input plus output across all internal model calls, not the length of the final answer. Cached input and reasoning output are subsets and are not added again. Native `turn.completed` counters accumulate across resumed sessions; each turn’s increment was independently recomputed from the raw JSONL. Fewer total tokens cannot be treated as a monetary saving without accounting for the cache distribution and actual pricing. No dollar-cost claim is made.

## Complete agent trial costs

Each agent used two decision turns and one remote diagnostic request. All requests completed successfully. Control 1, treatment 1, and control 2 each executed eight distributed optimizer updates. Treatment 2 executed reference eight plus candidate eight, matching its declared total of 16. Rank replicas are not counted as extra global updates.

| Trial | Raw wall (s) | Shared-service wait (s) | Wall excluding that wait (s) | CLI elapsed (s) | Allocated GPU-s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control 1 | 206.76 | 0.00 | 206.76 | 149.86 | 24.748 |
| Optional TrainCheck 1 | 245.48 | 48.28 | 197.20 | 146.41 | 24.330 |
| Control 2 | 189.36 | 0.00 | 189.36 | 130.85 | 24.901 |
| Optional TrainCheck 2 | 247.88 | 53.32 | 194.55 | 137.47 | 24.798 |

| Trial | Final input | Final cached input | Final uncached input | Final output |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 613,698 | 567,424 | 46,274 | 7,457 |
| Optional TrainCheck 1 | 566,261 | 508,288 | 57,973 | 6,868 |
| Control 2 | 461,659 | 410,240 | 51,419 | 6,894 |
| Optional TrainCheck 2 | 408,396 | 360,320 | 48,076 | 6,961 |

The shared-service lock serializes transport as well as execution. This wait is not a cluster scheduler measurement. Only the measured wait is subtracted; each agent’s own transport, setup, diagnostic work, and tool use remain counted. Reversing pair 2’s launch order did not reverse request arrival: the control reached the service first in both pairs. Raw wall times are retained to expose that asymmetry.

CLI elapsed includes local tools, startup, model/API waiting, and response generation. The service uploads frozen source and checkpoints for each request, so transfer costs are part of this harness. GPU-s is supervised worker lifetime multiplied by allocated devices, including initialization, state exports, and diagnostics; it is not active GPU utilization or pure training time. Standard audit snapshots execute equally in both arms.

Localization timing favored different arms in the two pairs. The optional-tool arm’s first valid submission was later in both pairs, while it consumed fewer input-plus-output tokens; uncached-token differences were mixed. Queue-adjusted final completion also favored different arms. Two pairs and zero actual TrainCheck uses do not establish a stable efficiency effect.

## Task and controls

- Ring18, two NVIDIA A16 devices (0 and 1), driver 535.309.01, Python 3.10.14, PyTorch 2.2.2+cu121, FP32. Shared-machine runs use isolated directories and the existing environment.
- A 10,080-parameter, four-layer Transformer: width 16, two attention heads, FF width 32, dropout 0.1, learned positional embeddings, vocabulary 32, sequence length eight. Circular previous-token sequence tagging requires cross-position information. SGD uses LR 0.15 and momentum 0.9; eight examples per rank, accumulation four, 128 tokens per distributed update.
- The previous backend saves complete per-rank snapshots. The current backend saves shared non-layer state plus rank-owned full layer bundles. Rank-major reconstruction loses logical-name association for middle layers while shapes still match. The defect is in the synthetic initial task fixture, not a newly discovered production TrainCheck defect.
- Public training seed 20261101, independent evaluation seed 777000; 120 prefix updates and eight shown continuation updates. Agents receive real curves, both implementations, and five existing checkpoint files. The prompt describes the storage change and expected continuation contract, without saying to find a bug or naming an invariant.
- Four fresh native CLI sessions, requested model `gpt-6-astra`, high effort, priority service, CLI 0.160.0. Same normal workspace-write policy, no approval bypass, no delegation, no previous chat context. Limits: 900 CLI seconds, six requests, at most 16 resumed updates and 180 seconds per two-GPU request. Arbitrary diagnostic update counts are instructional and source-audited; every actual count meets its declaration and cap.
- Controls retain ordinary logging, assertions, state helpers, and custom Python. Treatments additionally receive a generic observation manual and 31 instrumenter/support files. Twenty common packet files match byte-for-byte apart from the declared optional-capability prompt sentence; treatment packets contain 52 files. No inference engine or target checker is supplied.
- Standard requests return logs, metrics, and a terminal-state export; automatic restore-time audit snapshots remain private. Either arm can explicitly capture that same state in a diagnostic, under the same request and time budgets. Every agent chose a diagnostic. Private grading began only after all four final selections froze.

Packet read scope is an instruction, not an OS read-isolation claim. The independent audit examined all 47 completed commands and eight edit events; it found no observed external-context, delegation, or update-budget violation. No production typing/assertion contracts were weakened. Both treatments read the manual; treatment 1 also inspected instrumenter source after localization. Source and documentation reads alone do not count as TrainCheck activation: all trace-rank directories and archived traces were checked and contained zero trace files/records.

## Selection and calibration limits

All 12 original severity-screen configurations failed, and every source revision, phase result, and failed screen is retained. The selection was made after calibration and before any solver access:

| Version | Change | Prefix / continuation configurations | Severity screens passed |
| --- | --- | --- | ---: |
| v1 | Optimizer-state sharding, cloned layer initialization | 12/8, 8/8, 16/8 | 0/3 |
| v2 | Independent layer initialization | 12/8, 8/8, 16/8 | 0/3 |
| v3 | Full model-plus-optimizer layer bundles | 12/8, 8/8, 16/8 | 0/3 |
| v4 | Contextual target and positional embeddings | 32/8, 64/8, 120/8 | 0/3 |

V4 attempt 3 was selected as a learned checkpoint-contract continuation case. It was **not reclassified as passing the original severity screen**. Its original `qualified=false` report and two failed severity flags remain unchanged; a separate, hash-bound study-selection manifest records the changed study purpose. All 29 non-severity checks pass, including reference learning, exact reference restart versus uninterrupted training, exact known-correct restoration/continuation, and the original model/optimizer mismatch with other restored state intact.

At the restart boundary, validation loss was 0.665012464 for the reference and 0.679926716 for the original current pipeline (+2.24%, reference denominator). At update 128 it was 0.413439389 versus 0.439974099 (+6.42%, reference denominator). Intermediate curves cross. These observations support a restoration-contract case, not a large or uniformly worsening quality-regression claim.

## Independent validation and resource scope

The frozen evaluator compares every named model/optimizer tensor, dtype/shape, group contract, RNG state, progress/data cursor, rank/topology, and continued trajectory. It does not import candidate code. It validates frozen source and artifact manifests; restore state must be exact, while subsequent trajectories use the declared atol 2e-6 / rtol 2e-5. No record, tensor content, or meaningful identity is dropped to obtain parity.

All four distinct implementations pass the public checkpoint and new seeds 20261117/20261129 on two ranks, plus a fresh one-rank checkpoint at seed 20261117 (evaluation seed 777001). The one-rank case does not test changing the world size of an existing two-rank checkpoint. All 16 reports replay identically locally, with 25 successful GPU phases and unchanged frozen files. Both submitted turns of every agent map to passing snapshots. The original v4 implementation fails restoration and trajectory checks under the same evaluator; the independent known-correct control passes.

- 239 focused repository tests pass in 12.16 seconds; pinned configured hooks pass. The hook’s mypy scope skips these tools, so all 17 committed tool/test source files were explicitly checked with mypy 1.9 using `--explicit-package-bases --check-untyped-defs --disallow-untyped-defs --disallow-any-generics`.
- Operator tests separately cover artifact delivery, failure/counter accounting, the selection manifest, and milestone accounting (20 synthetic milestone tests). Native CLI initial/resume preflights verify cumulative usage. A real two-rank service preflight verifies output delivery and one complete generic TrainCheck event pair per rank, preserving 104 tensor pairs per rank plus scalar/container and RNG state in the earlier v3 setup. That preflight is not treatment-agent usage.

| Resource category | Allocated GPU-s |
| --- | ---: |
| Four agent trials | 98.777 |
| All calibration (72 phases) | 2010.000 |
| Two service preflight phases | 49.040 |
| Private grading (25 phases) | 724.607 |

Qualification, preflight, and private grading costs are excluded from agent efficiency totals and reported separately. Compression, transfer, local audits, and benchmark implementation work are not GPU training costs. This study does not estimate total researcher/engineering cost or a monetary bill.

## Reproduction and evidence

Committed workload/interfaces and measurement tools: [checkpoint pilot](../../tools/checkpoint_pilot/README.md). The executed application source is signed revision `900bd8e71015a51d6b2d092b72a6624170fa37e9`; final neutral packet documentation is signed revision `3cb6b5b2107eb6bb231ae8c16e8390cb3999cc1f`. Every candidate is identified by its full submitted-file manifest in `retrospective-grading/plan.json`.

Operator-only `protocol-final.json`, `study-selection.json`, `prelaunch-freeze.json`, `packet-manifests.json`, `environment.json`, and `usage-preflight.json` bind the setup. `trials/` retains prompts, raw CLI JSONL, receive timestamps, exact usage, every submission, request receipt, source/checkpoint hash, log, and diagnostic artifact. `milestone-annotations.json` binds independently adjudicated visible statements and passing-grade evidence; `milestone-summary.json`, `actual-accounting-audit.json`, `metrics.csv`, and `results-summary.json` contain the reported numbers.

Original packets are reconstructed byte-exact from immutable common sources and hash-verified inputs, with provenance recorded; live candidate edits are not substituted for initial source. All failed calibration versions and independent negative/positive controls remain in the archive. Empty trace directories are preserved to distinguish unactivated recording from missing artifacts.

For a new run, use fresh experiment paths in the archived machine-specific scripts. The following records the actual interpreter and host separation; it is not a command to overwrite this completed study.

On the Mac, in the experiment directory:

```sh
EXPERIMENT_ROOT=/Users/matt/gitrepos/traincheck/outputs/ring18-checkpoint-continuation-20261101
CHECKPOINT_LOCAL_PY=/Users/matt/miniconda3/envs/traincheck/bin/python
cd "$EXPERIMENT_ROOT"
"$CHECKPOINT_LOCAL_PY" prepare_packets.py --selection study-selection.json
MPLCONFIGDIR=/tmp/traincheck-checkpoint-mpl XDG_CACHE_HOME=/tmp/traincheck-checkpoint-cache /Users/matt/miniconda3/envs/acamate/bin/python plot_curves.py
"$CHECKPOINT_LOCAL_PY" study_driver.py
"$CHECKPOINT_LOCAL_PY" retrospective_grading.py prepare --selection study-selection.json --source "$EXPERIMENT_ROOT/qualification-source-v4/source" --protocol protocol-final.json
```

After uploading the generated grading directory, run scoring **on Ring18**:

```sh
REMOTE_GRADING_ROOT=/home/yuxuan/experiments/traincheck-checkpoint-20261101/retrospective-grading
CHECKPOINT_REMOTE_PY=/home/yuxuan/miniconda3/envs/DS-1801/bin/python
cd "$REMOTE_GRADING_ROOT"
"$CHECKPOINT_REMOTE_PY" retrospective_grading.py score --root "$REMOTE_GRADING_ROOT"
```

After retrieving the results, replay and summarize **on the Mac**:

```sh
EXPERIMENT_ROOT=/Users/matt/gitrepos/traincheck/outputs/ring18-checkpoint-continuation-20261101
LOCAL_GRADING_ROOT="$EXPERIMENT_ROOT/retrospective-grading"
CHECKPOINT_LOCAL_PY=/Users/matt/miniconda3/envs/traincheck/bin/python
cd "$EXPERIMENT_ROOT"
"$CHECKPOINT_LOCAL_PY" retrospective_grading.py audit --root "$LOCAL_GRADING_ROOT"
"$CHECKPOINT_LOCAL_PY" summarize_milestones.py --study-root "$EXPERIMENT_ROOT" --annotations milestone-annotations.json --output milestone-summary-replayed.json
```

The operator scripts are machine-specific and archived with exact invocation receipts. Preparation uses fresh directories and refuses to overwrite prior trials. Re-execution requires new experiment paths; restoring the archive and replaying its evaluator/accounting does not require another solver or training run.

Local evidence root: `/Users/matt/gitrepos/traincheck/outputs/ring18-checkpoint-continuation-20261101`.
Ring18 evidence root: `/home/yuxuan/experiments/traincheck-checkpoint-20261101`.

Sealed archive: `final-evidence-v2.tar.xz`, 10,011,932 bytes, SHA-256 `829aa1424f9b1c05ccbabcb4e275411adb8d54db48b22c4e302202eb33d76b1b`. All 3,113 evidence files plus the manifest and 1,480 directories were reopened and verified. The archive and its manifest were copied to the Ring18 evidence root and their remote hashes independently verified.

## Interpretation

The complete experiment works, and the agents preserve the distributed continuation contract. The case still has little need for dynamic observation: its decisive logical-name association is visible in a short source fragment, and checkpoints already expose the relevant state. A useful next validation should involve a realistic cross-library or cross-rank interaction whose cause requires runtime state to establish, while retaining ordinary diagnostics and a neutral prompt. These two pairs do not justify further tracing-performance investment on their own.
