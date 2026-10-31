# Ring18 two-GPU training and tracing qualification

Thirteen bounded two-GPU runs completed on 2026-10-31. DDP training and
checkpoint interchange work for this small workload, but the learned API rules
did not produce a fault-specific alarm for incorrect distributed loss
normalization. Active parameter sampling also exposes stale post-update
timestamps. This qualifies infrastructure and identifies gaps for #165; it is
not an agent-benefit experiment or an official RSI result.

## Workload and provenance

- Ring18, two NVIDIA A16 GPUs (indices 0 and 1), driver 535.309.01. All eight
  GPUs were idle at discovery and after completion; other devices were unused.
- Python 3.10.14, PyTorch 2.2.2+cu121, NumPy 1.26.0, pandas 2.2.2,
  polars 1.7.1, numba 0.60.0, orjson 3.10.18. Existing environment reused;
  source and outputs live in an isolated experiment directory.
- TrainCheck base `754277ae87e2189cebf16b405940bef1fee5904c` (PR #164);
  workload/tracing commit `15ed99cc2d72853ebeb44a665b3a91d8923dd99e`;
  complete-state comparator `ebd66a5f7062952ee0f059f2df8a61b2dd2a2575`.
  Both commits are signed. Runtime workload, tracing and core source hashes
  match across all runs; comparator/test additions between phases are recorded
  in each invocation's source manifest.
- A 5,504-parameter, two-layer Transformer, width 16, two attention heads,
  vocabulary 32, sequence length 8, dropout zero. Synthetic token targets are
  `(token + 1) % 32`; this is a numerical contract probe, not a model-quality task.
- DDP/NCCL, two microbatches of two sequences per rank per optimizer update,
  four updates, AdamW, clipping and StepLR. Each update contains 10 valid target
  tokens on rank 0 and 22 on rank 1. Full accumulation windows only.
- Seeds, deterministic algorithms, disabled TF32 and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` are fixed. FP32 and BF16 autocast are tested.
  BF16 uses no GradScaler; FP16 overflow behavior is outside scope.

## Independent correctness and checkpoint results

The independent first-update oracle concatenates the logical global batch and
computes a token-mean gradient in a separate model. Correct DDP accumulation
scales each local loss sum by `world_size / global_valid_tokens`. The injected
fault instead divides by each rank's local token count before DDP averaging.

| Configuration | Maximum absolute gradient error, both ranks | Oracle outcome |
| --- | ---: | --- |
| Healthy FP32, plain/API/subclass | 4.47035e-8 | Pass (`atol=2e-6`, `rtol=2e-4`) |
| Faulty FP32, plain/API | 0.0695832 | Fail with the same declared tolerance |
| Healthy BF16, plain/subclass | 0.000564575 | Pass (`atol=0.003`, `rtol=0.03`) |

The faulty model's two ranks still agree exactly. Rank agreement alone therefore
does not establish correct global loss normalization. The fault also changes the
final model relative to the healthy run.

All nine complete checkpoint comparisons pass:

| Left | Right |
| --- | --- |
| Plain FP32 continuous | API FP32 continuous |
| Plain FP32 continuous | Subclass FP32 continuous |
| Plain FP32 continuous | Plain save at update 2, plain resume to update 4 |
| Plain FP32 continuous | Plain save at update 2, subclass resume to update 4 |
| Plain FP32 continuous | Subclass save at update 2, plain resume to update 4 |
| Plain FP32 save at update 2 | Subclass FP32 save at update 2 |
| Plain BF16 continuous | Subclass BF16 continuous |
| API FP32 healthy reference | Fresh API FP32 healthy control |
| Plain FP32 fault | API FP32 fault |

Each comparison covers both ranks: 274 tensor pairs, 112 scalar pairs and
78 container pairs. Model, stored gradients, optimizer, scheduler, CPU/CUDA RNG
and configuration fields are compared without excluded fields or normalization.
All value bits and whole checkpoint-file hashes match. Storage aliasing and
original device placement are not reconstructed by the artifact comparison.

Resume uses fresh processes and the same world size. Checkpoints are taken at
complete update boundaries; stored gradients are evidence, and the next update
clears gradients before use. Data is regenerated from absolute step/rank indices,
so DataLoader cursor recovery is not tested. RNG is saved/restored, but dropout
is zero and these runs do not exercise stochastic model operations.

## Trace coverage and timestamp defect

API and subclass modes use the normal API instrumenter. Subclass tracking is
attached before DDP and retains parameter identity; model parameters are sampled
explicitly before and after each update. Proxy argument scanning is enabled in
subclass mode; API mode disables it. Standard tensor hashing is enabled in both
instrumented modes. Bookkeeping, reference calculations, checkpoint I/O
and rank-consistency measurements are suppressed from user tracing. Optimizer
state is not registered, and no cross-rank TrainCheck checker is installed.

Per rank, a four-update FP32 run has 1,876 API pre/post pairs in both instrumented
modes. Subclass mode additionally has 297 variable records across 27 parameter
names, including 108 before/after sample pairs. No duplicate API event keys were
observed. These counts are record audits, not a proof that every desired
application event was captured.

Every post-update parameter sample in both full FP32 and BF16 subclass runs
carries a timestamp **earlier than that update's `AdamW.step` pre-event**, despite
a changed recorded data hash. At the first executed update, all 27 parameters
also have identical before/after timestamps on each rank. Subsequent pairs need
not have equal timestamps, but their post-update timestamps are still stale.

Source inspection explains the observation: `ProxyParameter.dump_trace` uses
`last_update_timestamp`; explicit registry sampling does not refresh it, and
in-place optimizer mutation does not necessarily pass through the Python
assignment hook. This prevents treating these records as reliable evidence of
the temporal relationship between an optimizer call and its updates. No timestamp
rewrite or core fix was applied to make the experiment pass. Existing hashes
cover the instrumenter's first-row semantics, not a full tensor-content proof;
full-state equality above comes from the independent checkpoint comparison.

## Learned diagnostics on a fresh healthy run and the fault

Learn rules separately from each rank's healthy API reference. Use pandas and
one CPU worker; disable `FunctionLeadRelation` and `FunctionCoverRelation`, as in
the established pilot workflow. Freeze these rule files, then run the real
checker with coverage observation on the fresh healthy control and the fault.
Do not concatenate ranks and call the result a cross-rank check.

| Rank | Learned / active rules in each check | Healthy violations | Fault violations | Fault-only violated rules |
| --- | ---: | ---: | ---: | ---: |
| 0 | 592 / 592 | 2 | 1 | 0 |
| 1 | 593 / 593 | 2 | 2 | 0 |

All four checks complete with no dormant rules, unobserved outcomes or checker
errors. Activation does not imply all possible comparisons ran: individual
checkers can stop at their first violation. No model-variable rules are checked
in this API-only inference comparison.

The shared alarm is `DistributedDataParallel.forward()` containing
`Embedding.forward()`. Rank 0's healthy-only alarm is a distinct-arguments rule
for `LayerNorm.forward()`; rank 1's healthy and faulty runs both violate a
distinct-arguments rule for `TransformerEncoderLayer.forward()`. The full rule
identities and failing evidence are retained; no alarms were removed. Counts of
identity preconditions alone do not establish that relation parameters are
portable across processes.

Thus this configuration gives no fault-specific diagnostic for the independently
confirmed normalization error. Fewer alarms in rank 0's faulty run do not mean
better training. This is one fault and one fresh healthy control, not an estimate
of general detection accuracy.

## Cost, validation and remaining work

The thirteen supervised workload phases total 211.29 seconds of wall time.
For context, maximum per-rank loop durations were 0.348 s plain FP32, 3.549 s
API FP32 reference, 4.513 s API FP32 healthy repeat, and 5.753 s subclass FP32.
Plain/subclass BF16 took 0.345/4.739 s. These short runs include oracle snapshots,
first-use costs and, for subclass mode, hashing. They are operational observations,
not repeated steady-state overhead estimates. CPU inference overlapped part of
the later matrix. Full FP32 API/subclass traces total 4,946,510/5,595,602 bytes
across both ranks, excluding instrumentation logs and checkpoints.

- 28 focused tests pass locally on Python 3.10/Torch 2.2.2 and on Ring18 with
  CUDA visible. Configured hooks and explicit strict mypy over seven new
  production/test modules pass; the normal mypy hook does not cover `tools/`.
- An initial local sandbox attempt blocked OpenMP shared memory; the same
  composed suite passed with the required access. This was not bypassed by
  skipping the tracing tests.
- With `CUDA_VISIBLE_DEVICES` empty on Ring18's CUDA-enabled Torch build, the
  suite has 26 passes and two tracer-startup failures. Metadata inspection of
  `torch.backends.cuda` calls `current_device()` and attempts CUDA initialization.
  This additional compatibility failure is preserved; exposing a GPU permits
  all 28 tests to pass but does not fix the hidden-GPU case.
- The reused supervisor measures the launcher's process group; torchrun workers
  use separate sessions. Its RSS figure/8 GiB setting is not a complete worker
  memory bound, and cleanup relies on torchrun signal forwarding. Its trace-size
  limit covers API files only. Final process/GPU audit found no remaining study
  processes and all GPUs idle. Future unattended trials need stronger tree-wide
  supervision before claiming complete memory/cleanup enforcement.

Next gates for #165 are to repair and validate parameter observation timestamps,
establish relevant state/normalization diagnostic coverage, and qualify FSDP and
the selected feature tasks. Partial windows, FP16 overflow, stochastic resume,
different world sizes and natural agent implementations remain untested. Do not
start the paired agent study on the assumption that these results cover them.

## Reproduction and artifacts

Use `python -m torch.distributed.run --standalone --nproc-per-node=2
tools/distributed_probe/workload.py --output RUN --steps 4 --device cuda
--instrumentation plain` with the fixed environment above. Choose `api` or
`subclass`, `--fault local-mean`, or `--precision bf16` for the named variants.
Save with `--stop-after 2`, then use `--resume SAVE_RUN` in a fresh run. Compare
with `python tools/distributed_probe/compare.py --left LEFT --right RIGHT
--output comparison.json`. Invocation manifests retain the exact commands,
runtime source hashes and supervisor settings for every phase.

Remote root: `/home/yuxuan/experiments/traincheck-distributed-20261031` on Ring18.
Local evidence root: `/Users/matt/gitrepos/traincheck/outputs/ring18-distributed-20261031`.
The downloaded `evidence.tar.gz` contains 188 verified files, including complete
checkpoints, variable traces, rule/checker outputs, phase logs and source hashes.
Large API traces remain under the remote `runs/*/payload/rank*/trace` directories;
their hashes and sizes are in `results.json`.

Archive SHA256: `ba267cddd70abb54b7aae8a7ead085e6bb3f4fbe91771475d703659841425922`.
`evidence-bundle/evidence/diagnostics.json` records alarm identities;
`results.json` records checkpoint comparisons and raw timestamp evidence;
`retrieval-verification.json` records archive verification and the final 160
Python source-file hash matches. No traces, checkpoints or machine-specific
operator scripts are committed to Git.
