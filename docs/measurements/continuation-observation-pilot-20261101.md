# Training continuation with optional TrainCheck observations

## Question and result

Can an agent use TrainCheck as an optional observation tool while continuing a
training task, without being told to find a bug or being supplied a target
invariant? This development pilot exercises that workflow with two fresh agents:
ordinary development tools versus the same tools plus TrainCheck instrumentation.
The inference engine is outside the experiment.

Both agents independently diagnosed the same extra gradient-normalization
factor from source inspection and selected the same one-line revision at the
original learning rate. The treatment agent subsequently used a native
TrainCheck probe and its own Python checker to corroborate that diagnosis. This
establishes a working observation workflow; it does **not** demonstrate an agent
quality, discovery, speed, or cost advantage from TrainCheck.

This is one calibrated synthetic development case, not an official RSI benchmark
result, a blinded experiment, or an estimate of success rates across tasks.

## Neutral task and experimental boundary

Both agents received byte-identical copies of 11 common files: the previous and
current pipeline, shared runner, actual training/validation curves, run profile,
engineering guidance, and [TASK.md](../../tools/continuation_pilot/TASK.md). The
task says that a change to microbatch execution reduced training quality and
asks the agent to decide what to investigate or try next. It permits learning
rate changes, diagnostics, and rerunning either version while preserving the
effective batch, intended mean-token objective, and public interfaces. It does
not say that the implementation is faulty or supply a diagnosis or invariant.

The treatment additionally received [TRAINCHECK.md](../../tools/continuation_pilot/TRAINCHECK.md)
and 32 instrumenter/support Python files. The guide explains selected callables,
argument fields, explicit probes for local state, native event identities, and
arbitrary offline checkers. It makes no unmeasured low-overhead promise. There is
no inferred rule, targeted normalization checker, or inference engine in either
packet. Prior conversations, operator tests/protocol, other arm files, and private
evaluation were excluded by instructions. This was not an OS access boundary.

Both agents used the same inherited model/reasoning configuration with no parent
history. Exact operator dispatch messages are saved in each packet. Each arm
could request four two-GPU runs of at most 24 optimizer updates, including failed
runs, and perform tiny CPU operation checks. The 15-minute reasoning target was
advisory and excluded operator queue/setup time; tokens and actual reasoning
time were not independently measured or enforced.

## Workload and calibration

Ring18, two NVIDIA A16 GPUs, Python 3.10.14, PyTorch 2.2.2+cu121, FP32. The model
is a dropout-free, 5,504-parameter Transformer: width 16, two layers, two attention
heads, feed-forward width 32, vocabulary 32, sequence length 8. Synthetic targets
are `(token + 1) % 32`. SGD uses learning rate 0.15 and momentum 0.9. There is no
gradient clipping. This is a new small development workload, not the earlier
6.2M-parameter TorchTitan Llama inference trace.

Each rank generates its complete eight-example batch before splitting it into
microbatches. Accumulation changes therefore retain the same data and 128 global
tokens per optimizer update. All tokens are valid. The reference concatenates
microbatches and performs one full-batch backward pass; the current pipeline
uses DDP `no_sync()` except on the last microbatch. Training and independent
validation curves report unscaled mean-token cross entropy, not the scalar
passed to backward. Evaluation preserves RNG, model state, and module modes.

The operator-known initial fixture divides by accumulation in both `current.py`
and the shared `backward` helper. SGD was chosen because adaptive optimizers can
cancel uniform gradient scaling. This information was not in the agent prompt.
The committed `training_task/current.py` deliberately retains the initial task;
the agents' selected revisions are frozen separately in the evidence archive.

Calibration is part of the evidence:

- A first qualification launch failed before training because the supervisor
  hides CUDA by default. The corrected command explicitly supplies
  `CUDA_VISIBLE_DEVICES=0,1` to its child. The failure log is retained.
- The first completed qualification used 64 updates. Both versions nearly
  converged: the final-three-validation-point means differed by 0.0452539 nats,
  and the current/reference improvement ratio was 0.987573. This failed the
  preselected gap threshold of 0.05 and improvement-ratio threshold of 0.8.
- Before launching either agent, the development budget was changed to 24
  updates using these calibration curves. A fresh qualification passed the
  same criteria: final-three means 0.663464 versus 2.142884 nats, ratio 0.503334.
  Final validation loss was 0.00896090380 for reference and 1.18585254252 for
  current. All attempts are retained; this is not a first-attempt preregistered
  success. No held-out seed or solver outcome was used to select the task.

Protocol revision 2 SHA-256:
`0a71f6e452fee57862bb9a8cfa2b1d685efe5414a7dd22b75b1e36f12e073861`.
Implementation/evaluator revision: `8d511938eb5940cbc02023b9096df317768ae1bc`.
The workload was first committed as `13550b75cdc5ee14d1556a27e516ddb191ebc257`;
the runtime workload files are identical across those revisions.

## Agent decisions and development evidence

Both agents removed the same extra division, retaining LR 0.15:

```diff
- backward(loss / accumulation, accumulation)
+ backward(loss, accumulation)
```

The control agent wrote ordinary autograd regressions on a tiny embedding
operation. The treatment wrote a tiny CPU `FixedLogits` diagnostic, inserted a
`loss_input_probe` immediately before the shared backward helper, wrapped that
probe with TrainCheck, and parsed the native trace with its own checker. It did
not wrap native backward execution or instrument the GPU training runs.

Both CPU diagnostics measured gradient ratios 1, 1/2, 1/4, and 1/8 before the
revision for accumulation 1, 2, 4, and 8, then matching gradients after it. The
treatment's two native traces each contain 15 unique, complete pre/post probe
pairs. Its observed and unobserved post-revision diagnostic JSON outputs are
byte-identical. That covers the selected probe, not every backward event or
arbitrary training state.

Development seed 20261101, evaluation seed 777000, 24 updates per run:

| Arm | Run | Accumulation | Final validation loss |
| --- | --- | ---: | ---: |
| Control | selected current | 4 | 0.00896090368042 |
| Control | reference rerun | 4 | 0.00896090379683 |
| Control | selected current | 8 | 0.00896090053720 |
| Control | selected current | 1 | 0.00896090379683 |
| TrainCheck | selected current | 4 | 0.00896090368042 |
| TrainCheck | reference rerun | 4 | 0.00896090379683 |
| TrainCheck | selected current | 8 | 0.00896090053720 |

All seven requests completed, with unchanged snapshotted sources. Each agent's
complete checkpoint comparison against its reference rerun found maximum model
and momentum differences of 2.38419e-7 and 5.96046e-8, respectively, for
accumulation 4/8. CPU/CUDA RNG states and optimizer parameter groups matched
exactly. Control's accumulation-1 tensors/curves match reference exactly;
accumulation metadata differences are recorded explicitly. These are numerical
comparisons of every saved tensor, not a claim that all checkpoint files match
byte-for-byte.

## Frozen independent evaluation

Both final selections and sources were frozen at **2026-11-01 10:57:36 UTC**, before
scoring. All immutable packet files were rechecked. The 117-file freeze manifest
has SHA-256 `0fdf1f84cb14e15e0b97692b3a6877a2bfdda253ba212f0908eaaec7837662b4`.
Selected `current.py` hashes:

- Control: `c5d452fbfbd94dbd85a601e6e6d453a1dd59e34bb2cf2f59b6fc78f5db4db62e`.
- TrainCheck: `f3a29fdcae9fd25ecc3de21478b61d92d8ec0e5d810092ec585a811665d25bb9`.

The files differ in their explanatory comment, not the executed operations.
The private evaluator first computes frozen reference outcomes, then imports the
selected candidate in a fresh process. Quality and contract outcomes are
reported separately. Held-out quality uses training seeds 20261117/20261129,
evaluation seed 777001, 24 updates, accumulation 4, and two GPUs. Candidate order
is reversed for the second seed. Independent CPU contract scenarios cover both
seeds and accumulation 1/2/4, comparing all parameters, gradients at the actual
optimizer step, momentum, RNG, scalar/container structure, and the microbatch
bound. FP32 tolerance is `atol=2e-6, rtol=2e-4`; exact content hashes are retained.

All eight GPU runs completed with frozen sources unchanged and matching rank
curves. Final independent validation losses:

| Pipeline | Seed 20261117 | Seed 20261129 | Mean |
| --- | ---: | ---: | ---: |
| Reference | 0.011329145054 | 0.008691648662 | 0.010010396858 |
| Original current | 1.258648961782 | 1.070365890861 | 1.164507426322 |
| Control selection | 0.011329147383 | 0.008691646799 | 0.010010397091 |
| TrainCheck selection | 0.011329147383 | 0.008691646799 | 0.010010397091 |

Both selected candidates pass all six CPU scenarios, each comparing 82 tensor
pairs and scalar/container structure, with one optimizer step and the required
microbatch bound. The original current passes accumulation 1 but fails
accumulation 2/4 on both seeds, with 81 tensor mismatches per failing scenario.
The checker thus executed and rejected a known negative, rather than merely
reporting no violations.

An additional independent comparison of complete held-out GPU checkpoints
passes for reference/control, reference/treatment, and control/treatment across
both seeds and ranks: **12 checkpoint pairs, 672 tensor pairs**, without removing
any saved field. All control/treatment tensor hashes match exactly, along with
all scalar/structure checks. Reference/candidate comparisons use the stated FP32
tolerance. The two selected solutions have identical held-out curves; this is a
tie for this paired task, not a general equivalence result.

## Costs and limitations

Control used 4/4 training requests and treatment used 3/4. Their summed client
request times were 260.321 s and 205.142 s, including 79.021 s and 70.987 s of
queueing. These are operational receipts, not causal efficiency estimates:
control requested an extra boundary configuration, early uploads included type
checker caches, and setup/queue conditions differed. No GPU training request
retained TrainCheck instrumentation, and CPU probe overhead was not measured.
No tracing slowdown or low-overhead result follows from this experiment.

This simple case exposes the duplicate divisor directly in readable source;
both agents reached the diagnosis before the runtime confirmation. Two held-out
seeds validate the selected implementations on this task, but do not create two
independent agent trials. The CPU arithmetic check does not certify distributed
collectives, long-run stability, or checkpoint resumption. The GPU runs exercise
standard two-rank DDP only. More realistic continuation tasks should require
cross-component state observations before investing in a larger benchmark.

## Validation and reproduction

The 40 focused tests pass locally (1.29 s) and on Ring18 (3.58 s). They cover
deterministic batches, state-preserving unscaled evaluation, malformed run
options, the known negative, a correct revision, learning-rate compensation,
full-batch escape, multiple optimizer steps, exceptions, missing/invalid
interfaces, frozen-reference isolation, import-time tracing setup, and source
mutation. Pinned pre-commit checks pass. Explicit strict mypy covers eight
Python modules, including the tools/tests that the configured core-only mypy
hook does not select; tests also passed explicit pinned lint/format checks.
These checks concern the committed harness/evaluator. The archived one-off
operator transport scripts were not included in that static-check claim. No
remote CI result is claimed.

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_continuation_workload.py tests/test_continuation_evaluator.py
mypy --explicit-package-bases --check-untyped-defs --disallow-untyped-defs \
  --disallow-any-generics --ignore-missing-imports --follow-imports=silent \
  tools/continuation_pilot tests/test_continuation_workload.py \
  tests/test_continuation_evaluator.py
```

Private evidence roots:

- Local: `/Users/matt/gitrepos/traincheck/outputs/ring18-continuation-20261101`.
- Ring18: `/home/yuxuan/experiments/traincheck-continuation-20261101`.

Archive in both roots: `continuation-pilot-evidence.tar.gz`, 19,997,742 bytes,
442 individually verified evidence files plus the manifest. SHA-256:
`9f72d6e42c78e5ceb9f12439dadfaac0b5c0a21e3185a5f8aad2c7c5d7cc5d79`.
The archive was reopened and every member's size and content hash verified.

The evidence includes exact prompts, packet hashes, all qualification attempts,
requests, original input archives, raw native probe traces, metrics, complete
checkpoints, frozen selections, and independent evaluation reports. Large
artifacts and machine-specific transport scripts remain outside Git. Inspect
`independent-results.json`, `frozen/freeze.json`, `heldout/contracts/`, each
`phase-result.json`, and the per-rank metrics/checkpoints. Individual commands
and environment are preserved in `invocation.json`. The initial SSH launch
wrapper timed out while launching held-out scoring; the detached process
continued and completed. It was not resubmitted.

To repeat the fixed candidate quality check after restoring the archive, use
the same Python/Torch environment, reserve two GPUs, set a new output directory,
and run from a frozen candidate directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m training_task.engine --pipeline current --output /new/output \
  --steps 24 --accumulation 4 --seed 20261117 --eval-seed 777001 \
  --lr 0.15 --device cuda
```

For the separate CPU arithmetic check, run from the committed repository (or
the archive's `frozen/operator` directory), in a fresh process per candidate:

```bash
python -m tools.continuation_pilot.evaluate_candidate \
  --candidate /restored/frozen/control/training_task/current.py \
  --lr 0.15 --output /existing/report-directory/control.json
```

Exit status 0 means pass, 1 means a contract violation, and 2 means an evaluation
error with an unknown outcome. This reproduces the implementation check, not a
new independent agent trial. Repeating the agent experiment requires fresh
agents and freshly prepared neutral packets; never expose the operator protocol,
this report, or selected solutions in those packets.
