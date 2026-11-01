# Real cross-rank runtime case: agent-efficiency protocol

**Status: completed four-trial pilot, 2026-11-02.** All four agents identified the
clipping cause from source before their first GPU request. Each first submitted
a valid repair in turn 2, later passing all four private scenarios. No TrainCheck
instrumentation activation was observed in either treatment. This is a source-localizable
case and does not establish a TrainCheck efficiency benefit. Transport failures
and same-thread recoveries remain in the raw evidence; timing below reports both
raw and adjusted values. This is operator documentation, excluded from solver
packets.

The first two-update run had finite losses of 10.39899 and 10.62776. Among 27
replicated trainable parameters, cross-rank mismatches increased from zero before
the first update to 6 after update 1 and 18 after update 2. The corrected control
passed 630 model, 774 optimizer, 42 RNG and 162 replicated-parameter checks against
itself. Comparing the vulnerable run against that control failed 140 checks,
with maximum absolute model difference 0.00067138671875. These are qualification
results, not measurements of agent performance.

Removing inherited research annotations and isolating ordinary imports preserved
all archived state exactly: each vulnerable/corrected comparison covered six
rank/step snapshots, 384 tensors, 144,752,448 tensor bytes, 378 scalar values and
606 ordered dictionary keys, with no differences. Recorded CPU/CUDA/Megatron RNG
and optimizer values are included; unrecorded state and live object aliasing are
outside that comparison. Evidence: `private-state-parity.json` in the local
experiment directory below.

Both ordinary eight-update runs complete with finite losses and zero skipped or
NaN iterations. Final FP16/BF16 losses are 10.25196/10.25426. This short difference
does not establish a quality regression. The neutral task presented measured
curves without asserting one. An issue-named installation-path prefix in exported
text is replaced with `/runtime/python-env`, identically for both arms; raw logs,
exact mappings and original/public hashes remain operator evidence. Error/warning
wording, source filenames, line numbers and numeric values are preserved.

The private clipping observer records actual FP32 gradient selection and values
around native clipping, using the pipeline grid's slice-parallel rank. It is never
automatically provided to solvers. Its 26 CPU regressions cover call/exception
preservation, RNG and model state, flattened gradient views, real grid rank APIs,
missing/malformed metadata and duplicate evidence. Together with evaluator and
control-import tests, **91 tests pass**. Pinned hooks and explicit strict mypy
pass for the observer and tests; the repository mypy hook alone skips tools.

Setup failures are retained, including one incorrect private grid API and native
shutdown crashes after successful updates. The worker now explicitly releases
distributed process groups after application exit. A successful rerun alone does
not prove elimination of an intermittent shutdown failure; later receipts and
failure counts remain part of the results.

The development base is `codex/checkpoint-continuation-costs`,
[PR #170](https://github.com/OrderLab/TrainCheck/pull/170), at
`c79218c960a823f954bf49dad354975da39e6f24`. Subsequent source revisions, workload
configuration, packet manifests and commands were frozen before solver exposure.
The final evidence references are recorded below.

## Question and case selection

The question is whether optional access to TrainCheck helps an agent investigate
and repair a real distributed training problem more efficiently under a neutral
engineering task. The comparison measures causal localization, independently
valid code, agent-visible runtime verification and final completion separately.

The selected candidate is the historical DeepSpeed BF16 optimizer failure commonly
identified as **DS-1801** in the TrainCheck paper. Tensor-parallel training includes
both partitioned and replicated parameters. The affected implementation excludes
replicated parameters on nonzero TP ranks from clipping, although excluding their
duplicate contributions is appropriate when calculating the global norm. Thus
replicated parameters can receive different updates across TP ranks while the
training process continues.

| Candidate considered | Reason to consider it | Selection limitation |
| --- | --- | --- |
| DS-1801, BF16 clipping under tensor parallelism | Real silent training defect; combines rank, replication metadata, clipping activation and optimizer state; an existing Megatron–DeepSpeed environment and CodeParrot data are available on Ring18. | The repair is small once the relevant framework path is identified. Source inspection or prior knowledge can still succeed. |
| [PyTorch #104336](https://github.com/pytorch/pytorch/issues/104336), DDP device transfers | Real failure of gradient synchronization after moving an already wrapped model between CPU and GPU. | The device-transfer sequence is a recognizable source-level clue; it may repeat the weakness of the earlier pilot. |
| [DeepSpeed #5794](https://github.com/deepspeedai/DeepSpeed/issues/5794), MoE communication | Rank-local routing and collective arguments provide relevant runtime evidence. | The available artifact uses an inference workload with additional dependencies and a hang; it is less direct for this silent-training study. |

DS-1801 is selected for its real provenance, training relevance and available
environment, not because a preliminary agent failed to solve it. All subsequent
qualification attempts and failures must be retained. This study does not claim
that cross-rank observation is logically necessary for every successful diagnosis.
Source-based successes remain valid results.

## Historical and current source provenance

The upstream repository is now `deepspeedai/DeepSpeed`.
[PR #1801](https://github.com/deepspeedai/DeepSpeed/pull/1801), titled
“bf16+pipeline parallelism,” is merged. The following sources were inspected:

- The [vulnerable parent
  `a4cbf0c1d300d78c6124656458115d1050869142`](https://github.com/deepspeedai/DeepSpeed/blob/a4cbf0c1d300d78c6124656458115d1050869142/deepspeed/runtime/bf16_optimizer.py#L298)
  applies the nonzero-TP-rank filtering outside `if not for_clipping`.
- The [fix commit
  `e24814a10de04ce280efe2adb027b023e3336493`](https://github.com/deepspeedai/DeepSpeed/commit/e24814a10de04ce280efe2adb027b023e3336493)
  moves the condition inside that branch. Its literal intermediate patch uses
  `p` instead of `lp`; this commit alone is not the executable positive control.
- The [final merged source
  `56c522386878df2513ccb548de76b50191ae3af7`](https://github.com/deepspeedai/DeepSpeed/blob/56c522386878df2513ccb548de76b50191ae3af7/deepspeed/runtime/bf16_optimizer.py#L406)
  has the corrected branch and `lp` reference.

**Correction to an older discovery note:** the live Ring18 source inspection for
this study found that DeepSpeed HEAD
`32d97976ce08f00228ceb55a2d93005bd8b2769a` has the correct nested condition in
`bf16_optimizer.py`. The working-tree diff moves that condition outside
`if not for_clipping`, making the inspected dirty working tree vulnerable. The
older `outputs/trace-bug-impact/checkpoint-discovery.md` described this edit as a
repair; that characterization must not be used as provenance for this study.

The saved live diff is
`/Users/matt/gitrepos/traincheck/outputs/ring18-runtime-20261102/original-bf16-working.diff`,
SHA-256 `c04efe0f7f59bbcd973a6289a57b40fa4f832a7855b1b4ada9ebb73e3fcde8ad`.
Its two changed lines move the identical condition and `continue` out by one
indentation level. This artifact was read and hashed locally; the complete
source snapshots were subsequently frozen in the qualification and packet
manifests.

The candidate is an isolated snapshot of the inspected working-tree source.
The positive control uses the corresponding correct HEAD version of
`bf16_optimizer.py` while preserving the same remaining compatibility changes.
The evidence bundle retains the exact source diff and per-file hashes,
including the check that the control differs only in the intended logic. This is a
historical semantic reproduction on a compatible environment, not execution of
an untouched historical checkout. Shared repositories and installed environments
are inputs and must remain unchanged.

## Workload and qualification

The frozen native Megatron–DeepSpeed GPT workload uses CodeParrot, four layers,
hidden width 64, eight attention heads, sequence length 128, TP2, PP1, DP1,
BF16 and ZeRO stage 0. Micro/global batch sizes are both 1. Apex FusedAdam uses
learning rate 0.0005, beta2 0.999, weight decay 0.1 and clipping threshold 1.0.
Seed 1234 is public; 5678 is the additional validation seed. Ordinary public runs
contain eight updates. Private grading uses two updates, capturing steps 0–2.

Named parameters contain 1,722,880 elements per TP rank, including 9,856 replicated
elements. Counting each partition once and replicated parameters once gives
**3,435,904 logical parameter elements**. This is the GPT case used here, not
the earlier TorchTitan tiny-Llama trace. Ring18 uses NVIDIA A16 GPUs, Python
3.10.14 and Torch 2.2.2+cu121. GPU idleness is checked before each phase.

Qualification confirms the cause: clipping selects 27/27 replicated gradients on
rank 0 and 0/27 on rank 1 in the vulnerable implementation. The first-step maximum
cross-rank gradient gap grows from exactly zero before clipping to
0.7917701378 after clipping; the corrected control retains zero gaps. Clipping-
disabled and TP1 candidate/control states match exactly. The fresh seed reproduces
the discrepancy. Selective TrainCheck preflight records two real clipping calls
per rank with complete entry/return pairs, no exceptions and exact recorded
model/optimizer/RNG parity. It observes scalar norm arguments, not tensor/model
tracking. Full evidence is in `qualification-summary.json`.

Source-integrity checks cover all 271 framework/application Python files, all
139 cached native sources and four binaries. All remain unchanged. Private
snapshots stay on Ring18; compact reports, hashes and ordinary outputs are local.

The prespecified qualification gates were:

1. The candidate initializes successfully and executes finite training updates.
   The measured gradient norm actually activates clipping; configured clipping
   without an active event does not qualify.
2. Replicated trainable parameters agree at initialization and diverge after an
   affected update. Compare logical parameter identity within the relevant TP
   group, not the complete partitioned `state_dict` across all ranks.
3. The isolated corrected control preserves replicated-parameter agreement and
   completes the same training computation. Capture full values, shapes and
   dtypes, with any numerical tolerance declared before grading.
4. Pre/post-clipping FP32 gradient observations explain the differing updates.
   BF16 parameter rounding can mask an early difference; missing or unchanged
   low-precision observations are not proof that clipping was equivalent.
5. Clipping-disabled and TP1 controls separate the triggering conditions from
   unrelated startup or precision problems. Fresh seeds, and an additional
   TP2×DP2 configuration if feasible, provide independent validation.
6. An instrumentation preflight preserves the relevant model, optimizer and RNG
   behavior and establishes actual observation coverage. This preflight is
   operator work, not solver use of TrainCheck.

State observations must identify the run, global update, rank/group membership
and logical parameter. Wall-clock timestamp ordering across processes alone is
not sufficient alignment. Setup errors, nonactivating conditions, absent
observations, failed comparisons and successful comparisons remain separate
outcomes. Qualification and preflight GPU costs are recorded separately from
solver costs.

## Neutral solver task and matched access

The intended engineering context is an FP16-to-BF16 extension of the existing
tensor-parallel pretraining pipeline. The core prompt is:

> We are extending this language-model pretraining pipeline to BF16 while
> retaining tensor parallelism and the existing training configuration. The
> supplied outputs show the current short training run. Assess whether this
> implementation is ready for a longer run, decide what to investigate next,
> and implement any justified revision. Explain your conclusion using the
> evidence you collected.

Supply measured ordinary training outputs, configuration and relevant full
application/framework source. If curves show no material quality difference,
do not describe a quality regression that was not observed. Do not include this
protocol, issue identities, Git history, the private corrected source, a targeted
checker, or bug/fix annotations in solver packets. Preserve ordinary source names
and behavior; do not obscure code to make static reasoning artificially harder.
The exact allowed revision scope and preserved training contracts must be stated
and frozen before launch.

Two fresh matched pairs ran, with launch order reversed in the second pair.
Both used gpt-6-astra/high, the same tool interface and runtime, six remote
requests, 1–16 updates per request, 240 seconds per worker phase, 900 aggregate
CLI seconds and eight decision turns. These limits were frozen before exposure.
Controls may use ordinary logging, assertions and custom Python diagnostics.
Treatments additionally receive the generic TrainCheck observation documentation
and implementation; no rule targets this case and no task step requires its use.

Standard responses return native logs, training metrics and ordinary artifacts.
Do not automatically supply private grader state exports as a diagnostic shortcut.
Either arm may request and capture equivalent runtime state explicitly under the
same resource budget. Private grading began only after all final submissions
froze; grader feedback was not supplied during the task.

## Milestones, costs and evidence

| Milestone | Definition |
| --- | --- |
| First supported localization | First received visible statement containing a sufficient causal explanation supported by evidence available to the agent at that time. Source evidence qualifies; later traces cannot be credited retroactively. |
| First valid submitted implementation | Earliest captured submission later passing all prespecified independent contracts. This is a retrospective correctness measure, not proof the agent had already verified it. |
| First agent-visible runtime verification | First returned runtime evidence that supports the submitted repair under the declared checks. State exactly which conditions were checked; this can precede or follow independent qualification of that submission. |
| Final completion | Agent's final frozen implementation and report, including all accepted diagnostic requests and their outcomes. |

Times begin at each trial's launch. Retain raw wall time, shared-service waiting,
wall time excluding only that measured wait, and CLI execution time. Source-copy,
transport, startup, diagnostics and each agent's own execution remain charged as
appropriate; shared service waiting is not cluster scheduler time. Message receipt
and submission completion are observable upper bounds, not the instant of a
private reasoning step or edit.

Native input and output token counters are recorded at completed-turn boundaries.
For resumed sessions with cumulative counters, use validated deltas. Cached input
is a subset of input and reasoning output a subset of output; do not add either
twice. A within-turn localization can have a token interval rather than a precise
count. Missing usage is unknown, not zero. Report cache distribution and avoid
translating total-token differences into monetary savings without actual pricing.

GPU cost is supervised worker lifetime multiplied by allocated devices, including
initialization, diagnostics and failed requests. This measures allocated
GPU-seconds, not utilization or pure training compute. Keep solver, qualification,
instrumentation-preflight and private-grading costs in distinct ledgers. Record
process termination and ownership so failed distributed workers cannot silently
continue consuming resources.

Preserve prompts, raw CLI events with receive times, every source submission,
request/worker receipts, output manifests, environment versions and grading
artifacts. Record actual TrainCheck activation, trace coverage, failures and
produced records. Reading its manual or source is not instrumentation use; unused
availability cannot establish the effect or overhead of active use. Large data,
checkpoints and traces stay outside Git, with hashes and retrieval locations in
the final report. Source and protocol progress use signed commits and a focused
PR; no merge is part of this protocol.

## Interpretation limits and prior result

The [previous synthetic checkpoint pilot](checkpoint-continuation-costs-20261101.md)
had four first-turn repairs that later passed independent validation. All four
agents localized the ordering problem from source before requesting a GPU run;
neither treatment activated TrainCheck. Supported localization took
26.79–38.01 seconds, first valid submission took 99.39–111.12 seconds, and GPU
cost before those submissions was zero. These are measured results for that
particular case, not a general one-shot success rate for the paper's bugs.

This new case has public historical provenance and may be recognized from model
training data. A small, purposefully selected pairwise pilot cannot estimate the
incidence of silent errors in agent-authored ML systems or establish a general
TrainCheck advantage. It also does not evaluate the offline inference engine:
agents may author their own checks around observed state. If agents solve the
case from source, choose ordinary diagnostics, fail to use TrainCheck, or obtain
no efficiency benefit, retain and report those outcomes without changing the
task or success criteria after seeing them.

## Interrupted transport and recovery protocol

Both matched pairs used the frozen limits above. All four agents first proposed
the correct clipping hypothesis from source and requested runtime diagnostics
before editing the implementation. This case therefore must not be described
as requiring runtime observation for successful localization.

An SSH connection closed after control request 3 finished training but before
artifact download. Treatment request 2 failed while creating its remote directory,
before training launched. A bounded subsequent SSH check succeeded. The completed
control phase has exit code 0, unchanged source/native binaries, empty worker/GPU
process lists, and 17.31105645187199 seconds of two-GPU worker lifetime; it must
be recovered rather than rerun. The treatment request directory was absent.

Recovery preserved all original trial and error receipts. Pending results were
retrieved, the unlaunched request executed once, and the same CLI thread IDs
resumed with their original remaining request/turn/time budgets. New receipts
are append-only under `recovery-1/`. The second treatment's request 1 also
completed GPU work before an artifact-download failure; `recovery-2/` retrieved
that existing result and continued its original thread. Composed records
explicitly reference every original and continuation artifact. Report raw wall time including interruption and measured interruption
intervals separately. Select one effective receipt per logical request so recovery
does not double-count GPU cost. No private scoring or operator diagnosis is sent
to solvers. Independent grading began only after all four trials froze and is
now complete.

## Completed solver and independent-validation results

All four agents first localized the same clipping-selection cause from source,
then wrote ordinary cross-rank diagnostics to test that hypothesis. No TrainCheck
instrumentation activation was observed in either treatment. Both controls and both treatments
submitted their first independently valid implementation in **turn 2**, after
one diagnostic GPU request. All later submitted implementations also passed.

Independent grading covered every frozen submission, deduplicated by all Python
source bytes: eight distinct candidate sources and one known-correct reference,
each executed in four scenarios, for **36 GPU phases**. Each scenario compared
recorded model parameters, the complete exported optimizer state and recorded
Torch/Megatron RNG at steps 0, 1 and 2 against its fresh reference. Step 0 is
after the first backward pass. Model tolerances were `atol=0`, `rtol=0`;
optimizer container types/order, scalars and tensor contents remained significant.
Replicated parameters were also checked within their tensor-parallel group.

| Private scenario | Seed | TP / allocated GPUs | Clipping threshold | Turn-1 sources | Turn-2 and later sources |
| --- | ---: | ---: | ---: | --- | --- |
| Base | 1234 | 2 / 2 | 1.0 | All four fail | All four pass |
| Held-out seed | 5678 | 2 / 2 | 1.0 | All four fail | All four pass |
| TP1 control | 1234 | 1 / 1 | 1.0 | All four pass | All four pass |
| Clipping disabled | 1234 | 2 / 2 | 0.0 | All four pass | All four pass |

Thus the earliest valid-submission milestone is justified by all four gates,
not by the agent's own assertions or a successful training exit. These short
checks do not establish long-horizon convergence, checkpoint-resume correctness,
or coverage of other distributed configurations. No TP2×DP2 validation is claimed.

### Milestones and resource costs

All times are seconds from the trial's original start. In a `raw / adjusted`
cell, adjusted time subtracts the measured union of resource-lock waits and
operator suspension intervals. It is an accounting view, not a reconstruction
of uninterrupted execution. Source diagnosis has no prior queue/interruption.
“First valid” is the captured submission boundary, retrospectively graded;
“finish” is the final visible decision, not when private grading completed.

| Trial | Source diagnosis | Runtime diagnosis, raw / adjusted | First valid submission, raw / adjusted | Agent-visible repair verification, raw / adjusted | Finish, raw / adjusted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control 1 | 36.118 | 134.876 / 134.875 | 162.692 / 162.691 | 232.732 / 232.729 | 1,033.415 / 310.957 |
| TrainCheck 1 | 18.708 | 201.057 / 163.998 | 286.436 / 249.376 | 1,018.755 / 316.404 | 1,121.005 / 418.653 |
| TrainCheck 2 | 35.336 | 776.050 / 125.544 | 807.169 / 156.663 | 890.986 / 240.479 | 920.339 / 269.831 |
| Control 2 | 16.462 | 127.342 / 127.341 | 387.317 / 387.316 | 451.859 / 451.857 | 539.296 / 539.293 |

Agent-visible verification is the first completed inspection/check command that
exposes both ranks' corrected-run coverage, separate from a later prose claim.
Every corrected run actually contains all **16 updates on both ranks**, 52
clipped gradients per rank, and zero errors for the scaling/replica fields that
its agent chose to observe. Initial diagnostics used 3, 8, 4 and 4 updates in the
table's order. Focused tests executed on both workers: respectively 5, 7, 4 and
3 tests per worker. The public diagnostics have different coverage and are not
the independent uniform oracle. Control 1, TrainCheck 1 and Control 2 also ran
eight FP16 updates; TrainCheck 2 did not, and acknowledged that limitation.

The full-trial boundary occurs slightly after the visible finish statement.
Allocated GPU-seconds count devices × supervised worker lifetime, including
startup; they do not measure device utilization.

| Trial | Requests | Full raw wall s | Queue s | Operator suspension s | Full adjusted wall s | CLI s | GPU-s before first valid | Full solver GPU-s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control 1 | 3 | 1,037.150 | 0.003 | 722.455 | 314.692 | 173.658 | 34.731 | 104.607 |
| TrainCheck 1 | 3 | 1,124.788 | 37.062 | 665.290 | 422.436 | 271.649 | 35.239 | 94.831 |
| TrainCheck 2 | 2 | 924.628 | 42.890 | 607.618 | 274.121 | 159.723 | 34.730 | 69.355 |
| Control 2 | 3 | 543.570 | 0.002 | 0.000 | 543.568 | 397.481 | 34.720 | 94.298 |

The four solver trials used **363.091 GPU-s total**. The separate 36-phase private
grading used **1,069.616 GPU-s**; none is charged to solver milestones.
Qualification, instrumentation preflight and earlier setup work remain separate
operator costs in their own receipts; the two totals here are not a total
campaign-cost claim. Recollecting grades from existing snapshots added no GPU
training phases. Recovered completed requests are counted once through their
effective receipt; their original failures and unknown original measurements
remain preserved.

### Native token accounting

The following counts are native cumulative input-plus-output tokens, including
cached input. Intervals are inclusive previous/current completed-turn bounds;
no within-turn interpolation is used. First-valid and final completed-turn
counts are exact as reported. All source diagnoses occur in turn 1 before a GPU
request. Full per-category bounds are retained in the final JSON.

| Trial | Source-diagnosis interval | Runtime-diagnosis interval | First-valid tokens | Final completed-turn tokens |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 0–393,071 | 393,071–693,107 | 693,107 | 1,236,382 |
| TrainCheck 1 | 0–423,568 | 423,568–979,286 | 979,286 | 1,455,127 |
| TrainCheck 2 | 0–334,917 | 334,917–590,405 | 590,405 | 805,536 |
| Control 2 | 0–480,682 | 480,682–835,766 | 835,766 | 1,336,513 |

| Trial | Final input | Of input, cached | Final output | Of output, reasoning |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 1,226,306 | 1,137,280 | 10,076 | 1,975 |
| TrainCheck 1 | 1,440,861 | 1,350,784 | 14,266 | 4,260 |
| TrainCheck 2 | 796,875 | 727,936 | 8,661 | 2,396 |
| Control 2 | 1,327,625 | 1,243,776 | 8,888 | 1,431 |

Cached input and reasoning output are subsets, not extra tokens. These counts
are not a dollar-cost estimate. The final visible statement occurs within the
last turn; its own within-turn token interval is preserved separately from the
exact final completed-turn count.

### TrainCheck usage and the negative finding

TrainCheck 2 visibly read the generic manual. TrainCheck 1 only visibly listed
its filename; a listing is not counted as reading. Neither treatment emitted
observed TrainCheck trace files or added an explicit TrainCheck import in its
changed submitted Python. Both used ordinary custom state diagnostics. The
launcher option permitting TrainCheck access is not activation. No exhaustive
runtime module/dynamic-import audit was collected, so the precise conclusion
is **no instrumentation activation observed**, not proof that no incidental
import could ever have occurred.

This real distributed case still did not meet the intended difficulty criterion:
all four agents identified the cause from source in 16–36 seconds, before runtime
observations. Runtime evidence confirmed the consequence and supported the
repair, but was not necessary for their observed localization. These results
therefore do not establish a TrainCheck debugging-efficiency benefit or an
overhead for active use. They describe optional availability that went unused.

Two runs per arm on one selected case, unequal diagnostic lengths, one omitted
FP16 rerun, normal agent variation and asymmetric transport interruptions prevent
a causal speedup interpretation. Raw and adjusted values are both retained; no
benefit ratio is claimed. This finding also does not show that agents can solve
all TrainCheck paper cases in one shot. A subsequent case should first qualify a
failure whose cause remains ambiguous after source inspection and whose rank
state supplies the discriminating evidence, without hiding code or forcing use.
The next candidate is documented in the [DDP runtime-state
pilot](ddp-runtime-state-pilot-20261102.md).

### Evidence, replay and collection failure

Local operator evidence root:
`/Users/matt/gitrepos/traincheck/outputs/ring18-runtime-20261102/`.
The table below uses paths relative to that root. Private state snapshots remain
under `/home/yuxuan/experiments/traincheck-runtime-20261102/private-grading/` on
Ring18. Large snapshots, solver transcripts and datasets are not committed.

| Evidence | SHA-256 |
| --- | --- |
| `solver-completion-composed.json` | `c7c654a72841b75f66b0ed5de0b2af7668940c77295e58a467369b4d203c8c79` |
| `grading-plan.json` | `b2dd407117d09165b93d7596ef13ef13c4dc80055b6371b9f86ab3bff002e18c` |
| `grading/result-20261101T211500631757Z.json` | `6a9810b43694cb8b13c06da6edc8feb7ab489c59fc6357ca4342d6d6864307a6` |
| `milestone-annotations-final.json` | `fe1048a72fd9a77bef24f84f2ab110e2bb4bf6265792ac2fa021b65365dce509` |
| `milestone-summary-final.json` | `a858a1003cd14c6e179efdf410723e25caa99c0ddb5eaa4af2b12d6c0dfbaeea` |
| `private-grading-milestone-join.json` | `792219847ed15062d5b8b0b29090e52068c60fcc860d3a69df8077a9b6201c77` |

The join records every turn's submission path, source ID, Python-manifest hash,
optimizer-file hash and four scenario outcomes. Earliest valid submissions are:

| Trial | Submission path relative to the evidence root | Python source ID / SHA-256 |
| --- | --- | --- |
| Control 1 | `trials/control_1/submissions/turn-2` | `4a6f8230a5967b40a93d1ee792427048d72405ece18abd5944d5b1caf3510b11` |
| TrainCheck 1 | `trials/traincheck_1/submissions/turn-2` | `13db280c9f102dab5b50b8d663e4f6c4637bea6a5535bc811f16e2d350fbcc51` |
| TrainCheck 2 | `recovery-2/trials/traincheck_2/submissions/turn-2` | `e51f85808dc26b77adc7f13eca38e2aa82f1b26db69114b9a12c4805f745487a` |
| Control 2 | `trials/control_2/submissions/turn-2` | `d972171d51412890b226fab07f616ad1d2819a403e23fdc83a16b67a914faab6` |

`milestone-annotations-visible.json` and `milestone-summary-visible.json` remain
unchanged as the pre-grade analysis. Final artifacts are new files. Exact event
lines and hashes are bound in the annotations; raw transcripts are under
`trials/`, `recovery-1/trials/` and `recovery-2/trials/`. Recompute the final
accounting without training or solver interaction using:

```sh
python summarize_milestones.py --study-root "$PWD" --annotations milestone-annotations-final.json --output milestone-summary-recomputed.json
```

The output path must be new. `join_private_grading.py` validates the frozen plan,
all source/occurrence mappings, original phase receipts and all four scenario
outcomes before producing final annotations; it never launches remote work.

The initial collection produced only **unknown** outcomes because its caller
passed a tuple to an evaluator requiring a nonempty list of steps. It is retained
as `grading/result-20261101T211129986599Z.json`, SHA-256
`2b56e92008716bf7a7bc6d6642d7cb67fab6478f5460628ae5fc5b70feedf296`.
The collector's generated argument changed from `steps=(0,1,2)` to
`steps=[0,1,2]`; the old helper is archived. A focused integration regression
exercised the actual evaluator contract. Read-only recollection of the same
snapshots produced the final result above: no evaluator contract, plan, solver
feedback or GPU phase changed. Unknown evaluation was never treated as success.

The accounting/extraction and private-grade-join helpers pass 42 focused local
tests, explicit strict mypy and pinned formatting/lint checks. This validates
the checked contracts, not absence of every possible harness defect. Independent
state comparisons and retained failures are the basis of the reported outcomes.
