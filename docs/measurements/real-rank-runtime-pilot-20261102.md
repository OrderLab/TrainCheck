# Real cross-rank runtime case: agent-efficiency protocol

**Status: in progress, 2026-11-02.** The native TP2 case reproduces on Ring18;
qualification controls and the agent service are being validated. Agent trials
have not started, so no debugging-efficiency improvement is claimed. This is
operator documentation and must not be included in solver packets.

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
does not establish a quality regression. The neutral task will present measured
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
configuration, packet manifests, commands and raw evidence must be frozen before
the first solver sees the task.

## Question and case selection

The question is whether optional access to TrainCheck helps an agent investigate
and repair a real distributed training problem more efficiently under a neutral
engineering task. The comparison will measure causal localization, independently
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
indentation level. This artifact has been read and hashed locally; the complete
source snapshot and provenance manifest remain to be frozen.

The planned candidate is an isolated snapshot of the inspected working-tree
source. The positive control uses the corresponding correct HEAD version of
`bf16_optimizer.py` while preserving the same remaining compatibility changes.
The final evidence bundle must retain the exact source diff and per-file hashes,
including a check that the control differs only in the intended logic. This is a
historical semantic reproduction on a compatible environment, not execution of
an untouched historical checkout. Shared repositories and installed environments
are inputs and must remain unchanged.

## Planned workload and qualification

Use native Megatron–DeepSpeed GPT training on CodeParrot: four transformer layers,
hidden width 64, TP2, BF16 and ZeRO stage 0. Sequence lengths 128 and 256 are
qualification candidates; the chosen value, microbatch size, accumulation,
optimizer settings, clipping threshold, seeds and update count remain to be
frozen. Do not infer a model parameter count or performance result from this
configuration alone. Refresh the live Ring18 environment and available GPU
inventory before execution.

Qualification must establish all of the following before solver launch:

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

Plan two fresh matched pairs, with launch order reversed in the second pair.
Use the same model, reasoning setting, tool interface, runtime, request allowance,
update allowance and time limits. Freeze their exact values before exposure.
Controls may use ordinary logging, assertions and custom Python diagnostics.
Treatments additionally receive the generic TrainCheck observation documentation
and implementation; no rule targets this case and no task step requires its use.

Standard responses return native logs, training metrics and ordinary artifacts.
Do not automatically supply private grader state exports as a diagnostic shortcut.
Either arm may request and capture equivalent runtime state explicitly under the
same resource budget. Private grading begins only after all final submissions
freeze; grader feedback is not supplied during the task.

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
