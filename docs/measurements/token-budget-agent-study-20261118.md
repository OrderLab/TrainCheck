# Implementing token-budget accumulation under DDP

All six agents delivered a correct implementation on their first submission.
Each then ran a handwritten two-GPU PyTorch diagnostic before declaring the
same source ready. We observed **0/6 silent errors at delivery** and **0/6
self-detected and repaired numerical defects**. All three optional-tool agents
read the TrainCheck guide. None used it in the executed training or diagnostics.

This prospective Ring18 study asks whether six fresh agents deliver a numerically
correct feature when they add token-budget gradient accumulation to a working
fixed-mode training loop. Three agents receive the ordinary task and three can
also use TrainCheck. Each pair receives the same task, model settings, source,
compute and decision budgets. We retain all six outcomes, including declines,
errors and incomplete work.

## Task and endpoint

The task specifies the user-facing behavior of a token budget, including whole
synchronous microbatches, a final partial window and the objective for the same
effective batch. It provides a working fixed mode. Neither arm is told that a
bug exists. The optional arm gets generic TrainCheck documentation and code.
The solver may choose its own tests, use ordinary PyTorch or use TrainCheck.
No hidden grade or reference implementation is returned during the trial.

Every decision that submits a run or finishes freezes the production source.
After the conversations end, the evaluator checks the submitted source against
the same independent numerical reference, including fixed mode, update
boundaries, gradients and resulting parameters on both ranks. Reuse a replay
only when the complete inputs affecting it are identical. The reference uses
the corresponding concatenated global batch. Native qualification must show
that a correct implementation passes and that a normally completing, numerically
incorrect implementation can be distinguished.

Report three outcomes for each solver:

1. **Delivered silent error.** The agent declares the implementation ready,
   the affected training path completes normally, and an independent check
   finds incorrect training behavior. Record which check failed. Separately
   record crashes, an unsupported feature, explicit non-delivery and unknown
   infrastructure outcomes. A solver that warns about an unresolved discrepancy
   has not silently declared a verified implementation ready.
2. **Self-detected and repaired.** Bind the agent's own observation of a
   concrete defect to its operation record or artifact, and the earlier wrong and
   later correct implementation to independent checks. Label static code
   review and executed diagnostics separately. A plausible concern or change
   of implementation without evidence of a defect is insufficient. Record
   within-turn fixes separately when a recoverable earlier revision and
   provenance exist. Otherwise mark earlier correctness unobserved.
3. **First correct submission.** Identify the earliest frozen submission that
   passes the complete hidden contract. Keep earlier unknown grades visible.
   Count elapsed time from trial start, active CLI time, input/cached input/output
   tokens and actual allocated GPU-seconds incurred **before that submission**.
   A run requested by that submission comes after the boundary. A solver can
   submit a valid implementation with zero prior GPU use. Retain the corresponding
   time and cost for unsuccessful trials through their last decision.

## Design and reporting

- Three independent pairs. Fresh conversations and separate packets. The
  CLI worker has six decisions, four public runs, 900 cumulative CLI seconds
  and 120 seconds per public worker. Record exact launch order and environment.
- Record baseline revision, task and packet hashes, reference and controls,
  model, CLI settings, physical GPUs, raw operation records, snapshots, ordinary
  output and per-phase receipts. Keep case-specific operator material outside
  all solver packets. Record any deviations from this protocol explicitly.
- Record wall time and queue wait independently of active CLI time. Count cached
  input once in total tokens and show uncached input plus output separately.
  Allocated GPU-seconds equal device count times allocation duration. Training
  qualification and hidden evaluation are separate from the agents' GPU costs.
- Report the six agents individually, paired differences, actual tool use and
  any diagnostic that changed an agent's next action. Identify correctness
  failures even when the final training loss is finite or replicas agree.
  Three pairs on one task are an early validation of this workflow.

The implementation baseline for the optional tool is the signed snapshot work
in [PR #172](https://github.com/OrderLab/TrainCheck/pull/172). Prior numerical
qualification of global token-mean objectives is documented in the
[DDP study](ddp-runtime-state-pilot-20261102.md). This task requires a new
implementation and new reference runs for its token-budget boundaries.

## Native qualification and reproduction

The pilot uses a 725-parameter token classifier (embedding, tanh and a linear
projection with bias) and SGD with momentum. The experiment runs on physical
A16 devices 4 and 5 on Ring18, driver 535.309.01, Python 3.10.14 and PyTorch
2.2.2+cu121. Each rank receives five variable-length microbatches in the ordinary
example. The task and tests use non-padding target counts, including zero local
targets on a rank in one hidden case. All six solvers use native Codex CLI
0.160.0, requested model `gpt-6-astra`, high reasoning effort and priority
service. Each receives a fresh conversation and a separate workspace with
instructions limiting it to the packet. Pair order is C1/T1, T2/C2, T3/C3.
Both agents in a pair can work concurrently while the GPU service serializes
their requests. The next pair starts when both finish.

The withheld tests cover fixed mode, exact budget, overshoot with a partial
tail, rank skew, a whole epoch below budget and one-microbatch updates.
The oracle concatenates the examples for each global update and uses mean
cross-entropy directly. On both ranks the evaluator observes the gradient
before the optimizer step, parameters and momentum afterward, final state and
which microbatches formed each window. Tensor comparisons use absolute tolerance
`1e-6` and relative tolerance `1e-5`. It also checks grad-producing forward
sizes and overlap of outstanding forwards. Source review supplements those
checks for buffering of the input iterator.

The healthy fixed-mode starter completed ordinary DDP training on both GPUs.
The starter passed fixed-mode replay and explicitly reported the new feature
as unsupported. A separate correct implementation passed all six scenarios.
A control that divides by the requested budget instead of the actual number
of targets ran to completion and failed the numerical checks in four scenarios.
Independent imports on both workers confirmed that control lacked TrainCheck
and treatment could take a real CUDA snapshot using the supplied package.
Six qualification requests used 128.585 allocated GPU-seconds.

The operator sources are under [tools/token_budget_study](../../tools/token_budget_study).
Local raw evidence lives in `outputs/ring18-token-budget-20261118/` in the
canonical TrainCheck checkout. The matching Ring18 experiment directory is
`/home/yuxuan/experiments/traincheck-token-budget-20261118`.
`native-launch.json` binds the packets, CLI and controller, with SHA-256
`4f9f387e33858c67ed1c98bd61ad269b977cf102725cef0fdfe7bc13115e5c4f`.
`ring18-qualification-evidence.json` identifies environment and source hashes
and all native qualification receipts, with SHA-256
`6bf3c8fa7f736a835ae2dd9b5d47fcadbcdfdcc019c1ae261055d6856c30d60b`.
The operator retains native turn logs and copies production source at every
decision. An additional watcher samples edits approximately every 50 ms.
Intermediate revisions that exist only between samples can be missed.

## First correct submission and delivery

| Pair | Arm | First correct turn | Elapsed seconds | Active CLI seconds | Total tokens | Uncached input + output | Prior GPU-seconds | GPU-seconds through delivery |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Control | 1 | 61.954 | 61.949 | 109,666 | 25,186 | 0 | 25.008 |
| 1 | Optional TrainCheck | 1 | 56.194 | 56.184 | 90,894 | 26,382 | 0 | 24.510 |
| 2 | Control | 1 | 54.371 | 54.366 | 74,898 | 22,034 | 0 | 24.909 |
| 2 | Optional TrainCheck | 1 | 68.776 | 68.768 | 128,224 | 14,816 | 0 | 24.704 |
| 3 | Control | 1 | 60.547 | 60.542 | 94,550 | 10,070 | 0 | 24.916 |
| 3 | Optional TrainCheck | 1 | 64.251 | 64.241 | 95,300 | 22,724 | 0 | 24.796 |

Median elapsed time to first correct submission was 60.55 s in control and
64.25 s with TrainCheck available. Median total token counts were 94,550 and
95,300 respectively. Total tokens include cached input once. Every first
submission requested a diagnostic run, so the cost to produce the correct code
was zero GPU-seconds for every solver. All six then incurred the GPU cost in
the last column to verify their implementation. The paired times and token
counts moved in different directions across pairs. The observed choices
provide no measurement of the effect of actually using TrainCheck.

All six solvers chose to compare each distributed update against ordinary
training on its combined global batch. Their two-rank diagnostics completed
successfully. Each solver's production source was byte-identical at the first
request and at final delivery. Across the watched edits we saw only the
unsupported starter and one feature implementation per solver. Independent
replay of all six feature implementations passed all six scenarios on both
ranks, including gradients, parameters, momentum and update boundaries (72
rank/scenario outcomes). Each had at most one outstanding grad-producing
forward in these cases. Review of all six sources found streaming accumulation
without buffering a window of inputs.

This was a small new feature on a healthy baseline. The starter already showed
how to count targets across ranks and normalize a single-microbatch update.
The public contract specified equivalence to the combined global batch, which
gave both arms an executable check. The results apply to this task, model,
three pairs and six held-out scenarios. We cannot estimate how often agents
introduce silent errors across ML research tasks or how much TrainCheck saves
when a faulty implementation actually occurs.

The six native conversations had distinct IDs. In the recorded commands C2,
C3 and T2 read a generic host policy skill, and T2 also read its schema. No
access to peer solutions or private grading material was observed. All tool
adoption and self-correction verdicts describe observable actions.

## Cost and evidence

| Category | Physical two-GPU allocations | Allocated GPU-seconds |
| --- | ---: | ---: |
| Native qualification | 6 | 128.585 |
| Solver-requested diagnostics | 6 | 148.844 |
| Independent source replays | 8 | 210.974 |
| Total | 20 | 488.403 |

The 12 decision submissions and 18 within-turn source observations reduced
to eight distinct arm/source combinations, including the unsupported starter
in both environments. All 48 native scenario runs have known results. Each
allocation has a worker receipt and verified GPU release.

The [machine-readable results](data/token-budget-agent-study-20261118.json)
include both submission boundaries per agent, terminal clocks, native token
counters, and hashes of the raw evidence. In the local evidence directory,
`behavior-audit.json` records each solver's evidence for its delivery,
`native-operation-audit.json` verifies token and operation history, and
`ring18-allocation-audit.json` reconciles every physical allocation. To
recompute the costs from those retained submissions and private receipts,
run this command from the repository root, selecting a new output path:

```sh
python -m tools.token_budget_study.operator.score_submissions "$evidence_directory" "$new_output_json"
```
