# Implementing token-budget accumulation under DDP

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
