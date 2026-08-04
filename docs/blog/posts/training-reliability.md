---
date: 2026-07-13
draft: true
slug: training-run-not-scientific-evidence
categories:
  - ML Reliability
  - Distributed Training
description: A training result supports a claim about an algorithm only if we can distinguish the behavior of the algorithm from the behavior of its implementation.
---

# Your Training Run Is Not Scientific Evidence

Not yet, anyway.

A training run produces an observation. Turning that observation into evidence
about a model, objective, or optimization method requires another claim: that
the system executed the experiment we intended.

That claim is routinely difficult to establish.

When a run underperforms, the hypothesis space is enormous. The idea may be
wrong. The data mixture may be wrong. Optimization may be unstable. A kernel
may be numerically incorrect. Distributed state may have diverged. The code may
have updated a different set of parameters from the ones used in the forward
pass. Several of these can happen at once.

The result alone does not identify which experiment actually ran.

## This is not an observability-is-missing argument

Serious training stacks already collect far more than loss and accuracy. They
track gradients and update norms, learning rates, activations, throughput,
collective latency, memory, hardware health, data statistics, per-rank state,
and application-specific signals. Teams add canaries, checkpoint validation,
assertions, anomaly detectors, and increasingly detailed postmortem tooling.

That telemetry is indispensable. It catches many failures and makes many others
diagnosable.

The remaining problem is a semantic gap. Most signals describe what values the
system produced or whether its components remained operational. They do not, by
themselves, establish that the composition of those components implemented the
intended training procedure.

A plausible loss curve is compatible with an incorrect experiment. So is a
gradient norm within its historical range. Even agreement with a smaller run
may be uninformative if the failure appears only after introducing sharding,
mixed precision, compilation, or a different optimizer path.

The distinction is not between “ML problems” and “systems problems” as two
cleanly separable layers. Modern training makes that boundary porous. Precision
changes affect optimization. Parallelism changes parameter ownership and
update semantics. Data systems determine the effective objective. Compiler and
kernel choices change the numerical program.

The distinction that matters is between **observing an outcome** and
**establishing implementation fidelity**.

## One symptom, many valid explanations

Jack Morris once described a Distributed Data Parallel run whose loss fell
until roughly step 150 and then rose without a corresponding increase in
gradient norm. The discussion produced reasonable hypotheses: learning-rate
instability, initialization, regularization, and a missing
`optimizer.zero_grad()` call. ([Original
question](https://x.com/jxmnop/status/1778436832075678100))

The root cause was that the code called the raw `nn.Module` instead of its DDP
wrapper. Gradients did not synchronize, and each GPU learned independently.
([Root-cause
follow-up](https://x.com/jxmnop/status/1778520637193240892))

The interesting point is not that somebody made an easy mistake. It is that
the observed training dynamics supported several sophisticated explanations
that were all downstream of the wrong execution. More analysis of the loss
curve could have refined a model of an experiment that had never taken place.

This failure has a direct execution-level signature: parameters intended to be
replicas ceased to agree across ranks. That fact does not explain whether the
learning rate is appropriate or the research idea is sound. It does something
more basic first—it falsifies the assumption that the distributed program is
implementing synchronous data-parallel training.

## Detection, diagnosis, and validity are different problems

It is useful to separate three goals that are often collapsed into “training
monitoring”:

1. **Detection:** Is the run behaving unusually or violating an operational
   threshold?
2. **Diagnosis:** Which component or event most likely caused the observed
   behavior?
3. **Experimental validity:** Did the execution preserve the assumptions needed
   to interpret this result as evidence about the research hypothesis?

A loss spike can solve the first problem. A correlated gradient anomaly may
help with the second. Neither necessarily solves the third.

Experimental validity does not require proving an entire training program
correct—no practical checker can do that. It requires collecting enough direct
evidence to rule out important alternative explanations. Did replicas remain
consistent? Did the optimizer update the parameters that received gradients?
Did the intended initialization execute? Did rollout workers use the policy
version attributed to their samples?

These are narrower claims than “the run is correct.” They are also closer to
the assumptions on which we base our conclusions.

## Training has outgrown exact-output oracles

Traditional correctness techniques struggle here for a familiar reason: the
expected output of training is rarely known. Seeds, data order, hardware,
kernels, and distributed schedules legitimately alter the numerical
trajectory. At frontier scale, reproducing the full environment may itself be
impractical.

But the absence of an exact-output oracle does not imply the absence of a
specification. Training contains partial specifications everywhere:

- state that is replicated should remain consistent;
- state that is sharded should have the expected ownership;
- an optimizer should act on the parameters associated with its gradients;
- required operations should occur in the intended order and context;
- versioned components should exchange mutually compatible state.

These relationships say little about where the loss should end. They say a great
deal about whether the observed loss came from the procedure we meant to study.

That is the reliability layer we think is underdeveloped: not another anomaly
score over surface metrics, but evidence about the semantics of the execution
itself.

The next post develops this idea as **training invariants**. The core question
is not whether a checker can predict a stochastic trajectory. It is whether a
training system can continuously test the partial specifications that make its
results interpretable.

---

*This is the first post in a four-part series on trustworthy ML training. Next:
[We Cannot Predict the Loss Curve. But We Can Still Check the
Training](training-invariants.md).*
