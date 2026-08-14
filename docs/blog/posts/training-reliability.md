---
date: 2026-07-13
draft: true
slug: ml-training-wrong-loss-goes-down
categories:
  - ML Reliability
  - Distributed Training
description: Silent implementation errors in ML training can produce the same symptoms as ordinary optimization failures.
---

# ML Training Can Be Wrong Even When the Loss Goes Down

A falling loss tells us that the reported objective is decreasing. It does not
establish that the intended training procedure ran.

That distinction is easy to lose in modern ML because the design space is
enormous. When training disappoints, there are many legitimate explanations:
the data mixture, model architecture, objective, initialization, optimization
regime, or scale may be wrong. A silent implementation error can produce the
same symptoms.

**Wrong execution does not have to look like software failure. It can look like
ordinary ML.** A job can keep every accelerator busy, produce checkpoints, and
follow a plausible loss trajectory while synchronizing the wrong state,
updating the wrong parameters, or skipping part of the intended procedure.

## The loss matched. The gradients did not

In August 2025, a bug report compared TorchTitan's Mixture-of-Experts training
with and without expert parallelism. On the same inputs and weights, both paths
produced exactly the same loss. With two-way expert parallelism, however, every
expert gradient was almost exactly twice as large. The reporter also observed
equivalent loss curves in a training workload despite the doubled gradients.
([PyTorch issue](https://github.com/pytorch/pytorch/issues/160285))

The forward computation was correct. The backward computation had the wrong
semantics: combining FSDP with expert parallelism was missing the factor that
normalizes reduced gradients. The fix was merged into TorchTitan the next day.
([TorchTitan fix](https://github.com/pytorch/torchtitan/pull/1551))

This is a particularly inconvenient failure mode. A forward-loss parity check
passed exactly, and a longer training workload produced equivalent loss curves.
Adam-like optimizers can partially hide a uniform gradient rescaling; gradient
clipping and other optimizers need not. Either way, the curve did not establish
whether parallelization preserved the gradients it was supposed to compute.

## New ideas receive less debugging than established recipes

When a standard recipe unexpectedly stops working, the implementation is an
obvious suspect. There is a known-good result to recover, so the team keeps
debugging. When a new architecture, objective, or training method
underperforms, “the idea does not work” is a reasonable stopping condition.

This makes public bug reports survivorship-biased. The cases we can document are
the ones someone kept investigating until the implementation error was found.
We do not see experiments that were never revisited because their bad results
looked reasonable enough.

We cannot count the ideas lost this way. A silent implementation failure and a
legitimate negative result can leave the same artifact: a run that did not
perform well enough to continue.

## More curves do not resolve execution ambiguity

Serious training efforts inspect much more than loss and accuracy: gradient and
update norms, activations, data statistics, per-rank values, numerical health,
and application-specific signals. These measurements catch many failures and
constrain the hypotheses for many others.

But aggregate signals discard execution detail. The same gradient norm can come
from correctly synchronized replicas or several models drifting apart. A
parameter update norm says little if the optimizer owns a different parameter
from the one used in the forward pass. A plausible loss does not establish that
initialization followed the intended code path.

Nor is this a clean separation between “ML failures” and “systems failures.”
Precision changes optimization. Parallelism changes parameter ownership and
update semantics. Data systems determine the effective objective. Compilers and
fused kernels change the numerical program.

The practical distinction is between observations that describe the **outcome
of training** and observations that establish whether expected **training
relationships** held.

## Stochastic training still has partial specifications

We usually cannot say what the loss must be at step 1,000. We can often say what
must happen during that step: replicated state should remain consistent; an
optimizer should act on the parameters associated with its gradients; required
initialization should execute; and versioned components should exchange
compatible state.

These relationships do not prove that a run is correct or that an idea is good.
They help establish whether the run is informative about the idea at all.

We call them **training invariants**. The next post introduces
[TrainCheck](https://github.com/OrderLab/TrainCheck), our attempt to infer these
relationships from reference runs and report when a new execution violates
them.

---

*This is the first post in a three-part series on trustworthy ML training. Next:
[TrainCheck: Catching Training Bugs Before the Loss Curve
Does](traincheck-in-practice.md).*
