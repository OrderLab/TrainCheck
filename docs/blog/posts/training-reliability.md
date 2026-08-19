---
date: 2026-07-13
draft: true
slug: ml-training-wrong-loss-goes-down
categories:
  - ML Reliability
  - Distributed Training
description: A plausible loss curve can survive an implementation error that changes the training algorithm itself.
---

# ML Training Can Be Wrong Even When the Loss Goes Down

We routinely accept a plausible loss curve as evidence that an experiment
tested the idea we intended to test. That standard is too weak.

A falling loss establishes that the reported objective is decreasing. It does
not establish that replicas synchronized, the right parameters were updated, or
the distributed program preserved the gradients of the original model.

This is not a philosophical distinction. In a recent Mixture-of-Experts bug,
the loss matched exactly while every expert gradient was wrong.

## Same loss, two-times gradients

In August 2025, a bug report compared TorchTitan training with and without
expert parallelism. The test used the same inputs and weights in both paths.
The losses were identical to eight decimal places:

```text
Loss without expert parallelism: 0.65229332
Loss with expert parallelism:    0.65229332
```

The expert gradients were not identical. With two-way expert parallelism, each
was almost exactly twice as large. The total gradient norm changed from
`0.571427` to `1.143555`, a ratio of `2.001227`. The reporter also observed
equivalent loss curves in a non-test workload despite the doubled gradients.
([PyTorch issue](https://github.com/pytorch/pytorch/issues/160285))

The forward computation was correct. The backward computation was not.
Combining FSDP with expert parallelism was missing the factor that normalizes
the reduced gradients. A small change to the reduction path fixed the semantics
and was merged into TorchTitan the next day. ([TorchTitan
fix](https://github.com/pytorch/torchtitan/pull/1551))

The experiment had passed an obvious check: parallel and non-parallel execution
produced the same loss. That check had validated only the forward pass.

## A plausible curve does not make the bug harmless

The equivalent loss curves are not as surprising as they first appear.
Adam-like optimizers can partially cancel a uniform rescaling of gradients.
That can hide the error in parameter updates, especially over a short
comparison. Gradient clipping, optimizer epsilon, weight decay, other
optimizers, and changes in parallelism degree need not preserve that
cancellation.

More importantly, a team should not have to argue that an incorrect gradient is
probably harmless. If enabling expert parallelism scales gradients with the
expert-parallel degree, then it changes the training procedure. Any conclusion
attributed to the model or method now also depends on an accidental
implementation detail.

This is the uncomfortable point: **a result can look reproducible at the metric
level while failing to reproduce the algorithm.**

## More ML metrics do not answer the execution question

Serious training efforts inspect far more than loss: gradient and update norms,
activations, data statistics, per-rank values, numerical health, and
application-specific signals. These measurements catch many failures.

They still describe outcomes more readily than execution semantics. A total
gradient norm might reveal the TorchTitan discrepancy if someone compares the
right configurations closely enough. It does not say which parameters were
scaled incorrectly or which distributed operation introduced the factor. A
normal-looking norm does not establish that an optimizer owns the parameters
used in the forward pass or that every required operation ran on every rank.

Nor is this a clean separation between “ML failures” and “systems failures.”
Precision changes optimization. Parallelism changes parameter ownership and
update semantics. Compilers and fused kernels change the numerical program.
The system is part of the algorithm being evaluated.

## Negative results have the weakest protection

When an established recipe stops working, there is a known-good result to
recover. The implementation becomes an obvious suspect, and the team has a
reason to keep debugging.

When a new architecture, objective, or training method underperforms, “the idea
does not work” is a reasonable stopping condition. A legitimate negative result
and a silent implementation error can leave the same artifact: a run that did
not perform well enough to continue.

Public bug reports are therefore survivorship-biased. We see the cases someone
investigated until they found doubled gradients or divergent replicas. We do
not see the experiments that were abandoned because their wrong results looked
reasonable.

We cannot prove how often this happens. We can require better evidence before a
run is allowed to change a research decision.

Exact loss values are rarely specifiable, but parts of training execution are.
Parallelization should preserve intended gradients. Replicated state should
remain consistent. Optimizers should update the parameters associated with
their gradients. Required operations should execute in the right context.

The next post examines why violations of these relationships are difficult to
localize, which patterns recur across training stacks, and where execution-level
checks such as [TrainCheck](https://github.com/OrderLab/TrainCheck) can help.

---

*This is the first post in a three-part series on trustworthy ML training. Next:
[Why ML Training Failures Are So Hard to
Localize](traincheck-in-practice.md).*
