---
date: 2026-07-13
draft: true
slug: ml-training-wrong-loss-goes-down
categories:
  - ML Reliability
  - Distributed Training
description: A falling loss does not show whether distributed training executed the intended algorithm.
---

# ML Training Can Be Wrong Even When the Loss Goes Down

We often treat the loss curve as a health check for an experiment. A smooth,
decreasing curve gives us confidence. A spike or a sustained increase can warn
us that something may be wrong.

That instinct is useful. Loss is often the first sign of a broken run. But we
also ask it to answer a question it cannot: did this experiment execute the
algorithm we intended to test?

A training run today combines the model and optimizer with kernels, precision
casts, distributed collectives, compiler transformations, checkpoint code, and
framework defaults. Any of these can change the computation. The loss is
produced by the whole stack, yet it does not tell us which computation the stack
actually performed.

This leaves us with two problems. In the harder case, the implementation is
wrong while the loss looks healthy. In the more familiar case, the loss looks
wrong but cannot tell us whether the idea failed or its implementation did. The
first lets us trust an invalid run. The second can make us abandon an idea for
the wrong reason.

## A smooth curve does not mean the training is correct

People who train models already know that a good loss curve is not proof that
everything underneath it is correct. The BLOOM-176B training run makes this
limitation concrete. It is also unusually visible: the team documented the
failure in public, an openness that deserves credit.

During BLOOM-176B training, LayerNorm weights that should have been replicated
began to differ across tensor-parallel ranks. Loss and accuracy showed no
immediate anomaly, so the inconsistency remained undetected for ten days. The
eventual investigation found that gradient clipping in the BF16 optimizer ran on
only one tensor-parallel rank. The ranks were no longer applying the same update
to supposedly identical weights. ([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[OSDI '25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

The training signal did not expose the disagreement. Training continued even
though weights that should have been replicated no longer represented the same
state.

Training stacks have matured since BLOOM, but frontier work still exercises
configurations with limited prior testing. The specific bugs will differ; the
monitoring problem remains whenever we observe only the outcome without checking
the computation that produced it.

## A suspicious curve does not identify its cause

Not every implementation error stays hidden behind a healthy curve. Often the
curve eventually does look suspicious. That sounds easier: at least the run has
given us a warning. But what do we do next? Debug the system, tune the method,
or stop the run and conclude that the idea does not work?

In February 2026, a fix to the Flash Linear Attention implementation of Mamba-2
corrected how `dt_bias` and `A` were initialized. The author reported a
significant difference between training with the old and corrected
initializations. A follow-up fix addressed another path in which FSDP2's
distributed tensors had caused the intended initialization to be skipped.
([initialization fix](https://github.com/fla-org/flash-linear-attention/pull/739),
[FSDP2 fix](https://github.com/fla-org/flash-linear-attention/pull/753))

Before those fixes, a disappointing Mamba-2 curve could have invited a story
about the architecture, the data, or the optimizer. Correcting the known
initialization defects made the resulting curve better evidence about those
choices; it did not by itself validate the rest of the implementation.

A loss spike, plateau, or regression can come from data, initialization,
optimizer settings, numerical precision, random variation, or an implementation
error. With a new method, there are fewer known-good results for ruling these
explanations out. That ambiguity is why teams look beyond the curve.

## Existing monitoring practices are insufficient

Training teams already watch much more than loss.
[TensorBoard](https://www.tensorflow.org/tensorboard/get_started), [Weights &
Biases](https://docs.wandb.ai/models/track/log), and [MLflow
Tracking](https://mlflow.org/docs/latest/ml/tracking/) let researchers record
and compare metrics across runs. A useful dashboard may include the learning
rate, gradient and update norms, parameter distributions, throughput, memory,
and hardware utilization. These signals can surface signs of unstable
optimization, a stalled input pipeline, growing memory use, or departure from a
known baseline. But most routine monitoring still consists of high-level
numerical signals. They describe symptoms without establishing that the intended
computation ran.

A stable reference enables stronger checks. Instead of asking whether a value
looks plausible, we can compare a changed execution with one believed to be
correct and ask what should have remained unchanged. Yet a training trace spans
many tensors, operations, ranks, and steps. A reference run improves the
available evidence, but the space of possible comparisons remains enormous. It
does not by itself tell us which relationships matter or whether an unobserved
part of the execution was wrong.

## A broader risk: Silent implementation errors can be mistaken for negative results

When an established recipe stops working, there is a known-good result to
recover. The implementation becomes an obvious suspect, and the team has a
reason to keep debugging.

When a new architecture, objective, or training method underperforms,
researchers may reasonably conclude that the idea failed. Yet the same decision
to stop a run can follow from either a genuine negative result or a silent
implementation error.

Public bug reports record the cases someone investigated far enough to find
divergent replicas or faulty initialization. They cannot tell us how often an
implementation error is mistaken for a negative result, because many abandoned
experiments may never be diagnosed. Our claim is narrower: a loss curve alone
cannot distinguish a failed idea from a faulty implementation.

We can ask instead whether the execution maintained the relationships required
for the result to mean what we think it means. The next post examines those
relationships, how we might monitor them, and how much closer a violation brings
us to the actual fault.

---

*This is the first post in a three-part series on trustworthy ML training. Next:
[Why ML Training Failures Are So Hard to
Localize](traincheck-in-practice.md).*
