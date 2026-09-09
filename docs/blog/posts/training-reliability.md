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
immediate anomaly, and cross-rank consistency was not being checked. The team
found the divergence while investigating another problem; it had already
persisted for ten days. The eventual investigation found that gradient clipping
in the BF16 optimizer ran on only one tensor-parallel rank. The ranks were no
longer applying the same update to supposedly identical weights. ([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[DeepSpeed fix](https://github.com/deepspeedai/DeepSpeed/pull/1801),
[OSDI '25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![A decreasing loss curve above replicated weights that agree before
an update and diverge afterward](../../assets/blog/training-reliability/01-healthy-loss-hidden-divergence.png)

We later reproduced the bug in a smaller CodeParrot run. Its logged gradient
norm closely tracked a diagnostic run that synchronized the LayerNorm replicas
before each forward pass. Their validation loss and perplexity differed early,
then approached similar values over their shared first 2,000 steps.

The curves were not identical, but their differences did not identify the
violated relationship; gradient norm did not clearly distinguish the faulty and
diagnostic runs either. The relevant question was whether weights intended to be
replicated remained equal.

Training stacks have matured since BLOOM, but new configurations still pose the
same monitoring problem: high-level outcomes do not tell us which computation
ran.

## A suspicious curve does not identify its cause

When a curve does look suspicious, it still leaves a choice: debug the system,
tune the method, or stop the run and conclude that the idea does not work?

In 2024, Jack Morris reported a distributed data-parallel (DDP) run whose loss
fell until roughly step 150, then rose without a corresponding increase in
gradient norm. Replies to the [original
question](https://x.com/jxmnop/status/1778436832075678100) proposed learning-rate
instability, initialization, regularization, and a missing
`optimizer.zero_grad()` call. The same curve was consistent with all of them.

![A loss curve that falls, plateaus, and then rises above several possible
explanations, including data, initialization, optimizer settings, numerical
precision, random variation, and an implementation
error](../../assets/blog/training-reliability/02-suspicious-loss-many-causes.png)

Morris later [traced the
problem](https://x.com/jxmnop/status/1778520637193240892) to a training path that
called the raw `nn.Module` instead of the DDP wrapper. The gradients from that
path did not synchronize, so each GPU updated its model from its own local
gradients. The rising loss showed that something was wrong. It did not show that
the GPUs were no longer training one synchronized model.

A more direct check would compare the same replicated trainable parameter across
ranks after each update. In DDP, those replicas should remain equal. A
disagreement could have narrowed the investigation to distributed execution
before the loss began to rise.

The same ambiguity matters more when the method itself is new and there are
fewer known-good results. Recent Mamba-2 initialization fixes show why: a
disappointing curve could reflect the architecture, or initialization code that
never produced the intended state. ([initialization
fix](https://github.com/fla-org/flash-linear-attention/pull/739), [FSDP2
fix](https://github.com/fla-org/flash-linear-attention/pull/753))

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

## Silent implementation errors can look like negative results

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
