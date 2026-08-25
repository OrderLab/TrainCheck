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

We often treat a plausible loss curve as evidence that an experiment tested the
intended idea. But a loss curve supports a narrower claim: the reported
objective decreased.

It does not show that replicas synchronized, the right parameters were updated,
or the distributed program preserved the gradients of the original model.

A 2025 Mixture-of-Experts bug made that distinction concrete: the loss matched
exactly while every expert gradient was wrong.

## Same loss, doubled gradients

In August 2025, a bug report compared TorchTitan training with and without
expert parallelism. The test used the same inputs and weights in both paths.
The losses were identical to eight decimal places:

```text
Loss without expert parallelism: 0.65229332
Loss with expert parallelism:    0.65229332
```

With two-way expert parallelism, every expert gradient was almost exactly twice
as large. The total gradient norm changed from
`0.571427` to `1.143555`, a ratio of `2.001227`. The reporter also observed
equivalent loss curves in a non-test workload despite the doubled gradients.
([PyTorch issue](https://github.com/pytorch/pytorch/issues/160285))

Both paths computed the same forward-pass loss, but their backward passes
disagreed. Combining FSDP with expert parallelism omitted the factor that
normalizes the reduced gradients. Adding that factor corrected the gradients,
and the patch merged into TorchTitan the next day. ([TorchTitan
fix](https://github.com/pytorch/torchtitan/pull/1551))

## A plausible curve does not make the bug harmless

Adam-like optimizers help explain why the loss curves remained similar. They can
partially cancel a uniform rescaling of gradients, leaving parameter updates
similar over a short comparison. Gradient clipping, optimizer epsilon, weight
decay, other optimizers, and changes in parallelism degree need not preserve
that cancellation.

Even when the optimizer partly cancels the error, enabling expert parallelism
still changes the training procedure. Conclusions attributed to the model or
method then also depend on an unintended implementation detail.

**Metric-level reproducibility does not imply algorithmic reproducibility.**

## What existing monitoring approaches can—and cannot—show

[TensorBoard](https://www.tensorflow.org/tensorboard/get_started) and
[Weights & Biases](https://docs.wandb.ai/models/track/log) track user-selected
metrics, parameter and gradient distributions, and system telemetry across
runs. [Cockpit](https://proceedings.neurips.cc/paper/2021/hash/ae3539867aaeec609a4260c6feb725f4-Abstract.html)
adds diagnostics from gradient distributions and curvature.
[DeepDiagnosis](https://doi.org/10.1145/3510003.3510071) checks training-time
values for symptoms such as exploding tensors, unchanged weights, and vanishing
gradients. These tools can catch numerical and optimization failures before
they affect a final metric.

[PyTea](https://arxiv.org/abs/2112.09037) statically checks tensor-shape
constraints, and
[CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf)
compares model executions across deep-learning backends. In distributed runs,
[PyTorch's debug mode](https://docs.pytorch.org/docs/stable/distributed.html#torch-distributed-debug)
checks that ranks issue matching collective operations with consistent tensor
shapes.

Each check establishes a particular property. None alone shows that expert
parallelism preserved every parameter's gradient. In the TorchTitan case, a
total gradient norm would reveal the discrepancy if the parallel and
non-parallel configurations were compared directly. In a single run, however,
even a doubled norm might look plausible, and the collective operations could
still match in shape and order. The relevant check is relational: with the same
inputs and weights, enabling expert parallelism should preserve the intended
per-parameter gradients.

## One symptom supports too many explanations

A loss spike, plateau, or regression is rarely specific. Data, initialization,
optimizer settings, numerical precision, random variation, and implementation
errors can produce the same curve. With a novel method, there are also fewer
known-good results for eliminating these explanations.

Additional metrics can eliminate some causes but rarely identify which
execution relationship broke. Throughput, for example, shows that the job is
progressing, not that replicas agree or that the optimizer updates the intended
parameters.

Mixed precision, sharding, compilers, and fused kernels all change how the
authored model executes. An error in one runtime component may first appear as
two ranks disagreeing or as a model and optimizer referring to different
parameters. The system is therefore part of the algorithm being evaluated.
This ambiguity matters most when an experiment has no known-good result.

## Negative results are hardest to validate

When an established recipe stops working, there is a known-good result to
recover. The implementation becomes an obvious suspect, and the team has a
reason to keep debugging.

When a new architecture, objective, or training method underperforms,
researchers may reasonably conclude that the idea failed. The same observed
outcome—a run not worth continuing—can result from either a genuine negative
result or a silent implementation error.

Public bug reports are therefore survivorship-biased. We see the cases someone
investigated until they found doubled gradients or divergent replicas. We do
not see the experiments that were abandoned because their wrong results looked
reasonable.

We cannot prove how often this happens. We can require better evidence before
using a run to make a research decision.

Although exact loss values are rarely specifiable, we can state many execution
requirements. Parallelization should preserve intended gradients. Replicated
state should remain consistent. Optimizers should update the parameters
associated with their gradients. Required operations should execute in the
right context.

The next post examines why violations of these relationships are difficult to
localize, which patterns recur across training stacks, and where execution-level
checks such as [TrainCheck](https://github.com/OrderLab/TrainCheck) can help.

---

*This is the first post in a three-part series on trustworthy ML training. Next:
[Why ML Training Failures Are So Hard to
Localize](traincheck-in-practice.md).*
