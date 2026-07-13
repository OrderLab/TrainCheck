---
date: 2026-07-13
draft: true
slug: traincheck-catch-silent-errors
categories:
  - ML Reliability
  - Distributed Training
description: How TrainCheck uses execution evidence to detect silent errors during machine learning training.
---

# TrainCheck: Catch Silent Errors for More Dependable ML Training

Today’s machine learning training is expensive to run but remains difficult to
trust. The most dangerous failures are not always crashes. A training job can
keep GPUs busy, write logs, save checkpoints, and lower loss while still
updating the wrong state or executing the wrong training procedure. From the
outside, the run looks healthy.

**This creates a fundamental ambiguity when results disappoint: did the
algorithm fail, or did the implementation fail?** The same weak or unstable
metric may result from the data, model architecture, training configuration,
optimization process, or a silent implementation-level bug. Aggregate signals
can reveal that something went wrong, but often not what.

The cost extends beyond GPU time wasted by delayed detection. Engineers may
spend days tuning around a bug. Researchers may abandon a promising idea because
its implementation never worked correctly. In the worst case, an implementation
failure may be mistaken for evidence about the algorithm itself, turning a
software defect into an unsupported scientific conclusion.

 <!-- TODO: add a few
citations here.-->

This post aims to draw attention to a largely hidden layer of ML reliability.
Existing reliability efforts often focus on infrastructure failures, hardware
faults, and kernel correctness. Yet a training system can remain healthy at all
of these layers while still executing the wrong training logic.

We discuss how to detect silent training errors earlier through execution-level
validation. We introduce
- *Training Invariants*, relationships that should hold during correct training,
and
- *[TrainCheck](https://github.com/OrderLab/TrainCheck)*, an open-source tool that automatically infers and checks them.

<!-- Next blog post: Training reliability – The layers -->

## Case Studies: When Monitoring Turns into Guessing

The following cases illustrate a recurring problem in ML training: the visible
symptom appears in an aggregate metric, while the actual failure occurs much
deeper in the execution.

An X post by Jack Morris described a distributed data parallel (DDP) run whose
loss fell until roughly step 150, then rose without a corresponding increase in
gradient norm. Replies to the [original
question](https://x.com/jxmnop/status/1778436832075678100) proposed
learning-rate instability, initialization, regularization, and a missing
`optimizer.zero_grad()` call.

However, the actual root cause was not an algorithmic limitation, but an
implementation error. The code called the raw `nn.Module` instead of the DDP
wrapper. Gradients did not synchronize, so each GPU learned its own solution.
([Root-cause follow-up](https://x.com/jxmnop/status/1778520637193240892)).

Elana Simon encountered a PyTorch failure that initially looked like a
hyperparameter problem. Loss plateaued for her sparse autoencoder. Decoder
weights changed, while encoder weights stayed fixed despite receiving gradients.

In PyTorch versions before 2.4, `addcmul_` and `addcdiv_` could silently fail
when writing to non-contiguous outputs on the Metal Performance Shaders (MPS)
backend. ([Technical
investigation](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/))

Mayank Mishra reported incorrect Mamba-2 initialization in two
FlashLinearAttention paths. One path initialized the time-step bias parameter
with `torch.ones`. A distributed-tensor path skipped the intended
initialization. The project merged fixes for both paths, and Mishra reported a
substantial difference in training quality after correction. ([Public
report](https://x.com/MayankMish98/status/2026769614022259079), [layer
initialization fix](https://github.com/fla-org/flash-linear-attention/pull/739),
[distributed-tensor
fix](https://github.com/fla-org/flash-linear-attention/pull/753))

These failures arose in different parts of the training stack, but shared the
same diagnostic pattern. Aggregate metrics exposed only ambiguous symptoms. The
decisive evidence came from execution itself: a bypassed synchronization
wrapper, weights and optimizer state that failed to change, or initialization
operations that differed from the intended path.

## From Ambiguous Metrics to Direct Execution Evidence

Training is often treated as inherently difficult to validate because its
numerical outcomes are noisy and nondeterministic. Different seeds, data orders,
hardware, and kernels may produce different loss curves even when a run is
correct.

But much of this ambiguity comes from checking training at too high a level.
Aggregate metrics such as loss and accuracy compress a long execution into a
small number of outcomes. Many different failures can therefore produce the same
visible symptom.

At a lower level, training contains relationships that are far less ambiguous.
Replicated parameters should remain consistent across ranks. An optimizer should
update the parameters associated with the gradients it receives. Required
initialization operations should occur. These properties do not require
predicting the exact loss trajectory; they require checking whether the
execution preserves the relationships necessary for correct training.

The cases above illustrate this distinction. Their metrics were ambiguous, but
their execution was not. Each failure violated a concrete relationship that
should have held.

The cases above illustrate this distinction:

| Incident                   | Ambiguous symptom                    | Direct execution evidence                               | 
| -------------------------- | ------------------------------------ | ------------------------------------------------------- | 
| DDP wrapper bypass | Loss rose after initially decreasing | Replicated parameters diverged across ranks             | 
| MPS optimizer bug          | Loss plateaued | Encoder weights and Adam state did not change           | 
| Mamba-2 initialization bug | Training quality degraded            | Required initialization calls were missing or incorrect |

<!-- A training step is an ordered stream of events. Modules run, backward creates
gradients, optimizers read them, and parameters change. API calls have a start
and an end, so state can be compared before and after each call. Iteration,
stage, device, and rank identify comparable events. -->

In the [DDP incident](https://x.com/jxmnop/status/1778520637193240892),
corresponding replicated parameters should have remained equal across GPUs after
each update. Because the raw module bypassed DDP synchronization, the ranks
began updating independently. Comparing parameter hashes across ranks could have
exposed the divergence before it appeared as a rising loss near step 150.

Simon’s [MPS
incident](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)
violated a different relationship. After an Adam step with a nonzero encoder
gradient, the encoder weights and optimizer state should have changed. Instead,
both remained unchanged, providing direct evidence that the update had silently
failed.

The [Mamba-2 initialization
errors](https://x.com/MayankMish98/status/2026769614022259079) could have been
exposed even earlier. A correct construction trace should contain the intended
initialization operations and writes to the relevant parameters. The faulty
paths instead used different initialization calls or skipped required writes
entirely. These discrepancies were visible before training began. ([Layer
initialization fix](https://github.com/fla-org/flash-linear-attention/pull/739),
[distributed-tensor
fix](https://github.com/fla-org/flash-linear-attention/pull/753))

We call such expected relationships **training invariants**. An invariant may
state that corresponding values remain equal across ranks, that an optimizer
step changes a parameter, or that one execution event occurs before another. Its
applicability can depend on context: frozen parameters need not change, skipped
optimizer steps should not update state, active context managers may alter
behavior, and sharded tensors need not remain equal across ranks.

**Monitoring training invariants lets a checker report the first violated
relationship before its numerical effects accumulate into an ambiguous metric.**

## Introducing TrainCheck

[TrainCheck](https://github.com/OrderLab/TrainCheck) validates machine learning
training at the execution level. Instead of predicting what the loss curve
should look like, it checks whether a run preserves relationships observed
during correct training.

TrainCheck is an open-source system for validating machine learning training at
the execution level. Instead of predicting what the loss curve should look like,
it checks whether the run preserves relationships observed during correct
training.

TrainCheck begins with one or more short reference runs believed to be correct.
It records training events and selected program state, including module
execution, API calls, gradients, optimizer state, parameters, devices,
iterations, and distributed ranks. From these observations, it infers training
invariants: recurring relationships that characterize the reference executions.

An invariant may require corresponding values to remain equal across ranks, a
parameter to change during an optimizer step, one API call to occur before or
inside another, or an argument or output to follow a recurring pattern.
TrainCheck also records the context in which each relationship holds. For
example, a parameter should change only when it is trainable, receives a
gradient, and the optimizer step is not skipped.

TrainCheck then checks a target run against the inferred invariants. When an
applicable relationship no longer holds, it reports the first violation together
with the associated execution context. This provides more direct evidence than
waiting for the error to accumulate into a visible change in loss or accuracy.

The workflow has three stages:

1. Collect execution traces from reference runs.
2. Infer contextual training invariants from those traces.
3. Check a target run and report violated invariants.

This design avoids requiring users to manually specify every correctness rule.
It also avoids assuming that correct training must follow one exact numerical
trajectory. Instead, TrainCheck learns which lower-level relationships remain
stable across representative executions and checks those relationships where
they apply.

TrainCheck is designed to learn useful invariants from a small number of
reference runs rather than requiring a large training corpus. Many inferred
relationships can also transfer across related inputs, configurations,
pipelines, and software versions. This makes the invariants useful not only for
reproducing one execution, but also for validating new runs as training systems
evolve.

## BLOOM-176B: From Ten Days Undetected to One-Iteration Detection

A real incident during the training of BLOOM-176B shows what this workflow looks
like in practice and why execution-level validation matters at scale.

Tensor parallelism splits large matrix operations across GPUs. In the
Megatron-style implementation used for BLOOM, smaller LayerNorm parameters
remain replicated rather than sharded. Those replicated weights must stay
identical across tensor-parallel ranks.

During BLOOM-176B training, a bug in the BF16 optimizer’s gradient-clipping
logic caused clipping to run on only one tensor-parallel rank. The replicated
LayerNorm weights diverged, while loss and accuracy showed no immediate anomaly.
The problem remained undetected for ten days. ([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[OSDI ’25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![Historical BLOOM detection compared with the separate TrainCheck
reproduction](../../assets/blog/traincheck-launch/02-bloom-timeline.png)

To evaluate TrainCheck on this case, we reproduced the failure at smaller scale.
We inferred training invariants from the more mature FP16 implementation and
applied them to the faulty BF16 training job. The incorrect clipping logic was
triggered at iteration 2, and TrainCheck reported a cross-rank consistency
violation at iteration 3—one iteration after the error occurred.

The report identified replicated LayerNorm weights that differed across
tensor-parallel ranks. Inspecting the associated trace exposed the root-cause
behavior: gradient clipping had run inconsistently, causing different ranks to
apply different updates. ([OSDI ’25 paper, Sections 3.2 and
5.1](https://www.usenix.org/system/files/osdi25-jiang.pdf))

## AC-2665: From a Flat Loss to the Root Cause

The BLOOM case shows how TrainCheck can detect a silent error shortly after it
occurs. AC-2665 illustrates a different benefit: several related invariant
violations can work together to narrow an ambiguous symptom to a concrete root
cause.

In Accelerate issue 2665, a two-GPU Fully Sharded Data Parallel (FSDP) run
completed training steps while its loss remained constant. The reporter had
already confirmed that the same model learned correctly on a single GPU.
([Public issue](https://github.com/huggingface/accelerate/issues/2665))

We checked the failing run using training invariants inferred from an official
graph convolutional network example. TrainCheck reported several related
violations:

- Parameters stored in the optimizer did not receive gradients. 
- The optimizer’s parameters had no gradient state for `optimizer.zero_grad()` to clear. 
- `optimizer.step()` did not change the model parameters. 
- The step invoked none of the expected mathematical operations on those parameters.

Together, these violations pointed to a mismatch between the parameters used by
the prepared model and those stored in the optimizer. Inspection confirmed the
diagnosis: FSDP wrapping had created flattened parameters, while the optimizer
still referenced the original parameters. ([OSDI ’25 paper, Section
5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf), [root-cause
follow-up](https://github.com/huggingface/accelerate/issues/3256))

![Three example TrainCheck violations narrow AC-2665 to a model-optimizer
parameter mismatch](../../assets/blog/traincheck-launch/05-ac2665-diagnosis.png)

## How to Use TrainCheck

A minimal offline workflow uses four commands:

```bash
pip install traincheck

traincheck-collect \
  --pyscript reference.py \
  --models-to-track model \
  --output-dir reference_trace

traincheck-infer -f reference_trace -o invariants.json

traincheck-collect \
  --pyscript target.py \
  --models-to-track model \
  --output-dir target_trace

traincheck-onlinecheck -f target_trace -i invariants.json
```

For more details on checking and its integrations, please see [Checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md).

TrainCheck supports relationships over variable consistency, contained events,
API order, arguments, and API outputs. Preconditions allow expected exceptions,
such as frozen layers that remain unchanged or sharded tensors that differ
across ranks. ([Training invariant
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md),
[OSDI ’25 paper, Sections 3.1 and
3.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

## How Well Does TrainCheck Work?

We evaluated TrainCheck around four practical questions: what it can detect, how
much reference data it needs, what checking costs, and how it fits into existing
workflows.

### 1. What can TrainCheck detect?

**Takeaway: With representative reference runs, TrainCheck detects a broad range
of silent training errors and usually reports them immediately after they
occur.**

In the OSDI ’25 evaluation, we reproduced 20 real-world silent training errors.
TrainCheck detected 18. Every detected error was reported no later than one
training iteration after its trigger. The reports identified the exact root
cause in 10 cases and localized the failure close to the root cause in the
remaining eight. ([OSDI ’25 paper, Section
5.1](https://www.usenix.org/system/files/osdi25-jiang.pdf))

TrainCheck also uncovered six previously unknown bugs in popular training
libraries. Maintainers confirmed all six, and three were fixed before
publication. ([OSDI ’25 paper, Section
5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![TrainCheck OSDI 2025 evaluation
scorecard](../../assets/blog/traincheck-launch/04-evaluation-scorecard.png)


### 2. How much reference data does TrainCheck need?

**Takeaway: A small, diverse set of reference runs is often sufficient, and many
invariants can be reused across related environments.**

Across the 18 detected errors, two sampled reference programs achieved 91% mean
invariant coverage across configurations and 82% across pipelines. Five randomly
sampled tutorial programs achieved 76% mean coverage. These results suggest that
representative invariants can often generalize beyond the executions from which
they were inferred. ([OSDI ’25 paper, Section
5.5](https://www.usenix.org/system/files/osdi25-jiang.pdf))

In a precision experiment covering 63 programs without known bugs,
false-positive rates remained below 2% when invariants were inferred from five
or six input programs, and below 5% with two or three. Both the number and
diversity of reference runs mattered. ([OSDI ’25 paper, Section
5.3](https://www.usenix.org/system/files/osdi25-jiang.pdf))

Training invariants can also remain valid across software versions. In one
scoped experiment, we inferred invariants from an official graph convolutional
network workload under PyTorch 2.2.2. When applied under PyTorch 2.5.1, 94.2% of
those invariants remained valid and applicable. This result is specific to that
workload and version pair, but it illustrates the potential to reuse invariants
as training environments evolve. ([Cross-version reproduction
guide](https://github.com/OrderLab/TrainCheck/blob/main/docs/ae-eval-s5.3-transferability.md))

### 3. What does checking cost?

**Takeaway: Selective checking adds low overhead for most evaluated workloads,
although small and CPU-sensitive programs can experience larger slowdowns.**

The OSDI ’25 evaluation measured less than 2% overhead for most workloads using
selective instrumentation. The largest slowdown was 1.6× on a small,
CPU-sensitive workload. A newer benchmark across ten repository workloads
reports a median slowdown of 1.048× and a maximum central estimate of 1.362×.
([OSDI ’25 paper, Section
5.6](https://www.usenix.org/system/files/osdi25-jiang.pdf), [current benchmark
data](https://github.com/OrderLab/TrainCheck/blob/main/docs/assets/csv/overhead_e2e.csv))

![Current selective-checking overhead across ten repository
workloads](../../assets/blog/traincheck-launch/06-current-overhead.png)

### 4. How can TrainCheck fit into existing workflows?

TrainCheck instruments PyTorch eager-mode execution and has been applied to
workloads built with PyTorch, DeepSpeed, Megatron, Hugging Face Transformers,
and Accelerate. It can export checking results through OpenTelemetry and
integrate with experiment-tracking tools such as Weights & Biases, MLflow, and
TensorBoard.

TrainCheck is available today for experimentation, debugging, and validation in
supported training pipelines. It is not yet a universal drop-in production
monitor: compatibility with additional frameworks, compiled execution, and
larger-scale training jobs remains active work. We are also developing an
invariant hub so that teams can reuse validated invariants rather than inferring
every set from scratch.

## Practical Uses

TrainCheck can support several stages of the training workflow.

Teams can run inferred training invariants alongside existing dashboards to
detect silent execution errors during training. The checker can export summary
metrics through optional integrations with Weights & Biases, MLflow,
TensorBoard, and OpenTelemetry. ([Checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md))

Researchers can also use TrainCheck interactively for diagnosis. They can infer
expected behavior from a known-good run, check a suspicious execution, and
inspect the first violated relationships. As the AC-2665 case shows, several
related violations can narrow a broad symptom to a concrete model-optimizer
mismatch.

The same workflow can serve as a behavioral regression check when teams change
framework versions, distributed configurations, hardware, or training
infrastructure. More broadly, execution-level validation helps separate
implementation failures from failed research hypotheses, giving researchers
stronger evidence for interpreting experimental results.

## Current Scope

TrainCheck currently supports Python and PyTorch eager-mode execution, with
`torch.compile` disabled. It uses tensor hashes for equality checks rather than
fine-grained numerical comparison. The online checker reports violations, while
users or external policies decide whether a run should be stopped. ([OSDI ’25
paper, Section 6](https://www.usenix.org/system/files/osdi25-jiang.pdf), [usage
guide](https://github.com/OrderLab/TrainCheck/blob/main/docs/usage-guide.md))

The quality of inferred invariants depends on the representativeness of the
reference runs. Narrow references may miss specialized behavior or produce noisy
invariants. TrainCheck is therefore designed to complement tests, dashboards,
benchmarks, and human judgment. ([OSDI ’25 paper, Sections 5.3 and
5.5](https://www.usenix.org/system/files/osdi25-jiang.pdf), [inference
guidance](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md))

We are continuing to expand compatibility, evaluate larger training workloads,
and explore how execution-level validation can support reinforcement learning,
inference pipelines, and autonomous research agents.

## Try TrainCheck

Silent training errors preserve the appearance of progress. TrainCheck adds a
missing layer of evidence: whether the training process itself behaved as
intended.

- [GitHub](https://github.com/OrderLab/TrainCheck) 
- [Documentation](https://orderlab.io/TrainCheck/) 
- [Five-Minute Tutorial](https://orderlab.io/TrainCheck/5-min-tutorial/) 
- [OSDI ’25 Paper](https://www.usenix.org/conference/osdi25/presentation/jiang) 
- [PyPI](https://pypi.org/project/traincheck/)

Try TrainCheck on one of your training pipelines, and let us know which silent
errors, frameworks, or configurations you would most like it to support.