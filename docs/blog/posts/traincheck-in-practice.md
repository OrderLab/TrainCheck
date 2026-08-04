---
date: 2026-07-27
draft: true
slug: traincheck-catching-training-bugs
categories:
  - ML Reliability
  - Distributed Training
  - TrainCheck
description: How TrainCheck infers execution-level invariants, catches silent training errors, and helps localize their root causes.
---

# TrainCheck: Catching Training Bugs Before the Loss Curve Does

During the training of BLOOM-176B, replicated LayerNorm weights began to differ
across tensor-parallel ranks. Loss and accuracy showed no immediate anomaly.
The failure remained undetected for ten days.

The root cause was a bug in the BF16 optimizer's gradient-clipping logic.
Clipping ran on only one tensor-parallel rank, so nominally replicated weights
received different updates. The training job was alive. Its metrics appeared
ordinary. Its distributed state was already inconsistent. ([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[OSDI '25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![Historical BLOOM detection compared with the separate TrainCheck
reproduction](../../assets/blog/traincheck-launch/02-bloom-timeline.png)

This is the kind of gap [training invariants](training-invariants.md) are meant
to close. The loss trajectory was ambiguous, but the violated relationship was
not: replicated weights should remain consistent.

## From reference executions to a violation

[TrainCheck](https://github.com/OrderLab/TrainCheck) is an open-source system
for validating ML training at the execution level. It uses one or more short
reference runs believed to be correct, records selected execution events and
state, and infers contextual relationships that recur across them.

It then checks a target run against those inferred invariants. When an
applicable relationship stops holding, TrainCheck reports the first violation
with its API, variable, iteration, stage, device, and rank context.

The offline workflow is four commands:

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

TrainCheck can reason about variable consistency, state changes, contained
events, API order, arguments, and outputs. Preconditions capture exceptions
such as frozen layers, skipped optimizer steps, and sharded tensors. ([Inference
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md),
[checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md))

## BLOOM: detect the divergence, then inspect its cause

To evaluate the BLOOM failure, we reproduced it at smaller scale. We inferred
invariants from the more mature FP16 implementation and applied them to the
faulty BF16 job. The incorrect clipping behavior was triggered at iteration 2.
TrainCheck reported a cross-rank consistency violation at iteration 3—one
iteration after the error occurred.

The report identified replicated LayerNorm weights that differed across tensor
parallel ranks. Inspecting the associated trace exposed the underlying behavior:
gradient clipping had executed inconsistently, causing the ranks to apply
different updates. ([OSDI '25 paper, Sections 3.2 and
5.1](https://www.usenix.org/system/files/osdi25-jiang.pdf))

The important comparison is not ten days versus one iteration as a claim about
the same production environment; our experiment was a separate reproduction.
It demonstrates that the inconsistency was observable at the execution level
long before it needed to become a visible loss anomaly.

## AC-2665: several violations form a diagnosis

A public Accelerate issue shows a different use. A two-GPU Fully Sharded Data
Parallel run completed its training steps while loss remained constant. The
same model learned correctly on one GPU. ([Accelerate issue
2665](https://github.com/huggingface/accelerate/issues/2665))

We checked the failing run using invariants inferred from an official graph
convolutional network example. TrainCheck reported several related violations:

- Parameters stored in the optimizer did not receive gradients.
- Those parameters had no gradient state for `optimizer.zero_grad()` to clear.
- `optimizer.step()` did not change the model parameters.
- The step invoked none of the expected mathematical operations on them.

Together, these reports suggested a mismatch between the parameters used by the
prepared model and those held by the optimizer. Inspection confirmed it: FSDP
wrapping had created flattened parameters, while the optimizer still referenced
the originals. ([OSDI '25 paper, Section
5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf), [root-cause
follow-up](https://github.com/huggingface/accelerate/issues/3256))

![Three TrainCheck violations narrow AC-2665 to a model-optimizer parameter
mismatch](../../assets/blog/traincheck-launch/05-ac2665-diagnosis.png)

Here the value was not earlier detection. The flat loss had already announced a
problem. The value was turning a symptom with many possible explanations into a
small, testable hypothesis about execution.

## How often does this work?

In the OSDI '25 evaluation, we reproduced 20 real-world silent training errors.
TrainCheck detected 18, each no later than one training iteration after its
trigger. Reports identified the exact root cause in 10 cases and localized the
failure close to it in the other eight. TrainCheck also uncovered six previously
unknown bugs in popular training libraries; maintainers confirmed all six, and
three had been fixed by publication. ([OSDI '25 paper, Sections
5.1–5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![TrainCheck OSDI 2025 evaluation
scorecard](../../assets/blog/traincheck-launch/04-evaluation-scorecard.png)

Reference quality matters. In a precision experiment covering 63 programs
without known bugs, false-positive rates stayed below 2% when invariants came
from five or six input programs, and below 5% with two or three. Both the number
and diversity of references helped. Across the 18 detected errors, two sampled
reference programs achieved 91% mean invariant coverage across configurations
and 82% across pipelines. ([OSDI '25 paper, Sections 5.3 and
5.5](https://www.usenix.org/system/files/osdi25-jiang.pdf))

Selective instrumentation added less than 2% overhead for most evaluated
workloads, although a small CPU-sensitive program slowed by 1.6×. A newer
ten-workload repository benchmark reports a median slowdown of 1.048× and a
maximum central estimate of 1.362×. ([OSDI '25 paper, Section
5.6](https://www.usenix.org/system/files/osdi25-jiang.pdf), [current benchmark
data](https://github.com/OrderLab/TrainCheck/blob/main/docs/assets/csv/overhead_e2e.csv))

![Selective-checking overhead across ten repository
workloads](../../assets/blog/traincheck-launch/06-current-overhead.png)

## Where it fits—and where it does not yet

TrainCheck currently instruments PyTorch eager-mode execution and has been used
with PyTorch, DeepSpeed, Megatron, Hugging Face Transformers, and Accelerate. It
can export checking results through OpenTelemetry and experiment-tracking tools
including Weights & Biases, MLflow, and TensorBoard.

It is most useful in three situations:

1. **Diagnosing a suspicious run** by comparing it with a known-good or closely
   related execution.
2. **Regression-checking a pipeline change** such as a framework upgrade,
   precision change, distributed configuration, optimizer, or hardware move.
3. **Guarding an expensive run** with selected, high-confidence invariants
   alongside existing dashboards.

TrainCheck is not yet a universal production monitor. Compiled execution,
additional frameworks, and larger-scale deployments remain active work. Its
results are only as useful as the reference evidence and invariant context
behind them. It reports violations; people or external policies decide whether
to inspect, continue, or stop a run.

Those limitations lead to the questions we most want practitioners to answer:
Which silent failures consume your debugging time? Do you have a trusted
reference execution? Would you check during development, before a major run, or
continuously? And which unsupported part of your stack prevents you from trying
it?

- [GitHub](https://github.com/OrderLab/TrainCheck)
- [Documentation](https://orderlab.io/TrainCheck/)
- [Five-Minute Tutorial](https://orderlab.io/TrainCheck/5-min-tutorial/)
- [OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/jiang)
- [PyPI](https://pypi.org/project/traincheck/)

---

*This is the third post in a four-part series on trustworthy ML training.
Previous: [We Cannot Predict the Loss Curve. But We Can Still Check the
Training](training-invariants.md). Next: [We're Scaling Training Faster Than
We're Scaling Trust](scaling-training-trust.md).*
