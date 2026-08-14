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

Loss is often the wrong place to ask whether training executed as intended. A
curve can tell us that optimization behaved differently; it rarely tells us
whether replicas diverged, an optimizer updated detached parameters, or an
initialization path silently changed.

[TrainCheck](https://github.com/OrderLab/TrainCheck) checks those execution
relationships directly. It learns contextual invariants from reference runs,
then reports when a target run violates them. The bet is that exact training
outcomes are hard to predict, but many relationships required to produce a
meaningful outcome are not.

## From reference runs to a concrete violation

TrainCheck's workflow has three stages:

1. Collect selected execution events and state from short reference runs
   believed to be correct.
2. Infer contextual training invariants—relationships that recur when their
   preconditions hold.
3. Check a target run and report the first violated relationship with its API,
   variable, iteration, stage, device, and rank context.

![From reference runs to a concrete invariant
violation](../../assets/blog/traincheck-launch/03-workflow.png)

The invariants cover variable consistency, state changes, contained events, API
order, arguments, and outputs. Preconditions capture expected exceptions such
as frozen layers, skipped optimizer steps, and sharded tensors. ([Inference
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md),
[checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md))

## AC-2665: from a flat loss to a testable diagnosis

A public Accelerate issue demonstrates a different use. A two-GPU Fully Sharded
Data Parallel run completed its training steps while loss remained constant.
The same model learned correctly on one GPU. ([Accelerate issue
2665](https://github.com/huggingface/accelerate/issues/2665))

We checked the failing run using invariants inferred from an official graph
convolutional network example. TrainCheck reported that:

- parameters stored in the optimizer did not receive gradients;
- `optimizer.step()` did not change the model parameters; and
- the step invoked none of the expected mathematical operations on them.

Together, the violations suggested that the prepared model and optimizer held
different parameters. Inspection confirmed it: FSDP wrapping had created
flattened parameters, while the optimizer still referenced the originals.
([OSDI '25 paper, Section
5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf), [root-cause
follow-up](https://github.com/huggingface/accelerate/issues/3256))

![TrainCheck violations narrow AC-2665 to a model-optimizer parameter
mismatch](../../assets/blog/traincheck-launch/05-ac2665-diagnosis.png)

Here, the flat loss had already revealed a problem. TrainCheck's value was
reducing many possible explanations to a small hypothesis about the execution.

## How often does this generalize?

In the OSDI '25 evaluation, we reproduced 20 real-world silent training errors.
TrainCheck detected 18, each no later than one training iteration after its
trigger. Its reports identified the exact root cause in 10 cases and localized
the failure close to it in the other eight. TrainCheck also uncovered six
previously unknown bugs in popular training libraries; maintainers confirmed
all six, and three had been fixed by publication. ([OSDI '25 paper, Sections
5.1–5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![TrainCheck OSDI 2025 evaluation
scorecard](../../assets/blog/traincheck-launch/04-evaluation-scorecard.png)

The reference runs determine which invariants TrainCheck learns. In an
experiment covering 63 programs without known bugs, false-positive rates stayed
below 2% with five or six input programs and below 5% with two or three. Both
the number and diversity of references mattered. ([OSDI '25 paper, Section
5.3](https://www.usenix.org/system/files/osdi25-jiang.pdf))

Selective instrumentation added less than 2% overhead for most workloads in the
paper, although one small CPU-sensitive program slowed by 1.6×. A newer
ten-workload repository benchmark reports a median slowdown of 1.048× and a
highest central estimate of approximately 1.36×. ([OSDI '25 paper, Section
5.6](https://www.usenix.org/system/files/osdi25-jiang.pdf), [current benchmark
data](https://github.com/OrderLab/TrainCheck/blob/main/docs/assets/csv/overhead_e2e.csv))

## Where TrainCheck fits

TrainCheck is useful when a team has some evidence of correct behavior and
wants to know whether a new execution preserves it:

1. **Diagnose a suspicious run** by comparing it with a known-good or closely
   related execution.
2. **Regression-check a pipeline change** such as a framework upgrade,
   precision change, distributed configuration, optimizer, or hardware move.
3. **Guard an expensive run** with selected, high-confidence invariants.

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

TrainCheck currently instruments PyTorch eager-mode execution and has been used
with PyTorch, DeepSpeed, Megatron, Hugging Face Transformers, and Accelerate. It
can export results through OpenTelemetry, Weights & Biases, MLflow, and
TensorBoard.

It is not yet a universal production checker. Compiled execution, additional
frameworks, and larger-scale deployments remain active work. Its results depend
on representative references and correctly inferred preconditions. TrainCheck
reports violations; people or external policies decide whether to inspect,
continue, or stop a run.

We would especially like to hear which silent failures consume the most
debugging time, when teams would run this kind of check, and which unsupported
part of the stack currently prevents adoption.

- [GitHub](https://github.com/OrderLab/TrainCheck)
- [Documentation](https://orderlab.io/TrainCheck/)
- [Five-Minute Tutorial](https://orderlab.io/TrainCheck/5-min-tutorial/)
- [OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/jiang)
- [PyPI](https://pypi.org/project/traincheck/)

---

*This is the second post in a three-part series on trustworthy ML training.
Previous: [ML Training Can Be Wrong Even When the Loss Goes
Down](training-reliability.md). Next: [We're Scaling Training Faster Than We're
Scaling Trust](scaling-training-trust.md).*
