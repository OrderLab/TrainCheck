---
date: 2026-07-27
draft: true
slug: why-ml-training-failures-are-hard-to-localize
categories:
  - ML Reliability
  - Distributed Training
  - TrainCheck
description: Why a faulty operation may first corrupt state elsewhere, which execution relationships repeatedly break, and how runtime checks narrow the search.
---

# Why ML Training Failures Are So Hard to Localize

During BLOOM-176B training, LayerNorm weights intended to be replicated diverged
across tensor-parallel ranks. Loss and accuracy showed no immediate anomaly.
The inconsistency went undetected for ten days.

The root cause was in the BF16 optimizer's gradient-clipping logic. Clipping
executed on only one tensor-parallel rank, so replicated weights received
different updates.
([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[OSDI '25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

## The same relationships break across stacks

The BLOOM bug combined two recurring failures: replicated weights disagreed, and
clipping did not run on every relevant rank. Across frameworks and kernels, five
kinds of relationship recur:

- **Agreement:** state intended to be replicated differs across ranks or
  implementations.
- **Ownership:** gradients, parameters, and optimizer state refer to different
  logical objects.
- **State transition:** an operation that should update state does not, or an
  update occurs where none was expected.
- **Ordering and coverage:** a required operation runs on the wrong rank, in the
  wrong order, or only on part of the intended state.
- **Identity across components:** two services disagree about a model, batch,
  tokenization, mask, or policy version they treat as shared.

These relationships can be checked without predicting loss.

## Principles for monitoring training execution

Teams already use assertions, differential tests, per-rank logs, and execution
traces. Each has limits. Assertions require advance specifications. Exact
comparisons are brittle under stochasticity and legitimate implementation
differences. Raw logs and traces may be too large to inspect.

These limitations suggest five principles for useful training monitoring:

1. **Check relationships, not outcomes.** Validate agreement, ownership,
   transitions, ordering, and identity without predicting exact metrics.
2. **Record execution context.** Keep the step, stage, rank, object identity, and
   preconditions needed to distinguish bugs from valid behavior.
3. **Detect near the trigger.** Check before downstream metrics move.
4. **Retain the operations around a violation.** Report the broken property and
   the relevant preceding events.
5. **Instrument selectively.** Gather enough state without dominating training
   cost.

The goal is narrower than proving a run correct: check whether specific
execution relationships held and retain enough context to localize a violation.

## One implementation: TrainCheck

[TrainCheck](https://github.com/OrderLab/TrainCheck) implements this approach by
learning contextual invariants from reference executions and checking a target
run:

1. Collect selected execution events and state from short runs believed to be
   correct.
2. Infer recurring relationships together with their preconditions.
3. Check a target trace and surface invariant violations with the execution
   context available in that trace.

![From reference runs to a concrete invariant
violation](../../assets/blog/traincheck-launch/03-workflow.png)

The inferred invariants cover variable consistency, state changes, required
operations, API order, arguments, and outputs. Preconditions limit each invariant
to the context in which it applies—for example, consistency checks should cover
replicated, not sharded, parameters.
([Inference
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md),
[checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md))

For BLOOM, we checked a smaller BF16 reproduction using invariants inferred from
Megatron-DeepSpeed GPT pretraining reference runs. The bug triggered at iteration
2; TrainCheck reported diverged LayerNorm weights at iteration 3. The report
established the broken relationship; the incident's known root cause was
inconsistent clipping in the optimizer path. ([OSDI '25 paper, Sections 3.2 and
5.1](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![Historical BLOOM detection compared with the separate TrainCheck
reproduction](../../assets/blog/traincheck-launch/02-bloom-timeline.png)

This reproduction does not show that TrainCheck would have reduced the original
ten-day investigation to one iteration. It shows that the weights diverged
before the loss or accuracy changed.

## What the current evidence establishes

In the OSDI '25 evaluation, we reproduced 20 real-world silent training errors.
TrainCheck detected 18, each no later than one training iteration after its
trigger. Its reports identified the exact root cause in 10 cases and localized
the failure close to it in the other eight. It also uncovered six previously
unknown bugs in popular training libraries; maintainers confirmed all six, and
three had been fixed by publication. ([OSDI '25 paper, Sections
5.1–5.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![TrainCheck OSDI 2025 evaluation
scorecard](../../assets/blog/traincheck-launch/04-evaluation-scorecard.png)

The reference runs determine what TrainCheck can learn. Across 63 programs
without known bugs, false-positive rates stayed below 2% when five or six
programs supplied the inference traces and below 5% with only two or three. Both
the number and diversity of references mattered. Selective instrumentation added
less than 2% overhead for most workloads in the paper, although the toy GCN
workload slowed by 1.6×.
([OSDI '25 paper, Sections 5.3 and
5.6](https://www.usenix.org/system/files/osdi25-jiang.pdf))

TrainCheck is not a correctness proof. A violation may indicate a bug or valid
behavior absent from the references, so it still requires interpretation. No
violation means only that the inferred relationships held.

## Trying it on a training change

A minimal offline workflow uses a reference run and a target run:

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
  --invariants invariants.json \
  --output-dir target_trace

traincheck-check -f target_trace -i invariants.json
```

TrainCheck currently instruments PyTorch eager-mode execution and has been used
with workloads built on DeepSpeed, Megatron, Hugging Face Transformers, and
Accelerate. Checker results can be sent to Weights & Biases and MLflow.

Supporting compiled execution, additional frameworks, cross-service training,
and production-scale deployments remains future work. Those limits motivate the
final post's broader question: what evidence should results carry as ML
workflows span more components and automated decisions?

- [Five-Minute Tutorial](https://orderlab.io/TrainCheck/5-min-tutorial/)
- [OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/jiang)
- [GitHub](https://github.com/OrderLab/TrainCheck)

---

*This is the second post in a three-part series on trustworthy ML training.
Previous: [ML Training Can Be Wrong Even When the Loss Goes
Down](training-reliability.md). Next: [We're Scaling Training Faster Than We're
Scaling Trust](scaling-training-trust.md).*
