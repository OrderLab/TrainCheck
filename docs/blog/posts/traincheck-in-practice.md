---
date: 2026-07-27
draft: true
slug: why-ml-training-failures-are-hard-to-localize
categories:
  - ML Reliability
  - Distributed Training
  - TrainCheck
description: Why ML training symptoms are often far removed from their root causes, which failure patterns recur, and how execution-level checks can narrow the search.
---

# Why ML Training Failures Are So Hard to Localize

During the training of BLOOM-176B, LayerNorm weights that were supposed to be
replicated began to differ across tensor-parallel ranks. Loss and accuracy
showed no immediate anomaly. The inconsistency remained undetected for ten
days.

The root cause was in the BF16 optimizer's gradient-clipping logic. Clipping
executed on only one tensor-parallel rank, so nominally identical weights
received different updates. The visible symptom was inconsistent model state.
The cause was a rank-dependent operation earlier in the optimizer path.
([BLOOM training
chronicle](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing),
[OSDI '25 paper, Sections 1 and
2.2](https://www.usenix.org/system/files/osdi25-jiang.pdf))

That distance between symptom and cause is the central difficulty in debugging
ML training. The case is useful not because of when it happened, but because
the same structure appears whenever frameworks transform a local training
program into a distributed one.

## One symptom supports too many explanations

A loss spike, plateau, or regression is rarely specific. Data, architecture,
initialization, optimizer settings, numerical precision, random variation, and
implementation errors can all produce similar curves. The more novel the
training method, the fewer known-good expectations exist to eliminate these
hypotheses.

Additional metrics help, but they remain lossy summaries. A gradient norm can
show that optimization changed without showing whether replicas synchronized.
An update norm says little if the optimizer owns different parameters from
those used in the forward pass. A healthy throughput graph says that the job is
moving, not that it is executing the intended algorithm.

Training stacks also move the root cause away from the code a researcher wrote.
Mixed precision inserts casts and master weights. FSDP replaces and shards
parameters. Pipeline and tensor parallelism divide operations across ranks.
Compilers and fused kernels replace many visible operations with another
program. Each transformation may be locally reasonable while their composition
breaks a training assumption.

Finally, many failures are relational. Nothing looks wrong on one rank in
isolation; the error is that two ranks disagree. A model and optimizer may each
contain valid parameters; the error is that they are not the same parameters.
A rollout worker and learner may each hold a valid policy; the error is that
they disagree about which version generated the data.

## The failure patterns are more stable than the stacks

Specific frameworks and kernels change quickly. The relationships they violate
are more repetitive:

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

These patterns do not tell us the correct loss at step 10,000. They do tell us
what kind of evidence would narrow a localization problem. In the BLOOM case,
the useful observations were not “loss should equal X,” but “replicated weights
should agree” and “gradient clipping should execute consistently across the
relevant ranks.”

## What a useful solution needs to preserve

Teams already use several forms of execution evidence. Explicit assertions are
precise when a failure is anticipated. Differential tests compare a new
configuration with a trusted one. Per-rank logging exposes distributed state.
Execution traces retain the events needed for a postmortem.

Each approach trades coverage for effort. Assertions require people to specify
the property in advance. Exact differential comparison is brittle under
stochasticity, scaling, and legitimate implementation differences. Logs and
traces can contain the answer while remaining too large to inspect manually.

A useful checker therefore needs to operate between exact-output comparison and
generic anomaly detection. It should compare relationships rather than exact
values, attach the context in which those relationships should hold, and lead
from a violated property back to the operations that produced it.

## TrainCheck is one attempt at that design

[TrainCheck](https://github.com/OrderLab/TrainCheck) learns contextual training
invariants from reference executions and checks them against a target run. Its
workflow has three stages:

1. Collect selected execution events and state from short runs believed to be
   correct.
2. Infer recurring relationships together with their preconditions.
3. Report the first violation with its API, variable, iteration, stage, device,
   and rank context.

![From reference runs to a concrete invariant
violation](../../assets/blog/traincheck-launch/03-workflow.png)

The inferred invariants cover variable consistency, state changes, contained
events, API order, arguments, and outputs. Preconditions describe legitimate
exceptions such as frozen layers, skipped optimizer steps, and sharded tensors.
([Inference
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/infer.md),
[checking
documentation](https://github.com/OrderLab/TrainCheck/blob/main/docs/check.md))

For the BLOOM bug, we used the more mature FP16 implementation as a reference
and checked a smaller-scale reproduction of the faulty BF16 job. The bug
triggered at iteration 2. At iteration 3, TrainCheck reported that replicated
LayerNorm weights had diverged. The associated trace showed that gradient
clipping had executed inconsistently across ranks. ([OSDI '25 paper, Sections
3.2 and 5.1](https://www.usenix.org/system/files/osdi25-jiang.pdf))

![Historical BLOOM detection compared with the separate TrainCheck
reproduction](../../assets/blog/traincheck-launch/02-bloom-timeline.png)

This was a separate reproduction, not a claim that the original production run
could simply have replaced ten days with one iteration. It shows that the
violated relationship existed long before a top-level metric exposed it.

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
without known bugs, false-positive rates stayed below 2% with five or six input
programs and below 5% with two or three. Both the number and diversity of
references mattered. Selective instrumentation added less than 2% overhead for
most workloads in the paper, although one small CPU-sensitive program slowed by
1.6×. ([OSDI '25 paper, Sections 5.3 and
5.6](https://www.usenix.org/system/files/osdi25-jiang.pdf))

These results do not make TrainCheck a correctness proof. A violation may be a
bug, an unrepresented valid behavior, or a real difference that needs human
interpretation. No violation means only that the inferred relationships held.

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
  --output-dir target_trace

traincheck-onlinecheck -f target_trace -i invariants.json
```

TrainCheck currently instruments PyTorch eager-mode execution and has been used
with PyTorch, DeepSpeed, Megatron, Hugging Face Transformers, and Accelerate. It
can export results through OpenTelemetry, Weights & Biases, MLflow, and
TensorBoard.

Compiled execution, additional frameworks, cross-service training, and
larger-scale deployments remain open work. The final post asks where this kind
of validation is most useful next: large-scale pretraining, RL, or more
autonomous experimentation.

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
