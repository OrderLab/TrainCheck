---
date: 2026-08-03
draft: true
slug: scaling-training-faster-than-trust
categories:
  - ML Reliability
  - Reinforcement Learning
  - Distributed Training
description: As training becomes larger, more interconnected, and more autonomous, what evidence will we need before we trust its results?
---

# We're Scaling Training Faster Than We're Scaling Trust

Pretraining and reinforcement learning create different reliability problems.
Pretraining concentrates risk inside one enormous, repeated computation. RL
spreads correctness across changing policies, rollouts, rewards, learners, and
inference systems.

Both can produce plausible top-level metrics while executing the wrong training
procedure. Both are also becoming harder to inspect manually as experiments
grow larger and faster.

The open question is not whether training needs stronger validation. It is which
failures matter most, which relationships we can check, and where such checks
belong in real ML workflows.

## Pretraining concentrates the cost of one silent error

Large-scale pretraining presents the clearest economic case for early
detection. Its training loop is repetitive and relatively structured, but the
system beneath it combines data, tensor, and pipeline parallelism with optimizer
sharding, mixed precision, checkpointing, fused kernels, and custom
infrastructure.

A small inconsistency can propagate through many devices and checkpoints before
a top-level metric makes it undeniable. By then, the team must decide whether
to continue, roll back, restart, or accept a checkpoint whose state is not fully
trusted.

Pretraining also has properties that make execution validation plausible. Teams
often have smaller-scale runs, earlier software versions, alternative precision
implementations, or known-good configurations. Many expected relationships are
concrete: replicas should stay consistent, optimizer state should change with
its parameters, and pipeline stages should run in the intended order.

The adoption problem is equally concrete. Organizations operating at this scale
are few, their stacks are heavily customized, and the cost of instrumentation
interfering with a production run is high. A useful checker must work with
compiled and fused execution, transfer expectations across scales, and justify
which checks remain enabled throughout a long run.

## RL moves correctness across component boundaries

In modern RL, correctness no longer lives inside one
forward-backward-update loop. A system may couple rollout workers, inference
engines, reward and reference models, experience storage, policy learners,
distributed orchestration, and evaluation.

Each component can work locally while their relationships are wrong. A rollout
worker may use a stale policy. The learner and inference engine may tokenize or
mask inputs differently. Rewards may be associated with the wrong samples.
Versioned data may arrive out of order. An update may succeed locally but never
reach the workers expected to use it.

This is already observable in current systems. In July 2025, verl users reported
reward collapse after enabling a fused linear cross-entropy kernel. The fused
path returned log probabilities that disagreed with the unfused PyTorch path.
The investigation traced the remaining discrepancy to a Triton kernel that
assumed particular vocabulary-size divisibility; the affected model's
vocabulary violated that assumption. The optimization did not merely run the
same RL computation faster. It changed values used by the algorithm. ([verl
report](https://github.com/verl-project/verl/issues/2656), [root-cause
analysis](https://github.com/verl-project/verl/issues/2899))

Final reward and benchmark scores compress these interactions into a few
numbers. When they disappoint, the explanations range from “the algorithm does
not work” to “one service used yesterday's weights.” This is the ambiguity of a
conventional training loop extended across time, processes, and system
boundaries.

RL may therefore have a larger unexplored need for execution validation. It is
also the harder setting. What counts as a reference run when the environment
and policy co-evolve? How much policy staleness is acceptable? Can a checker
distinguish bounded delay from a stuck worker? Can it validate causal
relationships across traces produced by different services?

The right abstraction may need to extend beyond invariants inside one Python
process. It may need to describe data lineage, policy versions, causal ordering,
and contracts between rollout, reward, inference, and learning.

## Autonomous research makes invalid results propagate faster

Research agents increasingly launch experiments, inspect results, form
hypotheses, and decide what to try next. An implementation error in this loop
does not only waste one run. It can make a promising method look weak, cause the
agent to abandon that direction, and redirect many subsequent experiments
around a false premise.

Automation can also make richer validation practical. An agent can inspect
execution reports continuously, quarantine a suspect experiment, request a
reproduction, or refuse to update its hypothesis until additional checks pass.

For an autonomous research loop, deciding whether a result is trustworthy
enough to learn from becomes part of the system itself. Losses and benchmark
scores are not sufficient if the agent cannot tell whether the intended
training procedure produced them.

## Where should this field start?

Pretraining has extraordinary cost per failure, relatively structured
execution, and a small number of highly specialized users. RL has more
cross-component failure modes, less stable reference behavior, and a broader
set of rapidly changing systems. Autonomous research increases the consequence
of errors in both domains.

[TrainCheck](traincheck-in-practice.md) is one exploration of this space. Its
current approach—learn contextual invariants from reference executions and
report the first violation—has worked on silent errors in distributed PyTorch
training. We do not assume the same design transfers unchanged to every
pretraining or RL stack.

We are trying to answer three questions:

1. Which silent failures cost training teams the most time, compute, or lost
   research progress?
2. Which execution relationships remain meaningful when exact metrics are
   unpredictable?
3. What must a validation tool support before a team would place it in a real
   training workflow?

If you work on pretraining, RL, or autonomous research, we would like to hear
where this framing is wrong and which problem is worth solving first. Open a
discussion on [TrainCheck GitHub](https://github.com/OrderLab/TrainCheck), or
contact the OrderLab team.

---

*This is the final post in a three-part series on trustworthy ML training.
Previous: [Why ML Training Failures Are So Hard to
Localize](traincheck-in-practice.md). Start the series: [ML Training Can Be
Wrong Even When the Loss Goes Down](training-reliability.md).*
