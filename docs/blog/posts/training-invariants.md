---
date: 2026-07-20
draft: true
slug: checking-training-invariants
categories:
  - ML Reliability
  - Training Invariants
description: Training is nondeterministic, but correct executions still preserve checkable relationships. We call them training invariants.
---

# We Cannot Predict the Loss Curve. But We Can Still Check the Training

Suppose two correct training runs start from different random seeds. They see
examples in different orders, execute nondeterministic kernels, and follow
different numerical trajectories. Their losses will not match step for step.

Now suppose one of those runs silently stops updating half of the model.

If correctness means reproducing an exact sequence of numbers, training seems
almost impossible to check. But exact outputs are stronger than what we need.
We may not know what the loss should be at step 1,000. We often know what must
happen *during* step 1,000.

An optimizer should update the parameters associated with the gradients it
receives. Replicated parameters should stay consistent across ranks. Required
initialization should occur before training begins. A model used for rollout
should correspond to an intended policy version.

We call these expected relationships **training invariants**.

## Check relationships, not trajectories

Consider the DDP failure from the [previous
post](training-reliability.md). We cannot predict the exact parameter values on
each GPU after an update. We do not need to. If those parameters are replicated,
their values should agree across ranks.

An optimizer failure gives us another example. We cannot predict the exact
parameter update, but after an Adam step with a nonzero gradient, the relevant
weights and optimizer state should normally change.

The initialization failure is earlier still. We do not need to know whether a
particular initialization will lead to a good model. We can check that the
intended initialization operations ran and wrote the relevant parameters.

| Failure | Outcome-level question | Execution-level relationship |
| --- | --- | --- |
| DDP wrapper bypass | Should the loss rise at step 150? | Do replicas remain consistent after updates? |
| Silent optimizer failure | Should this model learn faster? | Does a parameter with a gradient get updated? |
| Wrong initialization path | Will this initialization train well? | Did the intended initialization execute? |

The right-hand questions are narrower. That is precisely why they are easier to
answer.

## Correct behavior depends on context

“Every parameter changes after every optimizer step” would be a terrible
invariant. Frozen parameters should not change. A gradient-scaler may skip an
update. Gradient accumulation deliberately postpones a step. Sparse updates may
touch only part of a tensor. Fully sharded parameters should not be equal on
every rank.

Useful invariants therefore need **preconditions**. A relationship may apply
only when a parameter is trainable, receives a gradient, belongs to the active
optimizer, and the step is not skipped. Cross-rank equality may apply to a
replicated LayerNorm weight but not to a sharded projection.

This turns a simple assertion into a contextual statement:

> When these execution conditions hold, this relationship should also hold.

Context is what lets execution checking tolerate legitimate variation without
declaring every unfamiliar behavior a bug.

## What kinds of relationships can we check?

A training step is an ordered stream of events. Modules run, tensors move,
backward creates gradients, optimizers read them, and parameters change. That
creates several useful families of invariants:

- **Consistency:** corresponding values agree across replicas, iterations, or
  related executions.
- **State change:** an API call changes—or deliberately does not change—the
  state it owns.
- **Ordering:** one event occurs before another, such as zeroing gradients
  before the next backward pass.
- **Containment:** an operation occurs inside the expected module, optimizer
  step, or context manager.
- **Arguments and outputs:** APIs receive values, devices, shapes, or modes that
  follow a recurring pattern.

None proves that the learning objective is scientifically sound. Together they
provide evidence that the implementation is doing what its surrounding code
and configuration imply.

## Who writes the specification?

Hand-written assertions are valuable when we already know what might go wrong.
The difficulty is coverage. Modern training stacks contain framework code,
distributed wrappers, fused optimizers, third-party libraries, and project
logic. Few teams can enumerate every relationship worth checking.

An alternative is to learn candidate invariants from short executions believed
to be correct. Observe API calls and selected state across several reference
runs; keep relationships that recur in the contexts where they apply; then
check whether a new execution preserves them.

![From reference runs to a concrete invariant
violation](../../assets/blog/traincheck-launch/03-workflow.png)

This makes a reference run a kind of behavioral specification. It also creates
an obvious risk: a narrow or already-broken reference can teach the wrong
behavior. Diversity, provenance, filtering, and understandable violation
reports matter. Learned invariants should complement deliberate tests and human
judgment, not acquire authority merely because they were inferred.

## A missing layer between tests and dashboards

Conventional tests usually validate anticipated cases before a long run.
Dashboards summarize outcomes while the run proceeds. Anomaly detectors flag
unusual signals. Execution invariants occupy a different position: they ask
whether the running system continues to preserve its expected internal
relationships.

These approaches answer different questions:

- **Tests:** Does this known input produce an acceptable result?
- **Monitoring:** Does the run look healthy from its signals?
- **Execution validation:** Is the training procedure still behaving like the
  procedure we intended to run?

The boundaries overlap, and that is healthy. Training reliability is unlikely
to have one universal detector. The more useful question is which evidence can
reduce ambiguity soon enough to change a decision.

In the next post, we make this idea concrete with TrainCheck. Two incidents show
the two outcomes we care about most: finding a silent error before the metrics
do, and turning a flat loss into a specific diagnosis.

---

*This is the second post in a four-part series on trustworthy ML training.
Previous: [Your Training Run Is Not Scientific
Evidence](training-reliability.md). Next: [TrainCheck: Catching Training Bugs
Before the Loss Curve Does](traincheck-in-practice.md).*
