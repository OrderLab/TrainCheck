---
date: 2026-08-03
draft: true
slug: scaling-training-faster-than-trust
categories:
  - ML Reliability
  - Reinforcement Learning
  - Distributed Training
description: As ML workflows involve more components and automated decisions, results need execution evidence that can be checked, combined, and carried forward.
---

# We're Scaling Training Faster Than We're Scaling Trust

ML is advancing along many fronts at once. Training systems are becoming more
optimized and distributed, workflows span more components, and software is
beginning to take on more of the experimental loop. That is exciting: it
expands the models, training procedures, and research processes we can build.

The reliability question is shared across these developments. Judging an idea
now depends on more code paths, component versions, data transformations, and
automated decisions. A final metric can look plausible when one of them is
wrong.

The central change is not simply that individual runs are larger. A result is
increasingly an input to another component or decision. We think the unit of
trust should therefore be a result together with its execution evidence, not
the metric alone.

## Results should carry their execution evidence

An experiment often leaves behind a checkpoint, metrics, a configuration, and a
code revision. These may not identify its data and tokenizer, which policy
generated its rollouts, which reward model scored them, or whether expected
training relationships held.

A result should travel with a compact record: relevant component versions, data
lineage, checks and their outcomes, and pointers to detailed traces. It need not
retain every event. It should say what was checked, against which expectation,
and where the supporting evidence lives.

This makes later correction possible. If a kernel, dataset, or model version is
faulty, teams should be able to find dependent results and mark them for
re-evaluation. Provenance supports both reproduction and precise invalidation.

## Checks should compose as workflows grow

Within a training job, checks can cover relationships such as replica agreement,
parameter and optimizer ownership, state transitions, and operation order.
Across components, the relationships change: a rollout should name the policy
that generated it, a reward should refer to the intended sample, and a learner
should know which versions of its inputs it consumed.

These checks should compose. A component should expose the identity and lineage
of its outputs, and consumers should check their assumptions about them. The
combined evidence should connect events across process and service boundaries
without requiring one tool to understand every implementation detail.

The verl fused-kernel issue gives one concrete example. A kernel returned
incorrect log probabilities when the vocabulary size was not divisible by its
block size; finding the cause required tracing reward collapse and
log-probability disagreement into the optimized kernel. ([original
report](https://github.com/verl-project/verl/issues/2656),
[investigation](https://github.com/verl-project/verl/issues/2899), [merged
fix](https://github.com/verl-project/verl/pull/5349))

This was a difficult diagnosis inside one component, not a cross-service
failure. If those log probabilities feed later rollouts or updates, however, a
larger workflow should record which results consumed them and mark those results
for review.

[TrainCheck](traincheck-in-practice.md) explores one part of this design:
contextual checks within a training job. The broader challenge is to connect
such checks with versioning, lineage, and contracts across tools.

## Combine routine checks with richer boundary checks

Not every check should run at every step. Lower-cost checks—version identifiers,
replica hashes, collective order, required events, and selected state
relationships—can run continuously or periodically and catch problems early.

Richer checks can run after a code change, before checkpoint promotion, when
rollouts reach a learner, or before an automated system accepts an experiment as
evidence. They can use fuller traces, differential runs, replay, or
cross-component lineage. The evidence record should include their outcomes.

This layered design is intended to keep routine monitoring affordable and
reserve expensive validation for checkpoint promotion, cross-service handoffs,
and automated decisions.

## Automated research needs evidence-admission rules

If a research agent plans follow-up experiments, it needs an explicit rule for
when a result is eligible to influence the next decision. Passing a loss or
benchmark threshold is not enough if required execution or provenance checks
failed.

The corresponding invalidation rule matters just as much. If an input or run is
later discredited, downstream conclusions and experiments should be marked for
review rather than remaining silently embedded in the research history. How to
represent and apply these rules across that history is an open challenge.

## Our position: make execution evidence part of the result

We can build this incrementally. Cheap checks can run continuously; richer checks
should run where a checkpoint, rollout, or experiment will influence later work.
As workflows become more automated, they should use a result only when the
required checks have passed.

We do not need a proof of the entire workflow to start. We can check specific
relationships that matter today, preserve the evidence those checks produce,
and support precise invalidation. If validation is designed alongside new ML
workflows, greater automation can make checking more systematic rather than
less.

If you build training systems, RL infrastructure, or automated research tools,
we would like to hear what evidence your results should carry and where stronger
checks would help most. Join the [GitHub
Discussion](https://github.com/OrderLab/TrainCheck/discussions).

---

*This is the final post in a three-part series on trustworthy ML training.
Previous: [Why ML Training Failures Are So Hard to
Localize](traincheck-in-practice.md). Start the series: [ML Training Can Be
Wrong Even When the Loss Goes Down](training-reliability.md).*
