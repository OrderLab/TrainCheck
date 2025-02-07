# Transferability Showcase

## Goal:
Show that the invariants inferred by TrainCheck can be applied to other pipelines that were different in task semantics.

## Methodology
1. For each input, infer invariants on torch 2.2.2
2. A filter step: keep only the valid invariants for the next step.
2. Apply invariants to different pipelines.
3. Report the presentage of transferable invariants.