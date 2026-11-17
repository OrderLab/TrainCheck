# Readiness of multi-worker DDP checkpoint continuation

The study asks whether optional TrainCheck access helps an agent find a
silent training difference before recommending release. Three new matched
pairs receive a two-rank PyTorch DDP MNIST workload with dropout, AdamW,
cosine scheduling, distributed shuffling and persistent DataLoader workers.
They can inspect the code, use the ordinary smoke run, and choose their own
diagnostics. Neither arm receives a case-specific checker or a diagnosis.
The optional arm receives only the existing generic TrainCheck observation
package and guide from the model-snapshot study. Both arms are free to use
ordinary PyTorch.

The ordinary smoke compares a zero-worker restart and runs an uninterrupted
multi-worker job. The release task asks for continuation under the same
settings. The fixture is small enough for repeated bounded two-GPU runs,
uses real cached MNIST digits, and represents a realistic data-read-ahead
failure mode rather than a production incident. Ring18 uses isolated request
directories with the same native CLI, time, turn and GPU budgets for both
arms. The model has 76,330 trainable parameters.

The independent evaluator compares a fresh trainer restored from a saved
version-one checkpoint with a frozen uninterrupted reference. It verifies
data order and loss at each step and final model tensors, AdamW state,
scheduler, step number and per-rank PyTorch CPU and CUDA random states.
Scenarios include different worker prefetch depths, an exact epoch boundary,
a restart near an epoch boundary and a smaller final batch. The graders
record runtime failures and missing evidence separately. This private
evaluator and its inputs are excluded from all six solver packets.

A readiness claim counts as a missed silent error when a normally completing
held-out continuation has a proven data or training-state divergence. To
credit pre-delivery discovery, require a visible explanation of the concrete
failure or binding to the agent's own diagnostic output. Record whether that
evidence appeared before or after the first correct source. Record whether
an agent actually used TrainCheck in the execution that supplied diagnostic
evidence. A doc read, plausible assertion, finite loss or eventual hidden
pass alone does not demonstrate detection. Capture final disposition, time,
tokens and allocated GPU-seconds, and preserve the complete trial results
even if all agents independently find the same issue.
