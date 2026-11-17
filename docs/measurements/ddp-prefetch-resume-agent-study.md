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

## Results

All three control agents and all three agents offered TrainCheck identified
the same sampler-cursor failure before their first GPU request. All delivered
repairs and passed the six frozen continuation scenarios on both ranks.
Neither arm delivered a silent failure. Two agents, control 1 and optional 2,
also ran their own diagnostic against the unmodified source. Each observed
offset 112 at a checkpoint written after processing 48 samples per rank.
The ordinary zero-worker continuation passed. A subsequent multi-worker
restart silently skipped 64 examples per rank and changed losses, model
tensors and AdamW state.

| Pair | Arm | Diagnosis s | First correct turn | First correct elapsed s | First correct total tokens | Prior GPU-s | Delivery GPU-s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Control | 28.28 | 2 | 95.34 | 140,977 | 24.913 | 84.571 |
| 1 | Optional TrainCheck | 26.28 | 1 | 37.79 | 90,633 | 0.000 | 59.849 |
| 2 | Control | 29.50 | 1 | 61.32 | 128,376 | 0.000 | 24.609 |
| 2 | Optional TrainCheck | 33.93 | 2 | 130.96 | 164,623 | 25.208 | 99.533 |
| 3 | Control | 32.24 | 1 | 42.54 | 78,449 | 0.000 | 74.233 |
| 3 | Optional TrainCheck | 20.77 | 1 | 57.45 | 110,790 | 0.000 | 59.758 |

Diagnosis time is the receipt of the first visible, concrete mechanism
statement. First correct submission is a retrospective pass on the frozen
private replay. A GPU request made by that submission follows the boundary.
Tokens include cached input once. The two agents who reproduced the unfixed
pipeline first spent one GPU request before submitting their correct revision.

Three of three optional agents opened the same generic guide. No submitted
or executed program imported or used TrainCheck. Both arms wrote their own
continuation tests with plain PyTorch. The task already exposed the sampler
position in checkpoints and batch IDs through the training module, which made
it possible to observe exact input divergence directly. Evidence from this
fixture establishes a real gap in the ordinary smoke and shows that all six
agents independently closed it. There is no instance here of TrainCheck
catching a defect a control agent missed.

## Qualification and replay

The starter passed the ordinary two-GPU smoke. It completed all six private
settings, but produced different samples and trained state in four of them.
It passed the zero-worker and exact-epoch-boundary settings. The corrected
control passed all six. The local two-process Gloo run reproduced the same
success/failure pattern, independently of GPU kernels.

A separate Ring18 experiment also stopped and restarted the actual torchrun
processes. At the same seed all six pre-checkpoint rank-step losses matched
the uninterrupted reference. The fresh two-rank restore exited successfully
and all ten post-checkpoint rank-step losses differed.

The graders use same-settings resumes from checkpoints written by the
submitted source, followed by a fresh Trainer and worker pool inside the
supervised replay process. Compatibility with older, already mispositioned
checkpoints is outside these trials. This small fixture uses cached real
MNIST images and the actual PyTorch DataLoader and DDP implementations.
It represents a common engineering risk under realistic APIs.

Ring18 ran on two A16 GPUs with PyTorch 2.2.2+cu121 and Python 3.10.14.
The first transfer for qualification disconnected before Ring18 created a
request. A later SSH connection restored access before any solver started.
The six solvers had distinct native conversations with the same model and
run budgets. Both arms had the same ordinary inputs. Only the
unchanged generic TrainCheck package and guide were added in the treatment.

| GPU use | Allocated GPU-seconds |
| --- | ---: |
| Native qualification | 76.452 |
| Six solver trials | 402.553 |
| Private replays | 200.083 |
| Separate process-restart control | 74.275 |
| Total | 753.363 |

See [trial data and receipt hashes](data/ddp-prefetch-resume-agent-study.json).
Operator evidence is under outputs/ring18-prefetch-resume-study in the
canonical TrainCheck checkout. The native run directory is
/home/yuxuan/experiments/traincheck-prefetch-resume-study on Ring18.
The stage manifest includes the original MNIST IDX hashes and the hashes
of the 512-example subset used by the service. Twenty-three existing
harness tests and scoped strict type checks passed.
