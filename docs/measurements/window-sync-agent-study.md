# Delaying DDP gradient synchronization to token-window boundaries

Three new matched agent pairs start with the same working streaming DDP loop.
The requested improvement is to delay gradient communication until each token
window ends. The task preserves the numeric trajectory and streaming input
contract, including epoch tails. One arm receives a documented optional
two-line TrainCheck diagnostic. The adapter runs public examples, profiles
native NCCL all-reduces and uses TrainCheck fingerprints for final replica
agreement. This compares access to the complete optional diagnostic workflow,
including its new task adapter. Both arms may write diagnostics and receive
the same GPU request, CLI and decision budgets. The task does not hint at a
pre-existing bug or a preferred implementation.

The operator stores each submitted production file and samples intermediate
edits. Native CLI input, cached input, output and elapsed time are kept apart
from GPU usage. First valid submission means the first frozen decision whose
code later passes every held-out check. Its cost ends before any GPU run
requested by that decision. Agent verification and independent replay are
separate events and GPU costs. Report time, token use and GPU cost for each
pair. Count submitted diagnostic source lines for each agent, including short
entry points that call the optional checker. Audit execution using actual
check output, not filename or documentation reads alone.

The prior [token budget study](token-budget-agent-study-20261118.md) supplies
the workload, independent concatenated-batch reference and frozen numerical
scenarios. The augmented scorer checks pre-step gradients, post-step parameters
and momentum, window boundaries, and repeated or early gradient communication.
It uses CPU profiler records emitted by the underlying NCCL process group with
input shape recording active. The workload uses single-element all-reduces for
token counts. Larger all-reduces carry gradient data. Compare the aggregate
gradient element count with the full trainable gradient payload at each step.
Separate buckets or parameters may be reduced individually. Each reduction
must occur following the final backward marker of that window and precede the
optimizer step. The measured quantities are NCCL launches and reduced tensor
elements for this workload's DDP reducer and ordinary tensor all-reduce
implementations.

Native qualification compares the working per-microbatch-sync starter, a
correct once-per-window implementation, and a private variant omitting
synchronization in the final partial window. The optional public check uses
separately specified public cases and prints evidence on each rank. Private
replay uses all six frozen scenarios, and is withheld until the solvers finish.
The low-level collective parser is shared, so controls verify its positive
and negative behavior. This is a small workflow pilot, with no model-training
performance claim.

## Results

All six agents first submitted a valid implementation in turn one and
delivered the same production file in turn three. Both ranks passed all six
held-out scenarios for every solver implementation. No solver delivered
a silent numerical failure or an early-gradient-communication violation.
The sampled edit histories contain only the working starter and one solver
implementation per trial, and no agent made a production edit after seeing a
GPU diagnostic result. Source samples can miss brief intermediate edits.

| Pair | Arm | Diagnostic lines | TrainCheck cases passed (both ranks) | First correct elapsed s | First correct total tokens | First correct uncached input + output | Delivery elapsed s | Delivery GPU-s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Control | 174 | 0 | 72.26 | 116,693 | 14,549 | 157.50 | 49.256 |
| 1 | Optional TrainCheck | 7 | 4 | 41.35 | 122,357 | 16,501 | 97.39 | 49.436 |
| 2 | Control | 169 | 0 | 69.31 | 100,290 | 26,818 | 126.37 | 49.622 |
| 2 | Optional TrainCheck | 7 | 4 | 34.40 | 106,146 | 13,986 | 83.56 | 49.623 |
| 3 | Control | 198 | 0 | 35.59 | 74,408 | 9,640 | 136.80 | 49.465 |
| 3 | Optional TrainCheck | 164 | 0 | 63.21 | 100,482 | 24,322 | 122.89 | 48.829 |

All six correct submissions predate the first GPU request, so all first
correct GPU costs are zero. Total tokens count cached input once. Each agent
used one diagnostic request and one ordinary example run. Timings and token
counts reflect three pairs with independently chosen diagnostics and queue
effects.

Two of three optional-arm agents opened the guide and ran the provided check.
Their seven-line entry points replaced 174 and 169 lines of handwritten
diagnostic code in their paired controls, a reduction of about 96% in
submitted diagnostic lines. Each executed all four public cases successfully
on both ranks, including the partial tail. The third optional agent did not
open the guide in the recorded operations and wrote 164 lines for its own
diagnostic. That pair’s control wrote 198. The custom diagnostics chose their
own inputs and coverage. The line counts describe authored scaffolding,
with equal frozen private scoring across all six implementations.

The two adopters reached first correct submission faster in their pairs.
The third available-tool run was slower to first correct submission. All
three optional runs had higher first-correct total token counts than their
controls. Uncached input plus output moved in both directions. GPU usage
through delivery was close because all six made the same number of requests.

## Communication results

All six agents used no_sync during token-mode accumulation, then issued
one all-reduce per parameter at each window boundary, including the partial
tail. In the held-out overshoot-and-tail case, at each rank the six
microbatches became updates ending at microbatches 2, 4, and 6. Under the
working starter, the rank reduced 4,350 gradient elements in six NCCL calls.
Every solver reduced 2,175 elements in nine NCCL calls. The operator’s
packed positive control reduced the same 2,175 elements in three calls.
Scalar reductions for token counts are separate from these numbers.

The new task allowed separate all-reduces per parameter or bucket at a
window boundary. This yielded correct values and halved the gradient
payload in that example, while dispatching 50% more gradient collectives.
The launch count and reduced tensor-element count are actual native
profiler observations. Throughput, link traffic and collective latency
were outside the experiment.

The private tail-drop control produced numerical errors in all three
scenarios with a partial tail. With one epoch-wide partial window it
emitted zero gradient reductions. The original per-microbatch starter
matched the numerical reference and failed the communication contract
where a window spanned multiple microbatches. These controls test distinct
ways the optimization can appear correct under a narrow check.

## Reproduction and evidence

Ring18 used devices 4 and 5 (NVIDIA A16, driver 535.309.01), Python 3.10.14
and PyTorch 2.2.2+cu121. Both arms ran the same native Codex 0.160.0 model
settings, requested gpt-6-astra with high reasoning and priority service.
Two agents worked concurrently per pair. Their GPU requests were serialized.
The scope relied on instructions. All six read a generic host policy skill and
four also read its generic schema. No reads of peer solutions or private
grading code were observed in the recorded commands.

Evidence is stored in outputs/ring18-window-sync-study/ in the canonical
checkout and in /home/yuxuan/experiments/traincheck-window-sync-study on
Ring18. The 18 decision submissions and sampled source states required
eight replays, counting the unchanged starter separately in each arm.
Across those replays the evaluator ran 48 two-rank scenarios and recorded
96 rank outcomes. The GPU watcher verified release after each allocation.

| GPU use | Allocated GPU-seconds |
| --- | ---: |
| Native qualification and profiler probe | 125.137 |
| Six solver trials | 296.231 |
| Independent grading | 200.303 |
| Total | 621.671 |

The [per-trial results](data/window-sync-agent-study.json) preserve exact
costs, tool adoption evidence, communication profiles and receipt
hashes. The private replay and cost join run with the repository’s
tools.window_sync_study.replay and tools.token_budget_study.operator.score_submissions.
Twenty-three focused local tests passed, along with the scoped mypy and
configured pre-commit checks.
