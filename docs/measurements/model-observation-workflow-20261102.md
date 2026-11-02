# Model snapshots and a fresh optional-tool comparison

Status: implementation, native snapshot compatibility and the fresh comparison
are complete. The optional-tool agent voluntarily used the new snapshot and
comparison APIs. Both agents first submitted a valid repair in turn two, and
both final sources passed all six frozen scenarios. This single interrupted
pair establishes working adoption and measured costs, not a causal speedup.

## Problem and implementation

In the [preceding FSDP pilot](fsdp-source-consistency-pilot-20261102.md), the
optional-tool agent read TrainCheck's callable-wrapper manual and then wrote a
79-line parameter/buffer fingerprint diagnostic. Its final diagnostic was 128
lines. No attempt to activate TrainCheck was observed in the reviewed operation
record. The operation record supports
an interface/task mismatch, not an observed installation failure or a measured
advantage of handwritten code over an equivalent TrainCheck implementation.

The signed candidate `498bf5f9f21c44c9d9487fb9b11b14585fd28304` adds
`traincheck.observation`: explicit model snapshots, validated file reading and
writing, selected-pair comparison and complete-rank comparison. It shares model
enumeration with the existing sampler and reuses the tensor tracker's exact
SHA-256 implementation. It preserves all materialized registered parameter and
buffer names, nonpersistent buffers and repeated object aliases. It does not
replace parameters, implicitly gather shards, infer observation points or run
the invariant engine. Missing ranks, duplicate observations and malformed input
are errors. A difference is an observation, not automatically a training bug.

The [generic guide](../model-observation.md) and runnable CPU/DDP examples expose
the workflow without this case's diagnosis, repair or source provenance. The
agent selects its own boundaries and predicates. Complete fingerprints require
synchronous device-to-CPU transfer; this change does not promise negligible
overhead or implement asynchronous tracing.

Importing the lightweight observation API now leaves PyTorch hooks unchanged.
The four existing top-level package exports load lazily and retain their object
identities; explicit instrumentor imports retain their prior activation
behavior. The legacy parallel tensor hash remains unchanged. Its quantization
and first-row scope are not used to establish complete-content fingerprint
agreement.

A focused regression also exposed a legacy proxy observer ignoring the existing
tracing-suppression scope. Snapshot collection emitted four extra variable
records and changed parameter metadata/timestamps with the parent observer.
The narrow suppression guard produces zero extra records and unchanged
metadata in the same test. Ordinary calls still produce their two pre/post
records, and explicit nested application tracing restores those observations.
This is a correctness result, not a tracing-overhead speedup measurement.

## Local validation

The candidate's full local test suite reports 1,074 passed, nine skipped and two
failures in 112.28 seconds. Both failures reproduce with the same assertions in
an isolated archive of parent `55305542d195086f04526bf7ace8febe250bc0e1`:
`test_traincheck_internal_attr_hidden` and `test_inject_eval_loop`. No new
failure was observed. The focused overlapping selection reports 102 passed
and five CUDA-only skips. These counts must not be added together.

Configured Ruff 0.3.3, mypy 1.9.0, Black 24.3.0 and isort 5.10.1 hooks pass.
New reusable modules and examples also pass scoped strict mypy. Tests cover
whole tensor contents, scalar/empty/noncontiguous/bfloat16/Boolean/complex
representations, signed zero and NaN payloads, cached-hash avoidance, alias/name
coverage, malformed inputs, exceptions, RNG/model/optimizer identity and object
lifetime, pure imports and coexistence with active tracking.

The actual two-process CPU example completes two updates per rank and writes
six snapshots. The final complete-rank comparison covers five tensors per
rank and reports equal fingerprints. On this Mac it required an explicit
loopback rendezvous rather than the standalone hostname route. This does not
establish CUDA/FSDP compatibility; that is checked separately below.

## Native Ring18 compatibility

Two-update matched plain/snapshot runs passed on both the two-GPU current
configuration and the one-GPU reference. All 60 complete native state-file
comparisons passed, including model, all buffers, gradients, optimizer, inputs
and Torch/CUDA/Python/NumPy RNG. All seven explicit snapshots cover 76 parameters
and 19 buffers each. Their 665 tensor fingerprints, shapes, dtypes and complete
ordered name sets match independently saved native tensor values.

The rank comparator reports differing loaded and pre-wrap states and matching
post-wrap states, consistent with the already qualified case. This also
illustrates why matching reconstructed replicas alone does not establish
coherent initialization. These observations are operator qualification evidence,
not information supplied to either solver.

| New operator preflight | Full state-file comparisons | Snapshots | Allocated GPU-seconds |
| --- | ---: | ---: | ---: |
| Two-GPU current, plain/snapshot | 42 | 6 | 70.677 |
| One-GPU reference, plain/snapshot | 18 | 1 | 25.176 |
| Total | 60 | 7 | 95.853 |

All four phase starts have matched completion and cleanup receipts, with no
unknown cost. Full tensor files remain on Ring18; 52 compact artifacts were
retrieved and hash-verified, and snapshot reading/rank comparisons were repeated
locally. The paired jobs include private observer/state I/O and have no timing
repeats, so their durations are not an overhead benchmark. They do not establish
live storage-alias preservation or compatibility with arbitrary sharded models.

## Fresh comparison protocol

The new pair reuses the preceding qualified 14,067,712-parameter native FSDP
case. All 3,784 public source files, the corpus and ordinary observations are
unchanged. Both agents get the same neutral readiness task, editable source and
request service. Treatment alone adds 38 explicitly allowlisted package,
documentation and example files. Use remains optional. Neither packet says
there is a bug or includes a case-specific observation recipe.

Both native CLI conversations start fresh with the same pinned model/settings.
Budgets remain six requests, 1–16 updates per request, 240 seconds of worker
lifetime, 900 aggregate CLI seconds and eight decision turns. GPU requests are
serialized, with queue waiting recorded separately. New tool compatibility
must pass before packet review and launch freeze.

The requested agent configuration is native CLI 0.160.0 with `gpt-6-astra`,
high reasoning effort and priority service. The exact binary hash and settings
are bound in `launch-plan.json`; the provider's underlying model weights are
not independently attested. The conversations begin without the parent chat's
task history. Packet equality and instructions are verified, but this is not a
claim of enforced filesystem isolation.

Ring18 runs Python 3.10.14, PyTorch 2.2.2+cu121, CUDA 12.1 and the supplied
Transformers 4.47.0.dev0 source. The current profile uses two NVIDIA A16 GPUs
(physical devices 4/5, driver 535.309.01); the reference uses one. The model is
14,067,712-parameter Pythia with six layers, hidden size 128, four attention
heads and sequence length 64. Two cache domains are represented on one physical
host; this is not a multi-host hardware trial. The fixed corpus, seed 1234,
source manifests and original oracle states are retained with hashes.

The six-scenario private criterion and original native oracle states are reused
read-only with hashes. The earlier failed strict cross-world numerical criterion
remains failed; acceptance continues to require coherent initial source and
complete same-topology trajectories. Every submitted source occurrence is
retained, and only exact complete source maps may share replay results.

Report mechanism hypothesis, runtime confirmation, first independently valid
submitted repair, visible verification and terminal completion separately.
Full cache-version diagnosis remains distinct from finding inconsistent rank
state. Tokens include cached input once; within-turn milestones retain token
intervals rather than interpolation. GPU cost is allocated device count times
supervised worker lifetime after verified release. New qualification, solver
and private grading costs are separate; historical qualification is not charged
again.

Tool-use accounting includes the new snapshot interface. Absence of API trace
files is not evidence of non-use. Documentation reads, preparation, actual
capture, complete rank coverage and successful comparison are separate facts.
Capture attribution must bind the submitted source, successful request and
validated runtime artifacts; a JSON filename alone is insufficient.

This is a fresh workflow pilot on a previously examined case. One matched pair
cannot establish a general or causal efficiency improvement, especially when
agents select different validation work.

### Transport interruption

The first source upload in each arm exceeded the controller's 60-second SSH
timeout. Both request receipts explicitly record that remote execution had not
started and allocated GPU cost was zero. The completed first CLI turns remain
part of the experiment: control used 78.330 CLI seconds and treatment used
69.314 seconds. Their cumulative input-plus-output token counts were 371,462
and 324,841, respectively, including cached input once. These are partial
costs, not time-to-diagnosis or repair results.

The first recovery transferred the delta successfully but rejected the reusable
preflight directory because that earlier run had generated Python bytecode
there. The strict full-source check failed before either request launched GPU
workers or resumed an agent. The directory and failed receipts were preserved.
A separate clean base now contains only the 3,824 manifest-selected source
files; it is never executed. Both actual pending submissions passed a remote
dry reconstruction with matching complete byte and permission maps.

Recovery retains the same native conversation IDs, submissions, logical request
numbers and remaining budgets. Its separate amendment replaces the full-source
upload with hash-verified reuse of this clean base and transfer of changed
contents, equally for both arms. The first delta packages contain seven blobs
each and approximately 440 KiB instead of approximately 20 MB. The frozen task,
optional package and acceptance criteria remain unchanged. Permissions are
preserved relative to the actual frozen submissions; the original protocol's
file-copy snapshots had already made those files mode 0644. This recovery does
not claim to preserve executable modes from the original agent workspace.

All three physical attempts at each logical first request retain separate
receipts and queue intervals. The failed assembly duration is not classified as
operator suspension or charged as a second logical request. Report actual
elapsed time, queue-adjusted time and time additionally adjusted for measured
operator interruptions separately. The failed upload itself remains visible;
operator suspension is not a measurement of exact network downtime. This
interruption further limits any end-to-end efficiency comparison.

The clean-base recovery subsequently completed both arms' first and second GPU
requests. Treatment's third upload then exceeded the same 60-second deadline
despite its 452,435-byte delta package. Its receipt explicitly records no remote
execution and zero GPU cost. Control continued independently and finished. An
append-only, same-thread recovery completed treatment's pending third request
with longer transport deadlines and unchanged worker, turn and CLI budgets.
Its accounting preserves the complete earlier composition, failed attempts
and original submissions. One prelaunch check also stopped before any service
started because the earlier timed-out upload subsequently appeared as a partial
manifest. That manifest was verified as the exact 603,648-byte prefix of the
2,129,045-byte original, with twelve other complete matching files. The partial
upload remains unused. A separate bound prelaunch amendment verified the
preserved partial map and absence of owned processes, then used a fresh complete
upload and the unchanged strict assembly checks. This abort consumed no agent
turn or GPU execution. These network interruptions preclude treating this as a
clean wall-time benchmark.

## Observed adoption

The treatment voluntarily imported `snapshot_model`, `write_snapshot` and
`compare_model_snapshots` in its first diagnostic. These calls executed on the
GPU workers. The two distributed requests returned four valid snapshots each;
the reference returned one. Every snapshot covers 95 registered tensors, for
855 tensor observations across nine snapshots. Turn three also successfully
executed `read_snapshot`, pair comparison and
`compare_ranks(expected_ranks=[0, 1])` locally with assertions. Turn four compared
both repaired ranks with the reference initialization.

The observations changed the agent's explanation: it rejected its initial
RAM-efficient-loading hypothesis after observing that the flag was disabled.
It identified 76 differing pre-wrap tensor fingerprints and 38 changes to rank
zero's loaded state during wrapping, then selected synchronization. It also
executed the preservation predicate against the original and revised snapshots,
observing failure and success respectively. The actual cache-version origin
remained unresolved in both agents' final responses.

No TrainCheck import, configuration or runtime integration error was observed.
The agent still selected observation boundaries, wrote an FSDP subclass adapter,
entered the full-parameter context and added a rank-zero snapshot broadcast.
Its first cross-rank inspection used manual JSON pairing before it adopted the
strict comparator. A local missing-`safetensors` error occurred before snapshot
reading; a subsequent checkpoint-reader retry and TrainCheck comparisons
succeeded. This is distinct from a TrainCheck integration failure.

## Independent correctness and first valid repair

All nine ordered submission occurrences were retained. Only exact
arm/content/permission identities shared replay results, yielding six source
maps and 36 actual scenario replays. Both initial sources passed the four
healthy/uniform controls and failed mixed and swapped generation. Every repaired
source passed all six scenarios. Thus 32 replays passed and four expected
baseline mismatches failed, with zero unknown outcomes. Control's final turn
five and treatment's final turn four both passed 6/6.

The first independently valid submitted source is turn two in both arms.
This is a retrospective correctness judgment at a frozen submission boundary,
not the earlier file-edit event or the later moment when the agent saw its
own validation results. All preceding sources were checked. Costs below include
the completed first diagnostic request and the completed first two CLI turns;
they exclude the second GPU request, which followed that submission.

| At the first valid submitted repair | Control | Optional TrainCheck |
| --- | ---: | ---: |
| Global decision turn | 2 | 2 |
| CLI execution time, seconds | 188.260 | 141.211 |
| Input plus output tokens | 981,070 | 715,584 |
| Cached input tokens, already included above | 901,376 | 641,024 |
| Uncached input plus output tokens | 79,694 | 74,560 |
| Allocated solver GPU-seconds | 35.031 | 25.289 |
| Actual elapsed time including interruptions, seconds | 2,113.871 | 2,075.498 |
| Elapsed time excluding queue and measured operator suspension, seconds | 434.263 | 365.628 |

With control as denominator, this pair used 24.99% less CLI execution time,
27.06% fewer total tokens and 27.81% fewer allocated GPU-seconds to its first
valid submission in treatment. CLI time excludes remote service time and
queue/operator pauses. The adjusted elapsed clock still includes failed
transfers; neither clock establishes a clean, general wall-time speedup.
There are no timing repeats, and the agents wrote different diagnostics and
regression checks.

Visible runtime confirmation of inconsistent initial state occurred in turn
two: 376.418 versus 338.619 seconds after removing queue and recorded operator
suspension. Visible repair verification followed in turn three: 568.975 versus
515.522 seconds on that same adjusted clock. These are received-event upper
bounds, distinct from submitted-source boundaries. Tokens for such within-turn
events are retained as completed-turn brackets, without interpolation. Neither
agent established the actual cache-version origin of the inconsistency.

## Completed-trial costs

| Completed-trial metric | Control | Optional TrainCheck |
| --- | ---: | ---: |
| Accepted logical execution requests | 4 | 3 |
| CLI execution time, seconds | 362.937 | 294.554 |
| Input plus output tokens | 1,852,226 | 1,269,560 |
| Cached input tokens, already included above | 1,750,784 | 1,181,184 |
| Uncached input plus output tokens | 101,442 | 88,376 |
| Allocated solver GPU-seconds | 118.342 | 72.822 |
| Actual wall time including interruptions, seconds | 2,729.242 | 4,171.665 |
| Wall time excluding recorded queue waits, seconds | 2,534.728 | 4,008.938 |
| Wall time additionally excluding measured operator suspension, seconds | 949.162 | 847.126 |

These are original-start totals across the preserved native conversations and
recoveries. Tokens include cached input once; reasoning tokens are already part
of output. The last clock still includes failed transfers and service overhead;
it is not a clean counterfactual wall time. Core qualification and independent
grading GPU costs are separate from the solver totals.
Token volumes are not dollar prices: most of the total token difference is
cached input, and input, cached input and output can have different prices.
The uncached-input-plus-output volume is 12.88% lower; this also is not a
billing estimate.

The terminal totals are 31.46% lower in tokens, 18.84% lower in CLI time and
38.46% lower in solver GPU-seconds for treatment, with control as denominator.
However, the agents chose different verification work. Both ran one current
update, sixteen current updates and sixteen reference updates. Control alone
then ran sixteen updates through the ordinary entry point to check whether its
diagnostic changed outputs. That extra request cost 35.627 GPU-seconds,
accounting for 78.27% of the 45.520 GPU-second terminal difference. Treatment did
not perform that extra noninterference check. These totals therefore do not
establish an intrinsic tool speedup. Restricting GPU accounting to the first
three matching request profiles gives 82.716 versus 72.822 GPU-seconds; this
still compares different agent-authored diagnostics without timing repeats.

| New experiment cost category | Allocated GPU-seconds |
| --- | ---: |
| Operator snapshot compatibility preflight | 95.853 |
| Both solver conversations | 191.164 |
| Independent 36-scenario grading | 995.453 |
| Total new GPU use | 1,282.470 |

All grading starts have known completion/cleanup costs, with no missing or
duplicate phase charges. Both GPUs were released after grading. Historical
qualification/oracle cost is not charged again. The grading jobs include full
private state observations and are not solver timing measurements.

## Evidence

Local evidence: `outputs/ring18-source-observation-20261102/` in the canonical
TrainCheck checkout. Remote execution:
`/home/yuxuan/experiments/traincheck-runtime-study-d-20261102` on Ring18.
Earlier oracles remain under `traincheck-runtime-study-c-20261102`.

`CHECKPOINT.json`, `core-validation.json`, `optional-manifest.json`,
`qualification-reuse.json`, `protocol-reuse.json` and `README-operator.md`
record the implementation, exact inputs, validation and reproduction steps.
Large model states and raw transcripts are not committed.

The final solver index is `recovery-3/solver-completion.json`; it references
control's preserved recovery-two composition and treatment's flattened
three-segment history. `independent-adoption-audit.json` binds the nine returned
snapshots, executed comparison operations and diagnostic sources.
`milestone-annotations-final.json`, `milestone-summary-final.json` and
`private-grading-join.json` bind the independent source grades to the original
turn timestamps, native usage counters and physical service attempts. Earlier
visible-only annotations remain unchanged.

The complete grading metadata is under
`grading-results-v1/independent-grade-v1/`, including `summary.json` with SHA-256
`6cc322509b79525b57f0078b8a5238bcbfe666021456491aed8274039b6f1989`.
All 203 exported member hashes were checked locally. The identical metadata
files were losslessly recompressed from a 27.6 MB gzip export to a 682 KB xz
export for transfer; no fields, sources or results were dropped. Native tensor
states stay on Ring18. The final milestone summary has SHA-256
`d3beb3ae0b7575e85019a43992bd0c43644c15b46a29bedac82acfbb11550c8b`.
The separately frozen grader and its reproduction/review instructions are under
`evaluation/grading/`; post-freeze recovery and evidence-path adapters retain
their prior versions, hashes, focused regressions and explicit amendments.

An independent export review checked all 203 files, complete source/permission
bindings, 36 phase receipts, outcomes and costs. Its record is
`evaluation/grading/export-review-v1.json`, SHA-256
`8b59ecdc66ccd545a3f67922143687204986d3bdf5a299a4bb3690e87726547b`.
It validates the exported evidence without rerunning training or recomputing
native tensor oracles. The compact final accounting and adoption report is
`final-adjudication-report.json`, SHA-256
`5239d04780f0d9475ab51d970d68f5503b827c74278da81827e83eac928d43f5`.
