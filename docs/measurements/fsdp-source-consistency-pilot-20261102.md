# Native FSDP initialization: optional TrainCheck pilot

**Status: completed; both final repairs pass all six independent scenarios.**
Both agents used cross-rank runtime diagnostics, but the optional-tool agent did
not activate TrainCheck. This single pair does not establish a tool efficiency
benefit. The earlier
[tensor-parallel clipping](real-rank-runtime-pilot-20261102.md) and
[DDP token-normalization](ddp-runtime-state-pilot-20261102.md) comparisons both
produced source-first diagnoses and no observed TrainCheck activation. They are
retained negative results. This candidate tests an external runtime condition:
different workers can resolve the same model identifier to different cached
weights before native FSDP partitions the parameters.

## Provenance and reproduction boundary

[Halo release PR #103](https://github.com/whitecircle/halo/pull/103) adds agreement
on model source and commit across ranks. It is a production fix with regression
coverage, contained in a larger release; the PR does not link a specific incident
report. The parent is `8bc800971e739777311862544e5f1a321e12e4ba`, and the merged
release is `5abddcb7ee12b46481f6d0facb00046c358724fb`.

The reproduction executes byte-identical selected parent definitions, including
the native cache-loading helpers and c10d-store coordination, bound to real
installed Hugging Face Hub, Accelerate and PyTorch APIs. Native GPTNeoX loading
and Torch FSDP perform the training. It does not run the complete current Halo
package: that package requires a newer Python/framework/CUDA stack than the
shared Ring18 installation. The original source and extraction hashes remain
available for review; no fake Hub API or replacement collective implements the
failure.

The [official Pythia model card](https://huggingface.co/EleutherAI/pythia-14m#model-details)
documents replacement of the model previously served as `EleutherAI/pythia-14m`,
moving that earlier model to `EleutherAI/pythia-14m-deduped`. The experiment uses
unchanged official artifacts:

| Artifact | Revision | Safetensors SHA-256 |
| --- | --- | --- |
| Earlier model, now `pythia-14m-deduped` | `f33025648652797a390d8c54835273845b437161` | `821667edaab2cb38aac634fd1bbda547ecf451f4ea514f47b5a0080be24154fb` |
| Replacement `pythia-14m` | `94f7c35d5e9f2e9bac8ca839329f505b4d007d5d` | `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1` |

These are separate repository histories, not ancestor/descendant commits. The
historical cache namespace and `refs/main` mapping are reconstructed around those
official blobs. This is not a preserved incident cache, and there is no claim
that the Halo failure mode involved Pythia. The model migration establishes an
authentic source of different published weights, not a record of that particular
deployment.

Two worker-local cache domains emulate logical nodes on one physical Ring18 host,
using GPUs 4/5, both NVIDIA A16. Each worker has its own local rank zero and cache
metadata; global ranks and the actual NCCL process group remain distinct. This
does not measure multi-host networking or a filesystem race. Requests use
independent copies of the official caches, not writable links to shared source
blobs. The shared repositories, environment and driver are unchanged.

The execution environment uses Python 3.10.14, PyTorch `2.2.2+cu121` and compiled
CUDA 12.1. The loaded Transformers source is pinned to
`a17f287ac039f92835b5cd9bd8ee28b584c9f65e`; the full environment and executed
source-file hashes are saved with the launch evidence.

## Workload and qualification

The workload uses the official 14,067,712-parameter GPTNeoX model, explicit native FP32
loading, eager attention, no TF32, deterministic algorithms, AdamW with learning
rate `1e-4` and weight decay zero, sequence length 64 and two global examples per
update. The qualification uses two optimizer updates. Identical pretokenized
README-derived inputs isolate the model-loading question from the official
tokenizers' different postprocessing configurations.

The raw checkpoints have different schemas: the replacement has 76 FP16 tensors;
the earlier artifact has 88 FP16 tensors and six Boolean tensors. The additional
18 entries include legacy attention buffers. Native loading reports no missing
or unexpected keys, but that alone is insufficient: the pinned loader can filter
buffer keys from its unexpected-key report, and `state_dict()` omits nonpersistent
buffers. The qualification therefore retains all raw keys and loading messages,
and separately captures complete named runtime buffers. Raw artifact equality
and runtime equality are different claims.

The first complete source-state check confirms that all 76 learned tensors load
exactly as native FP16-to-FP32 conversion, and both models have 19 bitwise-equal
runtime buffers. Native loading recreates some values: the earlier raw
`masked_bias` is negative infinity while the runtime uses finite `-1e9`, and raw
FP16 rotary frequencies differ from their recreated FP32 values. These are
recorded loader behaviors, not values discarded to manufacture equality.

The bounded matrix includes uniform earlier/replacement sources, mixed and
swapped sources, their one-GPU references, and same-world runs with additional
intermediate observations removed. A missing-cache control must produce the
native offline failure. Every run retains worker lifetime, allocation, cleanup,
source/cache integrity and raw failures. Native creation of `hub/version.txt`
initially caused an overly strict cache-integrity receipt to fail; that attempt
and its cost are retained, with immutable blob/ref checks separated from the
recorded metadata initialization.

The 18 recorded qualification attempts through the full-buffer matrix consumed
435.376 allocated GPU-seconds, including the missing-cache control and failed
integrity receipt. This is operator qualification cost, not agent cost; later
side probes, service preflight and grading must be accounted for separately.

Independent checks of the initial parameter-focused matrix established:

- Each uniform initial model exactly matches its corresponding one-GPU source.
- Native FSDP constructs the mixed model from 38 parameter tensors belonging to
  one source and 38 belonging to the other. The result differs from each complete
  authentic source in 38 parameter tensors. The swapped case follows the actual
  native shard ownership too; this is checked from values and padding metadata,
  not inferred from filenames or a model hash alone.
- All ranks can reconstruct the same hybrid. Post-wrap replica equality alone
  therefore does not establish coherent initialization.
- Same-world observed/unobserved final parameter, gradient, optimizer, buffer and
  recorded RNG values agree exactly for the checked pairs. The follow-up checked
  104 overlapping artifacts against earlier runs, 30 buffer comparisons, 34
  replica checks and 20 terminal observer comparisons, all exactly equal.

The original cross-world model tolerance **failed** for healthy uniform controls.
With model `atol=1e-5` and `rtol=1e-4`, maximum two-update differences were about
`3.758e-4` for the earlier model and `3.328e-4` for the replacement. Initial states
were exact. First-gradient relative L2 differences are approximately 0.5275% and
0.04506%, respectively. A key-bias-only explanation was not supported: most
first-step elementwise model violations occur elsewhere, including weights.
Measured gradient differences and Adam's update rule explain update sensitivity,
but do not identify its kernel-level numerical cause. It is not reported as
ordinary rounding proved harmless.

The failed criterion remains failed; no threshold was silently relaxed. The
operator explicitly selected a separate pre-agent scope: exact initial source
coherence and full same-topology healthy trajectories, without a claim of narrow
single-versus-two-GPU numerical parity. Its gate must bind both the successful
state checks and the retained failed original criterion. Two additional bounded
single-GPU controls used one sample per microbatch and two accumulation steps,
matching each distributed worker's forward batch shape. Both models then matched
the two-GPU parameters bitwise at steps 0, 1 and 2. All buffers and recorded RNG
values matched. The replacement model's gradients and optimizer state matched
exactly; the earlier model retained one subnormal gradient/moment discrepancy,
at most about `1.5e-43`, without a parameter difference. These controls consumed
24.690 allocated GPU-seconds. They localize the discrepancy to batch shape and
accumulation order, without proving which numerical kernel causes it. Neither
the submitted workload nor the original failed criterion changed.

## Matched comparison and acceptance

The neutral task extends a pretrained fine-tuning pipeline to two-GPU
FSDP, supplies ordinary short-run outputs, and requests a readiness decision and
justified revision. It does not assert a bug, prescribe a diagnosis or require
TrainCheck. This is pretrained fine-tuning, not checkpoint restoration with
optimizer and step state.

Both arms receive identical editable runtime source, data, ordinary diagnostics
and execution budgets. Treatment adds the same generic optional TrainCheck
manual and implementation used in the earlier pair. Both current and reference
backends execute the submitted source. Both ordinary eight-update service runs
passed, using 46.958 allocated GPU-seconds in total. Their genuine logs and all
16 curve rows are provided. These are cyclic-sample short-run training losses,
not held-out model-quality measurements.

The private actual-entry observer and six-scenario evaluation passed before
launch: six baseline runs produced the expected four healthy and two incoherent
outcomes, and six positive controls passed after changing only
`sync_module_states=True`. Mixed and swapped positives selected different
authentic sources, so the evaluator does not require the latest version. These
12 qualification jobs consumed 328.727 allocated GPU-seconds, including the
previously qualified actual-entry mixed run once.

The observer captures complete named parameters and buffers, optimizer and
recorded RNG values, local gradients, actual forward inputs and native shard
metadata. Step 0 is explicitly after the first backward and before the first
optimizer step. It supports multiple preliminary loads and unused optimizers,
binding the trained model through actual FSDP or stepped-optimizer identity.
Multiple root FSDP constructions or multiple actually stepped optimizers are
unsupported and must produce an unknown grade. The observer does not establish
live alias/storage identity beyond its recorded native layout.

The optional-tool preflight compared two-update runs with and without selective
TrainCheck observation of scalar `dropout` arguments. Each rank executed 38
native calls and emitted exactly 38 entry and 38 return records. All 42 saved
state comparisons matched, covering 2,050 tensor and 5,742 scalar checks, with
no unmatched, duplicate or variable-tracker records. An independent counter-only
run also matched the saved no-counter reference. The two new jobs used 69.336
allocated GPU-seconds. This establishes compatibility for that interface; it is
not a repeated overhead benchmark or validation of arbitrary APIs, tensor
metadata, model tracking or the inference engine.

Uniform-source controls must retain their respective complete initial models.
Mixed/swapped inputs may be repaired to either authentic coherent source; there
is no hidden requirement to select the newest one. The subsequent full runtime
model, gradients, optimizer, input coverage and recorded RNG must follow that
source's independently qualified same-topology trajectory. Detecting a mismatch
and aborting is recorded as detection/containment, not as a completed working
repair under the public contract.

Keep the previous diagnosis definitions. A specific pre-run explanation of
per-worker source disagreement and FSDP assembly counts as a source-supported
diagnosis even if the agent calls it a hypothesis. Later runtime confirmation
does not make that diagnosis runtime-first. If agents again solve the problem
from source, report it; do not force execution or tool use to make the case look
harder.

The pair retains six requests of 1–16 global updates, 240 seconds per
worker request, 900 seconds aggregate CLI execution and eight decision turns.
Measure source diagnosis, runtime-supported diagnosis, first independently valid
submitted repair, visible verification and final completion separately. Within-
turn tokens remain intervals; GPU cost is supervised allocated-device lifetime,
including startup and cleanup. Qualification and private grading costs are
separate from solver costs. No tool efficiency benefit is established.

An independent review of the exported packets found 3,790 shared files: all
except the optional-offer prompt are byte-identical. The treatment adds 32 pinned
tool/manual files. Both arms can edit the same 3,346 Python source files and have
the same fixed entry point, data and contracts. Original source, native names,
warnings and genuine tokenizer-revision provenance remain visible; they could
enable a source-first diagnosis. The review establishes no deliberately added
cause or issue hint, not that the case requires runtime observation. Both fresh
CLI agents use `gpt-6-astra`, high reasoning and priority service. GPU requests
are serialized, with queue time reported separately.

## Completed comparison and independent repair validation

Both trials finished without CLI or transport failures. Control submitted five
decision turns and used four GPU requests; optional TrainCheck submitted four
turns and used three requests. Every submitted source is retained. Nine ordered
source occurrences reduce to five exact complete source maps. All 30 independent
scenario replays finished: 26 passed, four failed as expected for initial
unrepaired sources, and none were unknown. Both earliest valid submissions are
turn 2, with no earlier unresolved source. Both final submissions pass all six.

| Submitted source occurrences | Independent outcome |
| --- | --- |
| Control turn 1 | Four healthy cases pass; mixed and swapped sources fail |
| Control turns 2–3 | Six of six pass; earliest valid repair is turn 2 |
| Control turns 4–5, including final | Six of six pass |
| Optional TrainCheck turn 1 | Four healthy cases pass; mixed and swapped sources fail |
| Optional TrainCheck turns 2–4, including final | Six of six pass; earliest valid repair is turn 2 |

The replay executes the immutable ordinary training entry, with complete source
integrity and state/input coverage checks. Both repairs change the training
pipeline only by enabling `sync_module_states` and adding comments. Standalone
diagnostic/comparison helpers differ across snapshots and were retained in their
complete source identities, even when not imported by that entry.

| Milestone, seconds | Control raw | Control queue-adjusted | Optional TrainCheck raw | Optional TrainCheck queue-adjusted |
| --- | ---: | ---: | ---: | ---: |
| Pre-run mechanism hypothesis | 30.251 | 30.251 | 33.934 | 33.934 |
| Runtime-confirmed state inconsistency | 131.170 | 131.168 | 184.793 | 134.135 |
| First independently valid submitted repair | 148.365 | 148.362 | 204.705 | 154.047 |
| First agent-visible repair verification | 228.722 | 224.517 | 271.333 | 218.093 |
| Terminal trial boundary | 432.754 | 408.810 | 414.381 | 333.094 |

Queue-adjusted time subtracts only measured shared-service waiting. Visible
statements use received-event timestamps, an upper bound on creation time.
First-valid submission time is retrospective; it excludes later private grading
latency and is distinct from the agent's own verification. Both agents' upstream
cache-origin diagnosis remains unresolved, as detailed below.

| Cost through first valid submission | Control | Optional TrainCheck |
| --- | ---: | ---: |
| Native CLI elapsed seconds | 93.470 | 99.253 |
| Input tokens, including cached input | 714,921 | 683,299 |
| Cached input subset | 661,504 | 628,352 |
| Output tokens | 4,994 | 5,031 |
| Input plus output tokens | 719,915 | 688,330 |
| Allocated solver GPU-seconds | 35.563 | 34.992 |

Native counters do not identify tokens at an intermediate statement. The
runtime-confirmation milestones therefore have cumulative input-plus-output
brackets of 293,535–719,915 for control and 304,371–688,330 for treatment. The
pre-run hypotheses occur within the first turn, giving only 0–293,535 and
0–304,371. No interpolation is used.

| Full trial | Control | Optional TrainCheck |
| --- | ---: | ---: |
| Raw elapsed seconds | 432.754 | 414.381 |
| Measured GPU-service queue seconds | 23.944 | 81.287 |
| Input tokens, including cached input | 1,358,537 | 1,127,302 |
| Cached input subset | 1,298,944 | 1,066,496 |
| Output tokens | 11,210 | 9,983 |
| Reasoning output subset | 1,910 | 2,931 |
| Input plus output tokens | 1,369,747 | 1,137,285 |
| Allocated solver GPU-seconds | 118.764 | 82.527 |

Cached input is already included in input tokens, and reasoning is already
included in output tokens. Neither subset is added again. The seven solver
requests consumed 201.291 GPU-seconds. The five operator qualification groups
above total 905.087 GPU-seconds, counting reused actual-entry/no-counter evidence
only once. Private post-trial grading used another 815.932 GPU-seconds across
30 newly executed phases; reused frozen oracles added zero GPU work. These three
categories total 1,922.310 allocated GPU-seconds for this case. GPU reservations
were released after completion. Oracle self-checks ran on the CPU controller;
their preflight elapsed time was about 52.47 seconds, including imports and hash
validation, and is not reported as pure oracle-comparison time.

The optional-tool agent read the TrainCheck manual but neither prepared nor
activated TrainCheck instrumentation. Both arms used ordinary custom diagnostic
wrappers, and all returned artifacts contained zero native TrainCheck API or
variable-trace files. Their validation choices differed: control compared eight
updates across backends and then ran 16 ordinary distributed updates; treatment
compared 16 diagnostic updates across backends. Control checked the 76 saved
parameter tensors, while treatment also checked nonpersistent named buffers,
covering 95 records. Full-trial cost differences therefore reflect different
chosen validation work, not matched measurements of a TrainCheck repair-speed
benefit.

Both agents proposed an unsafe-initialization mechanism before the first run:
unsynchronized FSDP could assemble inconsistent rank-local tensors. However,
their specific proposed origin was RAM-efficient loading skipping nonzero-rank
weights. Actual runtime diagnostics found the loading flags absent and rejected
that explanation, while confirming cross-rank initial-state differences and
38 changed tensors after sharding. Both submissions enabled state synchronization
in turn 2. Neither final answer identified the different cached model sources.
The result must therefore distinguish a source-supported mechanism hypothesis,
runtime-confirmed state inconsistency, and the upstream cause unresolved by
either agent. Operator qualification did establish the cache-version divergence.
This is not evidence of full root-cause localization by either agent from source
or from runtime.

## Saved evidence and current validation

Local evidence root:
`/Users/matt/gitrepos/traincheck/outputs/ring18-source-consistency-20261102`.
Remote root:
`/home/yuxuan/experiments/traincheck-runtime-study-c-20261102`.

The roots retain the official source/artifact provenance, original and revised
qualification specifications, all attempted native runs, and typed operator
helpers. `evaluation/source-proof-v1.json` records raw-to-loaded tensor and buffer
checks plus exact native shard reconstruction, SHA-256
`380d01eef23cb4e0ecc589bcb4db9c1b42d2beb120c2786cf04eb29ca174dd38`.
`evaluation/native-review-v1.json`, SHA-256
`15eaee07020af3c9a4a7472e4976aa4690cdecbc9d552649d1cc040f06e5e856`,
records complete observer/replica overlap and numerical differences.
`evaluation/original-crossworld-gate-v1.json`, SHA-256
`8d7b0be8c6a16504b474eac002c649b2b0e22b416752f4e8700c173a86c92dc1`,
preserves the original failed cross-world criterion with unchanged thresholds.
`evaluation/accumulation-diagnostic-v2.json`, SHA-256
`7a170941297b2a8a7bc2a394d21cb6dace3d0a1b616a8a5e7565d515a721c126`,
records the additional batch-shape controls.

The frozen launch plan has SHA-256
`c02a5cb735b52a00b918c5dd456415eadf7d97b4ee05f61f509da4dfc20fc4b2`.
It binds the packets, environment, operator code, native qualification, service,
actual-entry positive controls, optional-tool preflight and independent grading
contract. The independent neutrality review is
`prelaunch-neutrality-review.json`, SHA-256
`ffe22a0d859f9e571fbd5e5ff1e53a605db4a87e6e643dbdf8204564788a3b36`.
The six-scenario grading contract has SHA-256
`6abfeddd99d16aa109424f4cdec899708c3dfa78aa847efad785487fa517a43c`.

The completed grading summary is
`grading-results-v1/independent-grade-v1/summary.json`, SHA-256
`9b2d524617f61d549aa59a948cf7767deb5736637e5f9c5815c91bc93f2b13c3`.
Its replay plan has SHA-256
`2890a5e00cabbdd934c4f327e9c695c5e577a40fdcad3fa4c51c162f133eb5bf`.
All 165 exported reports/receipts were downloaded and hash-verified against
`grading-result-export-manifest.json`, SHA-256
`3e67d014a07063c258e9a7205c1e0513da2661095d137195f7ac7997c383f060`.
Full tensor snapshots remain in the isolated Ring18 evidence directory.
`grading-reproduction-and-results.json` records exact source-to-occurrence
mapping, replay commands, per-scenario results and cost scope.
Independent export verification is saved in
`evaluation/grading/export-review-v1.json`, SHA-256
`09d00a2e71f3b1515b3ebb169223fc9ba3610ed35f6b0bfd0b82bdc90720e493`.
The final visible-milestone/grade join is `private-grading-join.json`, SHA-256
`07caa05c568cf069157cc973fc8dd56d106dd80fbc83aaa4c1507e0b5bf63add`.
`milestone-summary-final.json`, SHA-256
`4729939543dbec89cb0feed9d0f98cbee0040bf4437494ba422f5becca1521ec`,
records the complete timing, token and GPU accounting. Earlier visible-only
annotations remain unchanged; private grades do not retroactively create
agent-visible milestones.

A local operator checkpoint contains 158 code, contract, review and accounting
files in `operator-evidence-final-v1.tar.gz`, SHA-256
`d47f9017f81bbd2da95cf78e2933ea30ba9e115623f23d52408886103f3a2f8f`.
Its verified file inventory is `operator-evidence-final-v1.json`. Large native
states and original solver source snapshots remain at their recorded locations.

Both agents' completion and every submitted source occurrence were frozen before
grading. Only identical complete source maps shared a result. Evaluation
compares ordered keys, shapes, dtypes and full tensor values against independent
same-topology oracles; it does not infer correctness from the patch syntax.
Missing evidence, malformed snapshots and unsupported observer coverage remain
unknown. The bounded validation covers two updates, one seed and one small
corpus, not generalization to other workloads.

The new request service and pilot adapter passed 35 focused local tests, strict
mypy for their changed helpers, and pinned formatting/lint checks. Packet
preparation passed nine additional tests and strict checks. The observer passed
six regression tests, the optional-tool preflight helpers passed 20, and the
reused accounting helpers passed 50, each with scoped strict typing and pinned
format/lint checks. These counts describe separate validation scopes and should
not be summed as unique tests. The native gates also passed under the explicit
same-topology scope above. The post-trial grader passed 14 focused tests plus
pinned strict mypy, Ruff, Black and isort checks, independently rerun by a second
reviewer. Its negative-path tests cover incomplete cost receipts, duplicate
finishes, malformed snapshots and a confirmed incorrect scenario alongside an
unknown one. The TrainCheck package is unchanged.

The recorded solver launch, from the local evidence root after qualification
and packet export, was:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/matt/miniconda3/envs/traincheck/bin/python pilot.py freeze
PYTHONDONTWRITEBYTECODE=1 /Users/matt/miniconda3/envs/traincheck/bin/python pilot.py check-launch
PYTHONDONTWRITEBYTECODE=1 /Users/matt/miniconda3/envs/traincheck/bin/python pilot.py run
```

The independent replay uses the frozen plan on Ring18:

```sh
/home/yuxuan/experiments/traincheck-token-normalization-20261102/venv/bin/python \
  /home/yuxuan/experiments/traincheck-runtime-study-c-20261102/evaluation/grading/grade.py \
  --plan /home/yuxuan/experiments/traincheck-runtime-study-c-20261102/grading-input-v1/plan.json \
  --execute-after-both-freezes
```

`grading-payload-allowlist.json` binds the staged execution inputs;
`grading-source-inventory.json` retains every original submission. The payload
contains public submitted source and minimal grading inputs, excluding raw CLI
transcripts and private model snapshots. Preserve the frozen directories; a new
repetition needs fresh isolated output roots and regenerated qualification and
launch manifests. Per-request receipts retain the exact native training commands.
