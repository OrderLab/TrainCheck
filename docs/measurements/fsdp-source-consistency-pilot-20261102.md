# Native FSDP initialization: optional TrainCheck pilot

**Status: both fresh agents finished; independent repair grading is in progress.** The earlier
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
that the Halo incident involved Pythia. The model migration establishes an
authentic source of different published weights, not a record of that particular
deployment.

Two worker-local cache domains emulate logical nodes on one physical Ring18 host,
using GPUs 4/5, both NVIDIA A16. Each worker has its own local rank zero and cache
metadata; global ranks and the actual NCCL process group remain distinct. This
does not measure multi-host networking or a filesystem race. Requests use
independent copies of the official caches, not writable links to shared source
blobs. The shared repositories, environment and driver are unchanged.

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

## Completed solver runs; independent grades pending

Both trials finished without CLI or transport failures. Control submitted five
decision turns and used four GPU requests; optional TrainCheck submitted four
turns and used three requests. Every submitted source is retained. These totals
are observed execution costs, not yet a statement that either repair passed the
independent acceptance criteria.

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
only once. Private post-trial grading remains a separate cost.

Both agents proposed an unsafe-initialization mechanism before the first run:
unsynchronized FSDP could assemble inconsistent rank-local tensors. However,
their specific proposed origin was RAM-efficient loading skipping nonzero-rank
weights. Actual runtime diagnostics found the loading flags absent and rejected
that explanation, while confirming cross-rank initial-state differences and
38 changed tensors after sharding. Both submissions enabled state synchronization
in turn 2. Neither final answer identified the different cached model sources.
The result must therefore distinguish a source-supported mechanism hypothesis,
runtime-confirmed state inconsistency, and an unresolved upstream cause. It is
not evidence of full root-cause localization from source or from runtime.

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

After both agents finish, every submitted source occurrence will be frozen and
graded. Only identical complete source maps can share a result. Evaluation
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
same-topology scope above. The TrainCheck package is unchanged.
