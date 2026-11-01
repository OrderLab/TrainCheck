# Native FSDP initialization: optional TrainCheck pilot

**Status: qualification in progress; no agent comparison launched.** The earlier
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

The workload uses the official 14M-class GPTNeoX model, explicit native FP32
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
but the underlying cross-world gradient discrepancy remains unresolved. It is
not reported as ordinary rounding proved harmless.

The failed criterion remains failed; no threshold was silently relaxed. The
operator explicitly selected a separate pre-agent scope: exact initial source
coherence and full same-topology healthy trajectories, without a claim of narrow
single-versus-two-GPU numerical parity. Its gate must bind both the successful
state checks and the retained failed original criterion. Two additional bounded
single-GPU microbatch-accumulation controls are planned to characterize batch
shape effects; they do not change the submitted workload or original results.

## Planned comparison and acceptance

The proposed neutral task extends a pretrained fine-tuning pipeline to two-GPU
FSDP, supplies ordinary short-run outputs, and requests a readiness decision and
justified revision. It does not assert a bug, prescribe a diagnosis or require
TrainCheck. This is pretrained fine-tuning, not checkpoint restoration with
optimizer and step state.

Both arms receive identical editable runtime source, data, ordinary diagnostics
and execution budgets. Treatment adds the same generic optional TrainCheck
manual and implementation used in the earlier pair. Both current and reference
backends execute the submitted source. A private actual-entry observer and
independent evaluation must pass their own positive controls before launch.

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

The planned pair retains six requests of 1–16 global updates, 240 seconds per
worker request, 900 seconds aggregate CLI execution and eight decision turns.
Measure source diagnosis, runtime-supported diagnosis, first independently valid
submitted repair, visible verification and final completion separately. Within-
turn tokens remain intervals; GPU cost is supervised allocated-device lifetime,
including startup and cleanup. Qualification and private grading costs are
separate from solver costs. No tool efficiency benefit is established.

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
`prelaunch-evaluation-design.md` records the diagnosis and repair policy;
`packet-draft-preparation.json` binds local task drafts. No formal solver packet
has been exported at this checkpoint.

The new request service and pilot adapter passed 35 focused local tests, strict
mypy for their changed helpers, and pinned formatting/lint checks. Packet
preparation passed nine additional tests and strict checks. These are local
contract checks, not a substitute for the pending native service, optional-tool
parity and independent actual-entry gates. The TrainCheck package is unchanged.
