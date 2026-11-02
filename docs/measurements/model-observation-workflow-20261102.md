# Model snapshots and a fresh optional-tool comparison

Status: the implementation and native snapshot compatibility are validated.
The fresh agent comparison is being prepared. No efficiency benefit is claimed
before the new agents finish and their submissions are graded.

## Problem and implementation

In the [preceding FSDP pilot](fsdp-source-consistency-pilot-20261102.md), the
optional-tool agent read TrainCheck's callable-wrapper manual and then wrote a
79-line parameter/buffer fingerprint diagnostic. Its final diagnostic was 128
lines. It never attempted TrainCheck activation. The operation record supports
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

## Evidence

Local evidence: `outputs/ring18-source-observation-20261102/` in the canonical
TrainCheck checkout. Remote execution:
`/home/yuxuan/experiments/traincheck-runtime-study-d-20261102` on Ring18.
Earlier oracles remain under `traincheck-runtime-study-c-20261102`.

`CHECKPOINT.json`, `core-validation.json`, `optional-manifest.json`,
`qualification-reuse.json`, `protocol-reuse.json` and `README-operator.md`
record the implementation, exact inputs, validation and reproduction steps.
Large model states and raw transcripts are not committed.
