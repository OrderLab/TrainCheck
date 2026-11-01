# Data-dependent DDP training: optional TrainCheck pilot

**Status: one matched agent pair running, 2026-11-02.** Native qualification,
held-out positive controls, optional-tool preflight and packet review passed
before launch. The pair started after the 2026-11-01 21:30:49 UTC protocol freeze.
This is private operator documentation, excluded from agent task packets. No
agent-efficiency benefit is established. Results from the earlier
[tensor-parallel clipping pilot](real-rank-runtime-pilot-20261102.md) are retained:
its four agents localized the cause from source before their first GPU request,
and neither treatment activated TrainCheck.

## Why this case

The next case reproduces the native Hugging Face Trainer behavior discussed in
[Transformers issue #34242](https://github.com/huggingface/transformers/issues/34242)
and addressed by [PR #34373](https://github.com/huggingface/transformers/pull/34373).
When ranks have different numbers of valid target tokens, averaging their local
mean-loss gradients does not yield the gradient of the global token-mean loss.
DDP can keep model replicas bitwise identical while training the wrong objective.
Replica agreement alone therefore cannot diagnose this problem.

The relevant runtime evidence is the actual example assignment and valid target
count on each rank, the denominator reaching the model's native loss function,
and the resulting gradient before the optimizer update. The qualification uses
counts of 30 and 78 per rank versus a global denominator of 108. Equal-count
controls distinguish this condition from a general distributed-training failure.
This makes the case sensitive to cross-rank execution state; it does not prove
that an agent cannot derive the problem by reading the source and data. Such
source-based successes remain valid results.

## Exact workload and provenance

- Vulnerable native Transformers revision:
  `a17f287ac039f92835b5cd9bd8ee28b584c9f65e`.
- Corrected native revision: `d21dbd1520937c993de1409215b1418bd6be74a1`.
  The correction requires `average_tokens_across_devices=True`; the disabled
  flag is a separate negative control.
- Native `Trainer` and `Qwen2ForCausalLM`, with 115,520 parameters, two layers,
  hidden width 64, four attention/KV heads and sequence length 64. The model's
  literal `**loss_kwargs` activates the historical Trainer path. Llama at this
  revision uses a different signature and would not qualify the same path.
- Float32, AdamW, learning rate 0.001, zero weight decay, gradient clipping disabled,
  constant learning rate, seed 1234, deterministic settings and TF32 disabled.
- Four examples per optimizer update, one equivalent single-GPU batch versus
  two-rank DDP, with gradient accumulation of one or two microbatches.
- A controlled local text fixture derived from repository README bytes, with
  fixed tokenization and equal/unequal valid-length variants. No pretrained
  weights or external dataset downloads. This reproduces a real framework
  defect with a small fixture; it is not a production-quality training study.
- Ring18 NVIDIA A16 GPUs 2 and 3, Torch 2.2.2+cu121, Python 3.10.14 and
  Accelerate 1.0.1. Additional dependencies are installed in an isolated
  experiment environment. The shared environment remains unchanged.

All 4,715 files in each native source tree match its downloaded upstream archive.
The archive SHA-256 values are
`8977b0f525446b5ea80db0444a48b4ee5dac30373a13c64e8d979e4281638c65`
and `2d12575c66970d5f04a8fa9814ca1136f73de3d54759af20b1ea9df43fb85274`.
The shared environment package-list hash is
`70eda80894c761ff4c75e673681e86749b74158756cd8792d39027d923829fb4`.

## Qualification and numerical error

Sixteen completed native runs cover equal/unequal token counts, accumulation
factors one/two, vulnerable/corrected/flag-disabled code, the equivalent
single-GPU reference, repetition and observer removal. Two earlier setup failures
are retained. These 18 attempts consumed 402.53 allocated GPU-seconds, separately
from later optional-tool preflight and solver work.

| Comparison against the equivalent single-GPU objective | Measured result |
| --- | --- |
| Vulnerable, unequal counts, accumulation 1 | Gradient relative L2 error up to 38.27%; model maximum absolute difference about 0.00582 |
| Vulnerable, unequal counts, accumulation 2 | Gradient relative L2 error up to 37.21%; model maximum absolute difference about 0.00594 |
| Corrected with global averaging | Gradient maximum absolute difference at most 8.57e-8 and relative L2 at most 3.29e-7; optimizer-state maximum absolute difference at most 1.12e-8 |
| Equal-count controls | Gradient and optimizer-state comparisons pass |
| Rank agreement within every two-rank run | Model replicas match bitwise, including in the vulnerable runs |
| Repetition and forward/loss observer removal | Bitwise equality of captured model, optimizer and CPU/CUDA/Python/NumPy RNG; 718 tensor and 10,448 scalar comparisons per comparison |

The initial model comparison used absolute tolerance 1e-6 and relative tolerance
1e-4. It rejected some corrected results: model differences reached about
4.49e-6, while equal-count accumulation controls also reached about 3.21e-6.
Small floating-point gradient differences can be amplified by Adam near zero.
The original strict report is retained. A separate calibrated report uses model
absolute tolerance 1e-5. Gradient/optimizer absolute tolerance remains 1e-6;
relative tolerance remains 1e-4 for all three components. This is
chosen during qualification, before any agent output, with a margin of over 500
between the model tolerance and the observed faulty-model deviation. It is not
bitwise equivalence between single-process and distributed arithmetic.

All observed ranks, updates, examples and native calls must be present. Missing
activation, malformed snapshots and failed workers are errors, not successful
zero-difference results. No parameter, optimizer array or RNG contents are dropped
to make comparisons pass. Observer parity covers the recorded state, not
unobserved application state or live object aliasing.

## Additional prelaunch checks

The scalar-argument TrainCheck preflight observes native
`torch.nn.functional.dropout` calls: 12 entry/return pairs per rank, with no
unmatched records. Recorded model, optimizer and RNG state is bitwise identical
to the ordinary matched run (718 tensor and 10,448 scalar checks), as are all
162 gradient comparisons. This verifies the tested direct wrapper interface;
it does not qualify arbitrary tensor metadata, model tracking or the inference
engine. Its measured cost is 34.444 allocated GPU-seconds. A preceding import-only
attempt failed because the CUDA library path was missing; its GPU cost was not
measured and is retained as unknown, not zero.

The independent grader attaches observers to the actual submitted Trainer,
model and optimizer. It does not replace the submitted training pipeline with
the qualification worker. An actual-entry replay matches every recorded state,
RNG and gradient value from the archived native parent execution bitwise.

Fresh seed 5678 uses a separate fixture whose global valid-target counts vary
across three updates: 124, 140 and 152. Corrected actual-entry DDP runs with
accumulation 1 and 2 pass all 58 declared comparisons against the equivalent
single-GPU reference. Maximum held-out differences are 1.121e-6 for gradients,
1.0548e-5 for model parameters and 1.576e-7 for optimizer state. These pass the
unchanged **combined** absolute-plus-relative tolerance; a maximum difference
can exceed the absolute term alone. The actual-entry replay plus three held-out
controls cost 87.914 GPU-seconds, separately from solver requests.

The frozen independent grading contract has seven scenarios: equal, unequal and
held-out data, each with DDP accumulation 1/2, plus single-GPU compatibility on
the original unequal-length data. It requires actual sample coverage, full
model/gradient/optimizer comparisons and exact recorded RNG against a matched
same-world corrected execution. It does not compare single-GPU RNG with two-GPU
RNG. A valid repartition may change rank-local sample assignments while preserving
each global four-example window; no particular normalization implementation is
required.

Native ordinary eight-update service runs verify both backends. A first pair of
service runs revealed that native serialized training arguments contained an
operator path whose experiment name could reveal the case. The service now
passes a relative output path to the same destination. Two fresh ordinary runs
confirm neutral serialized paths and byte-identical final model files relative
to the earlier runs. Both sets of receipts are retained. This service qualification
costs 74.882 GPU-seconds in total; no binary checkpoint is edited or normalized.

## Frozen agent comparison

The matched pair receives the same native source, local fixture, ordinary
backend logs, model and resource limits. The task asks whether the extension from
single-process to DDP training is ready for a longer run and what to do next.
It neither asserts a bug nor asks for any particular diagnostic. Logged backend
training losses are not presented as directly comparable held-out quality curves.

The treatment additionally receives the unchanged generic TrainCheck observation
manual and implementation. It may choose custom logging or no instrumentation.
The control has the same ability to author arbitrary local diagnostics and use
the remote execution service. Neither arm receives issue IDs, this report,
private observations, a targeted invariant or grading feedback.

It uses the same native CLI model/settings and accounting protocol as the previous
pilot: gpt-6-astra/high, priority service, CLI 0.160.0, six requests, 1–16 global
updates per request, 240 seconds per worker phase, 900 aggregate CLI seconds and
eight decision turns. Both the two-GPU `current` and one-GPU `reference` execute
the agent's submitted source; the latter is not a hidden corrected source.
Allocated GPU cost uses two or one devices accordingly and includes startup and
failed requests. Qualification and independent grading costs remain separate.

Record source-supported localization, runtime-supported localization, earliest
submitted source that later passes independent grading, agent-visible repair
verification and final completion separately. Preserve native token counters,
cache subsets, measured service waiting, every submitted source and raw events.
Within-turn token costs may only be intervals. Availability or manual reading
does not count as actual TrainCheck use.

An independent packet comparison confirms that all 1,850 exported native runtime
source/README/license files match the upstream parent archive, and all 12 actual
training examples retain their qualified arrays. The control has 1,862 files;
the treatment has 1,894. Only the initial prompt differs among shared files, by
one sentence offering optional access. The 32 treatment-only files are the
generic manual and 31 frozen TrainCheck source files. The source and data remain
visible, so this review does not establish that runtime observation is necessary.

## Evidence and reproduction

Local evidence root:
`/Users/matt/gitrepos/traincheck/outputs/ring18-token-normalization-20261102`.
Remote evidence root:
`/home/yuxuan/experiments/traincheck-token-normalization-20261102` on Ring18.

- `provenance.json`: native archive/tree, environment, corpus and operator hashes.
- `qualification-strict.json` and `qualification-model-calibrated.json`: separate
  full comparisons with original and calibrated model tolerances.
- `execution-ledger.jsonl`: successful runs and retained setup failures, commands,
  elapsed time, devices and allocated GPU costs.
- `operator/`: versioned qualification workers, analysis, regression tests and
  commands; snapshots and native outputs remain in the remote experiment tree.
- `TASK.md`, `pilot.py`, `protocol-reuse.json`: neutral task and isolated service
  configuration. Final packets are under
  `/private/tmp/traincheck-ddp-solvers-20261102-v2`.
- `launch-plan.json`, SHA-256
  `1046652737f8fe2a446f81d5c864a499a64ac9cdf9506ee6a157640f9cace98e`:
  exact model/settings/budgets, execution order, code, packet and five gate hashes.
- `actual-entry-positive-gate.json`, SHA-256
  `d66be5d0c9b0acb7f61885df30e24a19d6786ead61f5d6a708a459e4f637991d`:
  actual-entry parity and fresh-seed positive controls.
- `independent-grading-contract.json`, SHA-256
  `4ea06f939a254117cfb8fbb10c7b21e9f000b16f15d6e28ce520fe53da5977d5`:
  the seven scenarios and unchanged numerical/coverage criteria.
- `prelaunch-neutrality-review.json`, SHA-256
  `a79346c2664a1e63796b1557bbec76906d1c08784b69fcd6274515bbc692888f`:
  independent packet comparison and evidence review.

The operator request service and qualification code do not modify the TrainCheck
package. Large native sources, data, checkpoints and traces remain outside Git.
The PR records reproducible evidence and limitations; it does not claim a general
debugging speedup or silently promote a source-readable case to a runtime-only one.
