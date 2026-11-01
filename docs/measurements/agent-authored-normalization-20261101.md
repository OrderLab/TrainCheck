# Agent-authored normalization invariant: targeted feasibility pilot

This experiment asks whether an agent can turn a stated training objective into
an executable invariant that detects the distributed normalization defect from
the [Ring18 probe](ring18-normalization-coverage-20261031.md). It evaluates one
targeted bug family. It is not an RSI score, an autonomous bug-discovery result,
or an evaluation of TrainCheck's built-in invariant inference.

## Authoring and execution boundary

A separate agent, with no inherited conversation history, received the intended
global valid-token mean objective, token-summed microbatch losses, default DDP
gradient averaging, complete accumulation windows, and a scalar observation
interface. The parent selected normalization as the topic and supplied the
observation fields. The author received no faulty implementation, trace,
diagnostic result, numeric fixture, or expected equation. Access isolation was
instructional, not a filesystem sandbox or a formal blinded evaluation.

The agent derived and implemented:

```text
applied_scale approximately equals world_size / global_tokens
```

`global_tokens` covers every rank and every microbatch in the optimizer update.
The rule compares Python scalar multipliers with relative tolerance `1e-12` and
zero absolute tolerance. Its complete derivation, validation, and unsupported
cases are in [AUTHORING.md](../../tools/agent_invariants/AUTHORING.md).

The predicate was frozen before training/native-trace evaluation. A subsequent
lint-only rewrite preserved exact-float validation; original and final freeze
receipts are retained. Unit and synthetic-record tests had run before that
rewrite; the equation and tolerance did not change. The final rule SHA-256 is
`f8af836ff635cf5a8de94c0bc7e2457a3397687066c8799af89a5d414aa8df4b`.

The probe's opt-in `--observe-normalization` flag records the scalar arguments of
the actual loss multiplication through TrainCheck's existing native API wrapper.
It captures rank, update, microbatch, accumulation, local/global window counts,
and the multiplier actually applied. It adds no tensor-content serialization and
does not enable the model tracker. Count computation remains outside tracing.

[check.py](../../tools/agent_invariants/check.py) runs the authored Python
predicate offline. It requires exactly one successful pre/post pair per expected
rank/update/microbatch, validates native scalar payloads and cross-rank window
counts, and retains original event identities, timestamps, source locations,
records, and file hashes. Malformed/incomplete evidence reports an error with
unknown violation count, never a passing check. CLI exit codes are 0 for complete
pass, 1 for complete violations, and 2 for an error. Training completion is
verified separately through successful worker/supervisor exits and checkpoints.

## Matched GPU experiment

Final source revision: `fe482640d2979481419040fe3395b894cb8600fd`, based on PR #167
revision `1df0ee5d9f3c8d31c58111215dd69f11b35dfba8`. Ring18 NVIDIA A16 GPUs 0/1;
Python 3.10, PyTorch 2.2.2+cu121. The 5,504-parameter Transformer uses FP32,
dropout=0, AdamW, deterministic data, one Torch CPU thread, and four complete
optimizer updates per run. Observed runs select `--instrumentation api`; the
adapter also supports subclass mode, which this matrix does not exercise.
This is a correctness pilot, not a performance run.

The matrix contains six observed two-GPU runs: healthy/global-token and faulty
local-mean normalization at accumulation lengths 1, 2, and 3. Global window token
counts are respectively 16, 32, and 48. A seventh observed run uses one GPU and
local-mean normalization, which is mathematically correct in that configuration.
Two additional plain runs at accumulation 2 provide matched healthy/faulty
checkpoint controls. Each configuration is run once.

The first nine-run matrix used `26c77ce`. Review found extra persistent loss
locals could change temporary tensor lifetimes, including with observation off.
The final source restores the original disabled expression and avoids additional
persistent tensor aliases in the observed path. Both real-training lifetime
regressions fail with `workload.py` from `26c77ce` in the same test environment and
pass after correction. Preliminary runs are
preserved in their own directory; final claims use the corrected rerun only.

## Corrected-run results

All nine training phases exited successfully with unchanged source. The seven
observed runs completed all **104 expected rule applications**, with no missing,
duplicate, or exceptional observation calls. Counts below include both ranks
where applicable; one application corresponds to one loss multiplication.

| GPUs | Accumulation | Normalization | Completed checks | Violations | First violating update |
| --- | --- | --- | ---: | ---: | --- |
| 2 | 1 | global-token mean | 8 | 0 | — |
| 2 | 1 | local mean | 8 | 8 | 1 |
| 2 | 2 | global-token mean | 16 | 0 | — |
| 2 | 2 | local mean | 16 | 16 | 1 |
| 2 | 3 | global-token mean | 24 | 0 | — |
| 2 | 3 | local mean | 24 | 24 | 1 |
| 1 | 2 | local mean, correct on one rank | 8 | 0 | — |

Thus all three tested faulty two-GPU configurations triggered the same authored
rule, while all 56 checks in the four mathematically healthy runs passed. These
are repeated applications of one invariant to one bug family, not 104 independent
bug examples or an estimate of general detection accuracy. The separate
first-update global-batch gradient oracle agreed with the healthy/faulty
classification in every run. Both healthy and faulty runs still had identical
model replicas across ranks.

For healthy and faulty accumulation-2 runs, the complete plain/observed checkpoint
comparisons passed: **4 rank-file pairs, 548 tensor pairs**, including model,
gradients, optimizer, scheduler, CPU/CUDA RNG and provenance. Corresponding whole
checkpoint file SHA-256 values also matched. No fields were dropped or normalized
to establish equality. Plain controls were unobserved by this rule.

## Validation

The composed focused suite passed **113 tests** on macOS/Python 3.10 in 19.77 s
and **113 tests** on Ring18/Python 3.10 in 45.43 s. These are test-run durations,
not tracing/checking overhead measurements. Coverage includes actual native
records, gradients and RNG preservation, exception propagation, suppression,
temporary tensor lifetimes, malformed input, and incomplete/duplicate evidence.
The local PyTorch runs required elevated execution for OpenMP shared-memory
access; the first sandbox-only integration attempt failed on that environment
restriction, not a rule assertion.

Configured pre-commit hooks passed. Explicit strict mypy checked the eight
changed/new Python modules because the hook does not select tools/tests. Pinned
Ruff/Black/isort also checked tests, which the hooks exclude. Configured core
mypy passed 79 unchanged source modules; its legacy untyped-body coverage limit
remains. No remote CI success is claimed.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_agent_authored_rule.py tests/test_agent_invariant_observation.py \
  tests/test_agent_invariant_trace_check.py tests/test_distributed_probe_workload.py \
  tests/test_distributed_probe_tracing.py tests/test_distributed_probe_compare.py
```

## Interpretation limits

- This tests one human-selected contract and observation interface. It does not
  establish that an agent can discover which invariants to write on an unknown
  task, choose trustworthy observation sites, or improve a model autonomously.
- Counts originate in the workload. Cross-rank agreement does not independently
  establish that masks, count computation, loss reduction, or DDP reduction are
  correct. A consistently wrong count can escape this rule.
- It does not check missing/repeated backward calls, altered communication hooks,
  zero-local-token or partial windows, stochastic training, or mixed precision.
- First violating update refers to an offline trace location, not online
  detection latency. No checking-overhead or general detection-rate claim is made.
- This custom predicate bypasses the built-in relation-template limitation. It
  neither adds a relation family nor tests what existing inference could learn
  from these newly exposed scalars. Existing identity-related false alarms are
  outside this pilot.
- Complete checkpoint comparison preserves every stored value and field; it maps
  storage to CPU and does not reconstruct device placement or storage aliasing.

## Reproduction and artifacts

From a checkout at the final source revision, in the existing compatible
environment, run each case in its own fresh output directory. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/distributed_probe/workload.py --output /scratch/run \
  --steps 4 --device cuda --accumulation 2 --fault none \
  --instrumentation api --observe-normalization
python -m tools.agent_invariants.check --root /scratch/run \
  --world-size 2 --start-step 1 --steps 4 --accumulation 2 \
  --output /scratch/check.json
```

For the faulty case select `--fault local-mean`; for plain controls omit
`--observe-normalization` and select `--instrumentation plain`. A plain run has no
rule observations and must not be interpreted as passing the predicate.

Final remote artifacts:
`/home/yuxuan/experiments/traincheck-agent-invariants-20261101-v2`.
Preliminary artifacts remain at the same path without the `-v2` suffix.
Local evidence root:
`/Users/matt/gitrepos/traincheck/outputs/ring18-agent-invariants-20261101`.
The operator's exact nine commands, exit codes, environment, source hashes,
frozen rule receipts, raw API traces, checkpoints, metrics, and reports are kept
with the experiment. No credentials or training artifacts are committed.

The final evidence archive contains 266 manifest-verified files, including raw
API traces, and is 10,950,069 bytes. Its SHA-256 is
`c138f5e7e050939182ade7f60b79baad9a2c5833ca42ad4c36a237d6a549f09b`.
The source manifest covers 124 files at the stated revision. The archive receipt,
manifest, and source identities are also checked during local retrieval.
