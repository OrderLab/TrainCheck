# Parameter observation timestamp correction on Ring18

## Change and measurement scope

Baseline: `3bdb3ac8566954b187798ae7222189464f28e312` (#166).
Training candidate: `b1d2953ae8688c53fc71a9da1b0a2d138f5a81c7` (#167).
The later coverage-observer correction is separate from the training code; its
revision and rerun are recorded below. No training artifacts are rewritten.

Registry samples previously reused `last_update_timestamp`, which tracks Python
attribute assignments. In-place optimizer updates need not invoke that hook.
Both parameter trackers now timestamp `sample` and `selective-sample` at the
start of metadata observation, without updating mutation bookkeeping. Assignment,
call, iteration, and observer event timing remains unchanged. Suppressed and
retired variables remain silent; conversion failures still propagate.

These are observation times, not exact mutation times. The probe deliberately
samples before and after the whole training update. Its after-update sample lies
after `AdamW.step`'s API post-event; this correction does not manufacture an
APIContain relation placing the change inside that call. Concurrent application
mutation during metadata capture remains unsupported. No synthetic timestamp
offsets, row sorting changes, deduplication, or inference rule changes are used.

## Matched two-GPU experiment

Ring18 GPUs 0 and 1: NVIDIA A16, PyTorch 2.2.2+cu121, Python 3.10.14. Existing
`DS-1801` environment reused without installation or modification. The unchanged
probe is a 5,504-parameter Transformer (27 parameter tensors), DDP with two
microbatches per update, AdamW, clipping, scheduler, deterministic synthetic
batches, dropout zero, and four updates. Save/resume cases split at update two.
See [the original qualification](ring18-distributed-20261031.md) for the complete
workload and independent global-batch gradient oracle.

All nine phases completed with exit zero, unchanged source hashes and no
supervisor stop: fresh old-code subclass, fresh plain FP32, candidate subclass
FP32 reference/control/fault, candidate BF16, candidate subclass save, plain
resume from the new subclass checkpoint, and subclass resume from the previous
plain checkpoint. Only idle GPUs were selected. Each phase was bounded to 180 s;
the supervisor's memory accounting covers its process group, not torchrun's
separate worker sessions. These are correctness trials, not an overhead study.

### Timestamp and record evidence

The paired FP32 denominator is **216 parameter/update pairs**: 27 parameters ×
4 updates × 2 ranks. Each pair has one before and one after record.

| Measurement | Fresh baseline | Candidate FP32 |
| --- | ---: | ---: |
| Complete before/after pairs | 216/216 | 216/216 |
| Changed data hashes | 216/216 | 216/216 |
| After samples not later than optimizer post-event | 216/216 | 0/216 |
| Before/after timestamps equal despite changed hashes | 54/216 | 0/216 |
| Raw API records, both ranks | 7,504 | 7,504 |
| Raw VAR records, both ranks | 594 | 594 |
| Duplicate API event keys / malformed call pairs | 0 / 0 | 0 / 0 |

Across all six candidate subclass phases, **1,080/1,080 parameter/update pairs**
have complete samples, changed hashes, strictly increasing sample times, a before
sample preceding optimizer pre, and an after sample following optimizer post.
No missing/duplicate samples, unexpected samples, or malformed API call pairs
were found. Plain phases emitted no trace records. The existing tensor hash
samples the first row; it is not a proof of full tensor equality.

### Complete application-state evidence

All **eight complete checkpoint comparisons** pass, covering both ranks and 274
tensor pairs per comparison. Model parameters, gradients, optimizer state,
scheduler state, saved CPU/CUDA RNG and all provenance fields are compared without
dropping fields or normalizing IDs/values. Whole checkpoint file hashes also
match. The comparisons cover baseline/candidate, plain/candidate, fresh control,
both checkpoint directions, BF16, the faulty training run, and the saved halfway
state. The loader maps tensors to CPU; storage aliasing and original device
placement are not reconstructed by this comparison.

The independent gradient oracle is unchanged: healthy FP32 maximum error
`4.470348358154297e-08`, healthy BF16 `0.0005645751953125`, faulty local-mean
normalization `0.06958316266536713`. The fault still leaves ranks in exact
parameter agreement. This oracle is separate from TrainCheck invariant checking.

## Regression validation

- Same 21 focused tests: old source **8 failed / 13 passed**, candidate **21 passed**.
  Tests cover both trackers and sample modes, actual in-place mutations/hashes,
  observation-start ordering, RNG/value preservation, unchanged mutation times,
  registry staleness, suppression, retired ownership, and conversion exceptions.
- Real SGD API/VAR disk-trace test: old source **1 failed / 2 passed**; fixed
  version passes within **66 affected tests** on macOS. This verifies sample
  multiplicity, API chronology, changed hashes, and exact model/loss behavior.
- The combined affected suite, including all new regressions, passes on Ring18:
  **87 passed in 36.19 s**, with CUDA visible to the existing CUDA-enabled Torch
  build. This does not fix the separate CUDA-hidden startup limitation recorded
  in the original qualification report.
- Pinned pre-commit hooks pass. Configured mypy passes on 79 core files; both
  changed test modules pass strict scoped mypy and explicit pinned Ruff, Black,
  and isort checks. Legacy untyped bodies remain outside full checker coverage.

## Normalization coverage

The [raw-trace diagnosis](ring18-normalization-coverage-20261031.md) explains the
previous API-only miss and healthy false alarms. The applied normalization scalar
was not traced, numeric loss data was excluded, and current relations do not
encode global-token normalization. Correct timestamp chronology alone does not
supply that missing mathematical contract.

### Combined API and parameter inference

Rank-local inference on the corrected full FP32 reference learned 596 rules on
rank 0 and 605 on rank 1, using pandas, one worker, and the same disabled
FunctionLead/FunctionCover families as the original study. These are new frozen
rule sets; they must not be compared by index with the earlier API-only rules.
Inference completed in 31.91 s and 20.13 s respectively (single operational runs,
not a performance comparison).

The first fresh-control/fault checks each exposed one observer error:
`liveness.end_time must be a number`. Pandas supplies `numpy.int64` from
`Series.max()` for final state endpoints, while the coverage helper accepted only
builtin numeric types. These are valid finite timestamps, not absent state. The
original reports retain that rule as unknown and overall counts as null.

Observer-only correction `a6232194b5395bbc7417208ac87161510d3ba5d3` validates the
three observed timestamp fields at their dataframe boundary. NumPy integer
nanoseconds remain exact integers, including beyond `2**53`; finite floating
scalars require lossless conversion. Boolean, datetime/duration, missing,
nonfinite and otherwise malformed values remain errors. Generic external JSON
validators, inference logic, and checker predicates are unchanged.

Two real-pandas lifetime cases fail before this correction. All 55 targeted
coverage tests pass after it locally, including observer-on/off checker-result
parity for both a satisfied and a violated rule, precision, and invalid types.
They also pass on Ring18 (**55 passed in 4.34 s**). The composed local suite for
both fixes passes together (**142 passed in 22.61 s**). The changed reusable tool
and its new tests pass strict scoped mypy plus pinned lint/format checks; the
configured mypy hook does not select tools/tests, so those explicit checks matter.
The fixed checks use a separate `checking-source` snapshot and `analysis-fixed`
outputs, retaining the original failed observations and reusing byte-identical
trace files and frozen rules. No training or inference is repeated for this fix.

All four corrected checks complete successfully, with no observer/checker errors,
unknown outcomes, or dormant rules:

| Rank | Rules active | Healthy violations | Fault violations | Fault-only rule identities |
| --- | ---: | ---: | ---: | --- |
| 0 | 596/596 | 9 | 8 | None |
| 1 | 605/605 | 15 | 17 | DistinctArgument rules 240 and 293 |

Each recovered ConsistencyRelation passes. Every previously completed per-rule
report is exactly unchanged, including decision arrays and IDs; input trace/rule
hash maps match before and after the observer fix. This comparison concerns
per-rule results, not whole-report equality: harness provenance and the formerly
unknown rule necessarily change. Active coverage does not imply every opportunity
was visited, since a checker can stop at a rule's first violation.

The two rank-1 fault-only alarms concern repeated LayerNorm input metadata at step
4 and Linear input metadata at step 3. Each failing pair has equal serialized
argument dictionaries, including a repeated `_TRAINCHECK_data_ID`, but no numeric
tensor data/hash. Neither rule constrains normalization or its coefficient.
These are observed fault-only metadata signals; the evidence does not establish
that they diagnose normalization, prove allocator reuse, or prove equal numerical
inputs. The healthy violations remain visible. An agent-benefit or reliable
normalization-detection claim is not supported by this pilot.

## Reproduction and artifacts

Local evidence root:
`/Users/matt/gitrepos/traincheck/outputs/ring18-timestamp-20261031`.
Remote root: `ring18:/home/yuxuan/experiments/traincheck-timestamp-20261031`.
The previous evidence root remains unchanged. Large traces and checkpoints are
outside Git. `results.json` preserves phase provenance, source hashes, every
sample pair's actual times/hashes, full metrics, and complete comparisons.
`run_matrix.py` records exact commands and chosen GPU usage before every phase;
`audit_results.py` performs the validated audit. Source copies use the previously
verified baseline plus the exact signed commit diff, with all four changed file
hashes verified after upload. An interrupted full archive upload was not used.

The downloaded `evidence.tar.gz` contains **185 manifest-verified files**,
3,464,265 bytes, SHA256
`6c18774e8d5611a368e0e28295e15b4703870c697abfb594ffdc9d9b5b01250b`.
It preserves original and corrected coverage outputs as well as training states.
All 161 training-source Python files and all 162 checking-source Python files
match their respective committed revisions exactly. The final process audit
finds no remaining study processes and all eight GPUs idle. See
`retrieval-verification.json`, `fixed-coverage-comparison.json`, and the extracted
`evidence-bundle/evidence/` for verification details. The separate six-log API
audit archive is identified in the normalization report; large new API traces
remain on Ring18 with their hashes recorded in `results.json`.

Run from the remote root using its existing runtime:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/yuxuan/miniconda3/envs/DS-1801/bin/python run_matrix.py
PYTHONPATH="$PWD/source" /home/yuxuan/miniconda3/envs/DS-1801/bin/python audit_results.py
```

The matrix requires fresh phase output directories and refuses existing ones.
For a new study, copy the source and operator scripts to another isolated root
and update their explicit root constants. Do not overwrite the preserved runs.

Limits retained: no FSDP, changed topology, partial accumulation windows,
stochastic dropout, DataLoader cursor, FP16 GradScaler, or agent-effect claim.
