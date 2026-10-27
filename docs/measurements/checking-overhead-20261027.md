# Tiny-Llama selective API instrumentation, 2026-10-27

## Result and scope

On this 6,163,712-parameter TorchTitan Llama workload, selective API instrumentation costs **22.64x the plain steady-step time** (median paired slowdown across three fresh pairs, range 22.62–22.67x). That is **+2164% overhead relative to plain training**. The median of per-run medians is 43.861 ms plain and 994.425 ms selective. This is a deliberately tiny model, batch 1, sequence 256, with every step instrumented; it is not a claim about larger training jobs or interval-10 sampling.

All three pairs and a different-seed, 20-step holdout have matching SHA256 digests of initial/final parameter, gradient and buffer bytes, matching DTensor metadata, losses, and Python/NumPy/Torch CPU/CUDA RNG-state evidence. The holdout has 22.77x steady-step slowdown.

This measures **API-only collection for checking**, followed by separate offline validation. The policy selects no model tracker. Native TorchTitan FSDP remains enabled. FSDP model tracking and a concurrently running live checker are not validated by these results and remain separate work.

## Timing

Native `train_step` timings include `start_step` and the final CUDA synchronization. The steady statistic uses steps 3 onward. Configuration, model construction, evidence capture, outer train-loop housekeeping, final queue draining, and process shutdown are outside the step timer. The experiment sets sampling_interval=1 and warm_up_steps=0.

| Scope | Plain | Selective API |
|---|---:|---:|
| Pair 1 median steps 3–10 |43.969 ms|995.449 ms|
| Pair 2 median steps 3–10 |43.858 ms|994.425 ms|
| Pair 3 median steps 3–10 |43.861 ms|992.226 ms|
| Holdout median steps 3–20 |43.801 ms|997.229 ms|
| Pair 1 total 10 training steps, including cold first step |1.824 s|12.531 s|
| Pair 2 total 10 training steps |0.845 s|12.520 s|
| Pair 3 total 10 training steps |0.846 s|12.508 s|
| Holdout total 20 training steps |1.288 s|22.524 s|

The first plain pair has cold initialization cost, so total-time ratios vary much more than the steady-step ratios. These totals must not be substituted for the steady-step statistic.

A separate earlier full-API reference pair measured 44.570 ms plain and 5,079.953 ms full tracing per steady step. Selective's 994.425 ms is 80.42% lower than that full-trace time, but this is a cross-job policy comparison with a single full-trace reference, not a repeated optimization comparison.

## Invariant validation

The frozen 722-rule set was inferred from the full API reference: 419 APIContain, 156 ConsistentOutput, 142 ConsistentInputOutput, 5 DistinctArgument. One-worker inference took 58.654 s operational process time and 1.89 GB peak process-group RSS; this includes imports and parsing.

Both the fresh same-seed trace and the different-seed holdout report:

- 722 completed rules; 719 active (718 pass and 1 violates), 3 dormant.
- 1 violated rule, 0 checker/observer errors, 0 unknown outcomes, 0 skipped variable checks (none requested by this API-only policy).
- Exact complete CheckerResult parity with the direct checker for 722/722 rules. IDs, array contents and order are preserved; identical nonstandard NaN literals are compared only to the same literal at the same path, remaining distinct from null/missing.
- 114,373 actual decisions in the 10-step trace; 218,713 in the 20-step holdout.

The remaining violation is rule 614, which learned a literal process-specific CUDA stream handle. It is retained unchanged and is a portability false positive, not evidence of divergent training. Dormant rules 216 and 276 require a reference worker thread ID absent from the fresh run; 717 was already dormant in the reference. Context-dependent rules 661, 719 and 720 now activate.

All four selective traces passed independent raw/parsed-row and complete-call-pair audits, with no missing or duplicate scoped PRE/POST pairs. Matching report input hashes are checked against the actual audited files and the frozen invariant set. The full-reference parser reported skipped _CachedTorchDispatchMode contexts; those warnings were absent from these final selective audits. Row-count parity alone does not certify every possible context relationship.

The direct offline checker took 29.980 s for 10 steps (1.26 GB peak process-group RSS) and 54.520 s for 20 steps (1.84 GB). The coverage-observed runs took 53.364 s and 105.459 s respectively; these include observer overhead and are not ordinary checking timings. None of these offline durations is added to or presented as concurrent training overhead.

## What was fixed before using these timings

The first selective harness bypassed the production JSON-loading boundary, so integer argument-selector keys were not recognized by the dumper. Its approximately 14.1x timing is invalid as a complete checking result.

After fixing the harness, production policy generation still produced 61 violated rules: 54 missing-return cases, 1 incomplete distinct-argument case, 5 positional-only/variadic selector cases, and the real stream-handle portability issue. Context constructor inputs also caused avoidable dormancy.

PR #161 repairs the required evidence by unioning return, complete-argument and context requirements, and requesting full argument metadata when the selector cannot represent a required argument. Supported selectors stay selective. It preserves the 722 rules and their predicates. The final trace reduces 61 violations to 1 and raises active rules from 716 to 719; no rule or identity field was removed to obtain that result. This correctness repair costs more than the incomplete policy, so no speedup is claimed for it.

## Trace volume and actionable follow-up

| Representative 10-step trace | Full API | Selective API |
|---|---:|---:|
| Observed API names |367|244|
| API records |442,918|416,364|
| Complete calls |221,459|208,182|
| JSONL bytes, including newlines |255,365,856|143,453,745|

Selective reduces records by 5.995% and bytes by 43.824%. Each steady step still emits 29,864 records for 14,932 calls and about 10.20 MB. Metadata-only records form 74.36% of events and 66.41% of bytes. These are counts and sizes, not runtime percentages.

The six largest selective API byte contributors are registry lookup, export-state lookup, dispatch-rule lookup, checkpoint assertions, dispatch-rule search, and checkpoint storage-key lookup. Many lightweight framework calls remain selected by the frozen invariants. Dropping those records would change the evidence required by those rules; record reduction needs semantic/coverage review, not a silent filter.

File writing already runs on a background thread. Metadata conversion and JSON encoding happen on the producer before enqueue. A separate profile launcher has passed CPU smoke and strict typing/lint checks, but its H100 run has not been submitted because the shared SSH connection disappeared. No component time shares are claimed yet. Final queue drain alone cannot establish the cost of overlapping writes.

## Reproduction and provenance

- Baseline main: 6dccb47d0a8bc192fbe7cbbea700b66106bc2381.
- Full-reference composed source: b34e309; job 55324690.
- Corrected selective composed source: c2233f119e0eb0f01827a7ebb593b259f82596f2; job 55325270. Later 8cc73ab changes only a test-discovery fixture.
- TorchTitan: 2210414902d84d1219fd63ab8efeb24375a2a1a1.
- One H100 80 GB on Expanse, Python 3.12.3, Torch 2.16.0.dev20261012+cu130, CUDA 13.0, CUDA hashing enabled through PR #158. CPU validation uses labmach Python 3.11.4/Torch 2.16.0.dev20261012+cpu, one worker and single native/intraop/interop thread, bounded to 16 GiB and phase deadlines.
- Seed 42 paired order: plain/selective, selective/plain, plain/selective; 10 steps. Holdout seed 43: plain/selective; 20 steps. No parameter updates or setup math were replaced.
- Invariant SHA256: bad55c222f4eb3406bdfaec527ab1a75eaf66cdbe8f06c42d7f0775f4ee9224b.
- Selective source archive SHA256: 33363cd5dac62e800545b55924ad49d887f1634c5a588cba04a05c16255270bd.
- Final results archive SHA256: f8f7f9077ae322bcb9c766b57496c94fc2eca3d3e551f00d0039345a8976bdff.
- Actual policy SHA256, identical in all four selective runs: df4c91296774b2dfd3af09d0a25e21e25f26ff1de93cb97ab959fa77379c3211.

Durable downloaded artifacts are in the experiment owner's `outputs/checking-overhead-20261027/`, including source/runtime manifests, exact scripts, results, raw profiles when available, `job-55325270/paired-summary.json`, `final-policy-validation-summary.json`, `trace-volume-final-policy.json`, and `offline/results/final-policy-*`. Large traces stay outside Git. Expanse copies are under `/expanse/lustre/scratch/yjiang21/temp_project/traincheck-checking-20261027/job-JOBID`; this scratch is retention-limited. Lab artifacts are in `/home/yuxuan/experiments/checking-overhead-20261027`.

Reproduction uses saved `selective.sbatch`/`run_phases.py` on an allocated GPU, then `offline-final-policy.sh` and `audit-final-policy-repeats.sh` on labmach. Locally run `python analyze_final_trials.py` and `python trace-volume-final-policy.py`. The hash-checked coverage summary uses `analyze_reference_validation.py` against the remote study root with `--fresh-prefix final-policy --policy-status final-production-candidate`, because audit records contain remote absolute paths. `python analyze_final_policy_attribution.py` uses the downloaded mirror and local final traces to append all four audits and exact stream-handle differences. The saved source archives pin the actual measured code rather than a future branch tip.

PR #159 contains the workload harness; PR #158 supplies the H100 hashing correctness fix; PR #160 supplies actual coverage observation; PR #161 supplies the production policy repairs. Each targets main independently. The GPU experiment uses their composed code. No PR was merged by this study.
