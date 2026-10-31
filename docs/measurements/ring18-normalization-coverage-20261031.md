# Why the local-mean loss-normalization fault was missed

Audit date: 2026-10-31. Read-only source and saved-run analysis; no new training was launched by this audit. Baseline core: `754277ae87e2189cebf16b405940bef1fee5904c`; workload/tracing: `15ed99cc2d72853ebeb44a665b3a91d8923dd99e`. Source locations below refer to `b1d2953ae8688c53fc71a9da1b0a2d138f5a81c7`, whose timestamp change does not alter these relevant functions.

## Conclusion

The observed miss has three concrete causes, earlier than checker execution: the wrong scale is never recorded, numerical loss/gradient contents are absent from the API records, and the implemented/selected relation families do not express the global-token normalization equation. Fixing parameter observation timestamps alone will not close these gaps. The existing rules all activated; this is not evidence of a checker hang or wholly dormant rule set.

## Raw evidence and reproducibility

Six unmodified raw API logs were read from `ring18:/home/yuxuan/experiments/traincheck-distributed-20261031/runs/{api-fp32-full,api-fp32-control,api-fp32-fault}/payload/rank{0,1}/trace/trace_API_*.log` and saved under `/Users/matt/gitrepos/traincheck/outputs/ring18-timestamp-20261031/normalization-audit/raw/`. Export archive `api-traces.tar.gz` is 749,999 bytes, SHA256 `aaa98a3b90f79d3ea294389043343ece0852358d1662918db6f362325697ec47`. `normalization-audit/audit.json` in that local output directory records each raw log's hash and byte count, complete function/count and type-schema inventories, every direct float argument/count, unmodified cross-entropy/backward records, and unmodified failed-rule details. Original frozen rules and coverage files remain at `/Users/matt/gitrepos/traincheck/outputs/ring18-distributed-20261031/evidence-bundle/analysis`.

Reproduce the JSON analysis with:

```sh
/Users/matt/miniconda3/bin/python3.12 /Users/matt/gitrepos/traincheck/outputs/ring18-timestamp-20261031/normalization-audit/audit.py
```

Every raw log has 3,752 events: 1,876 pre/post call pairs across 90 distinct functions. All six have identical function counts, typed argument/return structural inventories, and direct float-argument values/counts. This is a coverage/type comparison, **not whole-trace equality**: no IDs/timestamps were stripped from raw records and no claims of exact output parity follow from these inventories.

## Where the fault-specific information disappears

1. `tools/distributed_probe/workload.py:459-471`: `local_tokens`, the count tensor, `dist.all_reduce`, extraction of `global_tokens`, and `loss_scale()` execute inside `session.suppress()`, outside the traced update. These values are in run metrics, not API trace inputs. The metrics are not fed into the inference engine. Healthy scale is `2/32 = 0.0625`; faulty scales are `1/10 = 0.1` on rank 0 and `1/22` on rank 1. See `loss_scale()` at lines 125-134.
2. The actual use is `loss = token_loss(logits, batch) * scale` (`workload.py:491`), not a wrapped user API with an explicit scale argument. `tracer.py:974-980` skips magic methods including `Tensor.__mul__`. `config.py:258` sets `INSTR_DESCRIPTORS=False`; `tracer.py:1166-1170,1234-1238` also excludes Tensor method descriptors. Actual raw coverage contains no `__mul__`, `.sum`, `.item`, `.div`, `.mean`, or `all_reduce` calls. `_foreach_mul_` appears four times as part of gradient clipping; it is unrelated to application loss scaling. Exact floating values `0.0625`, `0.1`, and `1/22` occur zero times anywhere within the raw args, kwargs, or returns of all six logs.
3. API primitive serialization itself preserves float values (`dumper.py:574-581`), so this is not float rounding or JSON loss. Wrapped cross-entropy appears eight times/rank/run, followed by eight `Tensor.backward` calls. Its output has metadata `shape=[16], nbytes=64, dtype=torch.float32`; `backward`'s receiver has `shape=[], nbytes=4, dtype=torch.float32`. Neither record has tensor numerical `data` or its hash. API `var_to_serializable` delegates tensor objects to `obj_to_serializable`, which explicitly calls `convert_var_to_dict(..., include_tensor_data=False)` (`dumper.py:541-543`). Enabling the tensor-hash setting does not override this API-path flag. It applies when tensor data dumping is actually requested, such as variable/model sampling.
4. The parser preserves the recorded typed args/kwargs/returns: `trace/utils.py:39-53` skips flattening those fields, and `trace/types.py:157-167` exposes them on `FuncCallEvent`. There is no missing numeric scale for the parser to recover, and no recorded loss tensor contents to compare. A later loss `.item()` used for metrics is also inside the suppressed bookkeeping scope (`workload.py:507-517`).

## Why the learned relations do not compensate

Both ranks learned the same family counts except one distinctness rule:

| Relation | Rank 0 | Rank 1 | What it can establish here |
| --- | ---: | ---: | --- |
| APIContainRelation | 117 | 117 | Call/variable containment; optional exact argument constraints |
| ConsistentOutputRelation | 222 | 222 | Returned tensor metadata equals a learned value |
| ConsistentInputOutputRelation | 246 | 246 | Equality between input/output tensor properties |
| DistinctArgumentRelation | 5 | 6 | Recorded positional argument metadata differs across calls |
| ThresholdRelation | 2 | 2 | Output tensor property bounded by a named `min`/`max` input |
| ConsistencyRelation | 0 | 0 | No variable records were fed to these API-only rank-local inferences |

- `get_events_of_funcs_with_tensors()` filters `ANALYSIS_SKIP_FUNC_NAMES` (`consistency_transient_vars.py:247-256`), including `torch.nn.functional` and `torch._tensor.Tensor.backward` (`config.py:67-91`). Thus even their available shape/dtype metadata does not generate transient consistency/threshold rules. DistinctArgumentRelation uses the same skip list (`DistinctArgumentRelation.py:42-56`).
- `get_input_tensors()` explicitly selects tensor/parameter argument types (`consistency_transient_vars.py:194-213`). ConsistentInputOutputRelation generates equality hypotheses by intersecting tensor-property values (`:692-750`); it does not use arbitrary primitive scalar arguments to infer products or ratios.
- ConsistentOutputRelation generates tensor-property-to-constant rules (`:303-433`), not a relation constraining arbitrary scalar inputs.
- ThresholdRelation reads only numeric inputs whose parameter names contain `min` or `max` (`:216-244`), and only considers output metadata close to those bounds during generation. Both learned rules are `clip_grad_norm_().output[0].is_cuda <= max_norm` and `is_leaf <= max_norm`, with `max_norm=1.0`. They compare boolean metadata because Python treats bool as int; they do **not** bound the returned gradient norm's numeric value. No threshold rule concerns loss normalization.
- APIContain can in principle constrain captured scalar child-call arguments, but the scale's multiply call was never captured; even adding capture would yield at best a reference-specific constant argument rule, not the equation `scale * global_valid_tokens == world_size`.
- FunctionLeadRelation and FunctionCoverRelation were explicitly disabled for these runs. Their event-order/coverage semantics would not supply the missing arithmetic relation.
- Adding model parameter VAR records, fixing their timestamps, or joining rank traces does not by itself establish correct normalization: in this fault, DDP still synchronizes identical faulty parameters on both ranks. Parameter equality therefore passes the fault. Correctness requires observing loss weights/counts or comparing with a mathematical/global-batch gradient reference.

All 592/593 frozen rules were active on fresh healthy and faulty traces, with zero dormant, unobserved, or unknown outcomes in the recorded coverage. Rank 0 had two healthy violations and one faulty violation; rank 1 had two in each. No fault-only rule identity was violated. Checkers can stop after the first violation for an individual rule, so 'active' does not mean every opportunity was visited.

## Healthy alarms, separately from the miss

The shared containment alarm is directly explainable. Rule 23's saved child `Embedding.forward` argument includes `_TRAINCHECK_weight_ID=139983274014448` for rank 0 (`139772221489616` for rank 1). All eight actual child calls are present in both fresh healthy and fault runs. `containment-windows.json` additionally verifies that each of the eight DDP forward intervals contains exactly one complete Embedding forward call, on the same PID/TID, in every run; this is not merely global presence. Comparing every saved receiver field with the raw child receiver shows exactly one changed field, this process-local weight ID. For healthy rank 0 the new ID is `140486160791280`; for healthy rank 1 it is `140303637676896`. `APIParam.check_event_match()` (`base_cls.py:437-463`) calls `Arguments.check_for_violation()` (`:243-250`), which compares saved values exactly. This identity literal rejects the child despite the call being present. The coverage field `identity_bearing_invariants=0` counts precondition identity clauses, not this literal in relation parameters.

The other healthy violations are DistinctArgument rules: rank-0 rule 240 on LayerNorm and rank-1 rule 519 on TransformerEncoderLayer. Their recorded failing call pairs are both in step 1 and have byte-for-byte equal `args` dictionaries, including the same `_TRAINCHECK_data_ID`, shape/dtype, and flags. The checker compares these dictionaries (`DistinctArgumentRelation.py:127-170,438-448`), so its failure is consistent with its input. The trace has no tensor values, and the dumped ID is `id(tensor.data)` (`dumper.py:474-477`), not a content identity. This evidence alone cannot prove whether the underlying input values were equal or whether allocator reuse caused the repeated ID; do not describe it as proven numerical argument duplication or a demonstrated allocator bug.

## Smallest meaningful next changes (proposals, not implemented)

1. Add a narrow typed normalization observation at the loss-scaling boundary: update/microbatch/rank, reduction semantics, local/global valid-token counts, DDP divisor/world size, and applied scalar. Keep it separate from existing independent-oracle measurements. Alternatively opt into the specific scale/multiply APIs, but denominator provenance still needs observation. This is an observability addition, not an inference-speed improvement.
2. Add and test a declared contract for this workload, `scale * global_valid_tokens ≈ world_size` (or `scale ≈ world_size/global_valid_tokens` with an explicit float tolerance and supported DDP reduction contract). This should reject both erroneous ranks immediately while allowing varying global token counts. It is a new arithmetic contract, **not** evidence the current learned engine detected it. A generic scalar-constant learner is a smaller inference extension but overfits this fixed-count probe and must not be presented as general distributed normalization verification.
3. Before claiming agent benefit, repair the cross-run identity-literal containment false alarm independently, and distinguish metadata distinctness from tensor-content distinctness. Do not hide these healthy alarms by deleting IDs in evaluation. Preserve a healthy varying-token-count control, an equal-count case where local means are mathematically valid, FP32/BF16 behavior, and explicit unsupported/unknown outcomes.
4. Retain the combined-global-batch gradient oracle as independent fault ground truth. Use it to evaluate any new contract's detection/false-positive behavior; do not train the contract on fault labels or count oracle assertions as existing TrainCheck alarms.
