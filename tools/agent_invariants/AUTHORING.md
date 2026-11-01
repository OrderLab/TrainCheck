# Authored objective-normalization invariant

## Contract used

Training uses PyTorch DistributedDataParallel with its default averaging of
gradients over `world_size` ranks. An optimizer-update window contains exactly
`accumulation` microbatches on every rank. Each microbatch's unscaled loss is the
SUM of per-token cross entropy for valid, unmasked tokens. The application
multiplies that scalar by the recorded built-in Python `float` `applied_scale`
and invokes backward. `DDP.no_sync` applies to every microbatch except the final
one. No GradScaler is present; any gradient clipping happens after backward.
The intended objective is mean loss over all valid tokens on every rank and in
every microbatch of that optimizer-update window. All windows are complete,
global token counts are positive, and this probe supports positive local
window totals on every rank.

`step` uses positive, one-based optimizer-update numbers; `microbatch` uses
zero-based indexing. This convention follows the capture-adapter protocol
clarification supplied before freezing; it contains no faulty-run information.

## Derivation and focused invariant

Let W denote the number of ranks, N the valid-token count for the entire update
across all ranks, and L[r,m] the token-SUM loss on rank r, microbatch m. The
intended loss is:

    J = (1 / N) * sum_r sum_m L[r,m].

With scalar multipliers s[r,m], accumulation followed by default DDP averaging
produces, before clipping:

    G = (1 / W) * sum_r sum_m s[r,m] * gradient(L[r,m]).

Matching each microbatch's coefficient to the intended objective gives the
single invariant checked at every observed loss application:

    applied_scale = world_size / global_tokens.

Consequently G equals gradient(J), subject to ordinary floating-point gradient
arithmetic and the stated DDP/accumulation assumptions. The full-window token
total already accounts for accumulation, so an extra accumulation divisor is
not part of this invariant. Unequal local token counts do not change the rule
because the supplied unscaled losses are token sums, not local means.

This checks objective coefficients for arbitrary gradients. Special cases such
as zero gradients or accidental cancellation can make a wrong multiplier yield
the same numerical gradient; such cases still fail the normalization contract.

## Requested fields and validation

`Observation` is a frozen dataclass with these required fields:

| Field | Contract and use |
| --- | --- |
| rank: int | Zero-based rank; 0 <= rank < world_size. |
| world_size: int | Positive number of DDP ranks; gradient-average denominator. |
| step: int | Positive, one-based optimizer-update number; external grouping. |
| microbatch: int | Zero-based position; 0 <= microbatch < accumulation. |
| accumulation: int | Positive number of microbatches in each complete window. |
| local_tokens: int | Positive valid-token total on this rank over the whole window. |
| global_tokens: int | Positive valid-token total across every rank over the whole window. |
| applied_scale: float | Actual finite built-in Python float used at this loss application. |

Integers explicitly exclude booleans. No values are coerced or defaulted.
Single-rank local/global totals must agree; for multiple ranks the global total
must leave at least one valid token for every other rank. Required factors that
underflow to zero or otherwise cannot be represented as finite positive Python
floats are unsupported and raise `ValueError`. Wrong types raise `TypeError`;
invalid domain values raise `ValueError`. The caller must report these as
invalid/unsupported observations, not passing decisions or successful coverage.
A finite zero or negative applied multiplier is an observed normalization error
and returns a failed `Decision`.

`Decision` is frozen and returns a stable `rule_id`, `passed`, the actual
`observed` scalar, computed `expected` scalar, and an explanation. The rule is
stateless; it does not infer missing observations or maintain a cross-rank cache.

## Tolerance rationale

The comparison is `math.isclose(observed, expected, rel_tol=1e-12,
abs_tol=0.0)`. These values concern Python scalar multipliers, not mixed-precision
tensor reductions. A 1e-12 relative tolerance permits small binary64 rounding
differences from algebraically equivalent scalar arithmetic, while requiring
approximately twelve decimal digits of agreement. Zero absolute tolerance
prevents small expected factors from passing solely because their absolute
magnitude is small; a zero multiplier cannot pass. Deviations within the stated
tolerance are intentionally accepted, so this is not an exact symbolic proof.
At the extreme subnormal boundary the floating-point comparison may effectively
require exact equality. No tolerance is tied to a batch shape or rank count.

## Assumptions and limits

A pass establishes only that this recorded scalar agrees with the coefficient
required by the supplied contract and supplied metadata. It does not establish
that the record reflects actual execution. In particular, this rule cannot
independently detect incorrect token masks or counts, a loss that is actually a
mean or weighted sum, a detached multiplier, extra backward scaling, DDP
communication hooks that change reduction, incorrect no_sync use, absent or
repeated backward calls, gradients carried over from an earlier update,
premature optimizer steps, optimizer/clipping defects, or incorrect model
behavior. Later clipping can change the final update even when pre-clipping
objective normalization is correct.

The evaluator must independently establish truthful loss-site capture, exactly
one relevant observation for every rank/microbatch, coherent step identities,
consistent world_size/accumulation/global_tokens within a window, stable
local_tokens within each rank's window, and sum(local window totals) equal to
global_tokens. Repeated per-microbatch window totals must not be added repeatedly.
A single passing observation is not evidence of complete execution or coverage.
Zero-local-token windows and partial accumulation windows are outside this
probe's contract. Mixed precision reduction error is not assessed by this
scalar-only invariant.

## Authoring provenance

This artifact was derived only from the supplied task contract. No TrainCheck
repository, history, failure implementation, trace, diagnosis, prior output
directory, or faulty-run evaluation was inspected during authoring. The access
restriction is instructional, not a filesystem sandbox; this is an authoring
feasibility probe and is not claimed to be formal blinding. Only this file and
its adjacent authored_rule.py were written/read. No distributed training was run.
