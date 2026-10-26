# TrainCheck engineering guidance

This file applies throughout the repository. Preserve TrainCheck's explicit type
and runtime contracts: catching malformed data early is part of correctness.
Follow the user's task scope and preserve unrelated work. Resolve routine choices
using these rules; ask when an unresolved choice changes public behavior or scope.

## Types and data contracts

- Fully annotate new functions and changed function interfaces, including private
  helpers, worker payloads, container element types, and return values. When
  changing an untyped helper's behavior, give it an explicit contract.
- Represent structured data with appropriate aliases, `TypedDict`, dataclasses,
  or small protocols. Prefer the simplest representation that states the real
  contract; do not introduce abstractions solely to satisfy a checker.
- Do not remove annotations or assertions, widen a type to `Any`, coerce values,
  or add ignores merely to silence errors. A necessary contract change needs a
  stated reason, caller review, and tests. Document narrow uses of `Any`, `cast`,
  `type: ignore`, and `noqa`; existing typing debt is not a reason to add more.
- Treat external JSON, framework metadata, and dataframe values as untrusted
  inputs. Validate and narrow at the boundary, using `object` for unknown values
  where practical. `cast` and `TypedDict` do not perform runtime validation.
- Required fields must be required in both the type and implementation. Validate
  them and use direct access, such as `record["func_call_id"]`; do not substitute
  a default when a required field is absent or malformed.
- `.get()`, `getattr()`, and optional types are appropriate when absence has a
  defined meaning, such as a cache miss or an unavailable framework capability.
  State that meaning and preserve it in the result. Do not blanket-ban these APIs.
- Keep missing, empty, false, unobserved, skipped, unsupported, and failed states
  distinct where they mean different things. Never turn an unknown result into
  success, zero violations, or an empty dataset through a permissive default.
- Types must describe accepted values, including legacy dataframe scalars. Do
  not claim all IDs are integers or silently coerce them unless that is the
  actual validated schema.

## Failures and compatibility

- Retain explicit assertions for internal invariants. Reject malformed external
  inputs with explicit validation and useful errors; assertions alone disappear
  under optimized Python execution.
- Catch exceptions narrowly. A broad catch is justified for cleanup followed by
  re-raising, or an explicit failure result whose caller handles it. Do not swallow
  errors, continue with fabricated data, or silently fall back after corruption.
- Keep compatibility fallbacks at clear boundaries. Distinguish an absent optional
  API from a present API with malformed metadata or unexpected runtime failure.
  Expose unsupported behavior and test supported and fallback paths.
- Annotate context-manager exception and traceback parameters. If `__exit__`
  never suppresses an exception, use `Literal[False]` for its return type.
- Prefer small, cohesive changes. Separate deliberate semantic changes from
  performance changes so reviewers can assess each independently.

## Tracing, inference, and performance

- Instrumentation must preserve application behavior: RNG streams, model state,
  return values, exceptions, and checkpoint compatibility. Test the relevant
  contract when changing wrappers, serialization, suppression, or sampling.
- Tracer bookkeeping and serialization must not recursively generate user trace
  events. Check record completeness and multiplicity, including failure paths
  and concurrency where applicable.
- Optimizations must preserve row order, dtypes, duplicates, identity, boundary
  conditions, and error behavior. Document cache ownership, data stability, and
  invalidation; test mutation or reuse when supported. A fast path needs the same
  semantics as the fallback. Inputs unsupported by the fast path must reach the
  fallback with unchanged semantics.
- Compare outputs without normalizing away meaningful differences. Dropping
  records, IDs, or array contents to obtain equality does not establish parity.
- Invariant validation must report what actually executed: activation/coverage,
  dormant rules, skips, failures, and violations. Zero violations alone is not
  evidence that the intended invariants were checked.
- For performance claims, record the baseline and candidate revisions, workload,
  data/trace identity, configuration, environment, and reproducible commands.
  Use matched conditions and repeats/order controls when variability matters.
  Separate profiled timings from ordinary runs and component timings from total
  inference time. Give the denominator for percentages, memory cost, output
  equivalence, and material limitations. Label single-run results accordingly.
- Keep large traces, datasets, checkpoints, credentials, and machine-specific
  scratch files out of commits. Store durable results in the agreed location and
  record how to find and reproduce them. Use isolated experiment directories and
  environments on shared machines; heavy cluster work belongs on allocated
  compute resources, not login nodes.

## Validation

- Use the repository's Python requirement in `pyproject.toml` and the tool versions
  and options in `.pre-commit-config.yaml` and `.github/workflows/`. Do not disable
  hooks, signing, assertions, or checks to make a change pass.
- Run the configured checks for changed code, for example
  `pre-commit run --files path/to/changed.py`, and relevant CI checks before
  claiming readiness. Run `pre-commit run --all-files` in an isolated clean
  checkout when broader validation is needed. Hooks can edit files: inspect and
  retain only changes belonging to the task.
- CI is a minimum, not complete coverage. Currently the mypy hook is scoped to
  `traincheck/`, untyped bodies are not fully checked, tests have lint/format
  exclusions, and tools-only changes may skip CI. Explicitly type-check changed
  reusable tools and run relevant tests even when automation does not select them.
- Fully check new modules and reusable helpers; use stricter mypy options such as
  `--check-untyped-defs`, `--disallow-untyped-defs`, and `--disallow-any-generics`
  for their scope as appropriate. For legacy modules, compare with the base and
  introduce no new errors. Do not expand a focused task into a repository-wide
  strictness migration. Report any remaining errors and checker coverage limits.
- Behavioral bug fixes need focused regression tests that fail before and pass
  after the fix. Exercise meaningful negative paths: missing required fields, wrong types,
  malformed records, propagated exceptions, and affected fallback behavior.
  Add state, RNG, cache, or concurrency tests when those contracts change.
- Run existing tests relevant to affected callers and compare outputs for
  optimizations. Do not add tests that merely repeat implementation details.
  Documentation-only edits normally need content/link and diff checks, not a
  runtime test suite. Report exactly what ran and what could not run.

## Git and review

- Inspect branch, worktree status, and applicable guidance before editing. Preserve
  unrelated changes and shared environments. Use an isolated branch/worktree when
  needed; stage named files or hunks, never sweep unrelated work into a commit.
- Checkpoint coherent progress with small, signed commits and focused PRs as
  authorized. Preserve configured signing; if signing fails, report the failure
  instead of silently producing an unsigned commit. Do not merge without explicit
  user authorization.
- PR descriptions should state the problem, final approach, behavior/contracts
  preserved or changed, validation, and limitations. Include measured before/after
  numbers for performance work, with units, baseline, and end-to-end context.
  Do not invent improvement numbers for documentation or correctness-only work.
- State dependencies between stacked PRs. Recheck the composed result after
  rebasing or integrating changes; a branch-local pass does not prove the final
  stack works. Keep experiment evidence separate from unverified expectations.
