"""Summarize existing TrainCheck coverage without rerunning or changing its rules.

Use the TrainCheck checkout on PYTHONPATH. This reader validates the fields used
for its conclusions; complete rule/trace evidence stays in the hashed input file.
It never treats a checker process's exit status as an invariant outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tools.fresh_validation.contracts import (
    JSONObject,
    boolean,
    integer,
    json_object,
    mapping,
    required,
    sequence,
    string,
)


def nullable_bool(value: object, label: str) -> bool | None:
    return None if value is None else boolean(value, label)


def count(value: object, label: str) -> int:
    result = integer(value, label)
    if result < 0:
        raise ValueError(f"{label} cannot be negative")
    return result


@dataclass(frozen=True)
class RuleView:
    index: int
    relation: str
    description: str | None
    status: str
    coverage_status: str
    passed: bool | None
    active: bool | None
    dormant: bool | None
    skipped: bool | None
    decisions: int | None
    identity_bearing: bool


def read_rule(value: object, position: int) -> RuleView:
    rule = mapping(value, f"per_invariant[{position}]")
    index = count(required(rule, "invariant_index"), "invariant_index")
    if index != position:
        raise ValueError("Invariant indices must retain their complete source order")
    status = string(required(rule, "status"), "status")
    coverage = string(required(rule, "coverage_status"), "coverage_status")
    if status not in {
        "completed",
        "checker_error",
        "observer_error",
        "unsupported_coverage",
    }:
        raise ValueError(f"Unknown checker status: {status}")
    if coverage not in {"observed", "disabled", "unsupported", "error"}:
        raise ValueError(f"Unknown coverage status: {coverage}")
    passed = nullable_bool(required(rule, "check_passed"), "check_passed")
    triggered = nullable_bool(required(rule, "checker_triggered"), "checker_triggered")
    active = nullable_bool(required(rule, "active"), "active")
    dormant = nullable_bool(required(rule, "dormant"), "dormant")
    observed = boolean(required(rule, "coverage_observed"), "coverage_observed")
    if status == "completed":
        checker = mapping(required(rule, "checker_result"), "checker_result")
        mapping(required(checker, "invariant"), "checker_result.invariant")
        if passed is None or triggered is None:
            raise ValueError("Completed checking requires a known outcome")
        if (
            boolean(required(checker, "check_passed"), "checker_result.check_passed")
            != passed
            or boolean(required(checker, "triggered"), "checker_result.triggered")
            != triggered
        ):
            raise ValueError("Checker result and coverage outcome disagree")
    elif passed is not None or triggered is not None:
        raise ValueError("Incomplete checking must retain unknown outcomes")
    if status in {"checker_error", "observer_error"}:
        error = mapping(required(rule, "error"), "error")
        string(required(error, "type"), "error.type")
        string(required(error, "message"), "error.message")
    decisions = None
    skipped = None
    if coverage == "observed":
        if not observed or status not in {"completed", "checker_error"}:
            raise ValueError("Observed coverage has an incompatible checker status")
        decisions = count(required(rule, "decisions_completed"), "decisions_completed")
        if active is not (decisions > 0) or dormant is not (
            status == "completed" and not active
        ):
            raise ValueError("Activation disagrees with completed relation decisions")
        skipped = boolean(
            required(rule, "variable_check_skipped"), "variable_check_skipped"
        )
        fully_validated = boolean(
            required(rule, "fully_validated_active"), "fully_validated_active"
        )
        if fully_validated != (active and not skipped and status == "completed"):
            raise ValueError("Fully validated status disagrees with activity or skips")
    elif observed or active is not None or dormant is not None:
        raise ValueError("Unobserved activation must remain unknown")
    if status == "unsupported_coverage" and coverage != "unsupported":
        raise ValueError("Unsupported coverage must be explicitly marked")
    if status == "observer_error" and coverage != "error":
        raise ValueError("Observer errors must have error coverage")
    description = required(rule, "description")
    if description is not None:
        description = string(description, "description")
    identity = sequence(required(rule, "identity_clauses"), "identity_clauses")
    for clause in identity:
        mapping(clause, "identity clause")
    return RuleView(
        index,
        string(required(rule, "relation"), "relation"),
        description,
        status,
        coverage,
        passed,
        active,
        dormant,
        skipped,
        decisions,
        bool(identity),
    )


def summarize(coverage_value: object) -> JSONObject:
    coverage = mapping(coverage_value, "coverage")
    if integer(required(coverage, "schema_version"), "schema_version") != 3:
        raise ValueError("Expected coverage schema_version 3")
    if not boolean(required(coverage, "inputs_unchanged"), "inputs_unchanged"):
        raise ValueError("Coverage inputs changed during checking")
    rows = count(required(coverage, "trace_rows"), "trace_rows")
    hashes = mapping(required(coverage, "input_sha256"), "input_sha256")
    if not hashes:
        raise ValueError("Coverage requires input hashes")
    for path, digest in hashes.items():
        text = string(digest, f"input_sha256[{path}]")
        if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError("Malformed input SHA256")
    rules = [
        read_rule(raw, i)
        for i, raw in enumerate(
            sequence(required(coverage, "per_invariant"), "per_invariant")
        )
    ]
    counts: Counter[str] = Counter()
    problems: list[JSONObject] = []
    for rule in rules:
        counts[rule.status] += 1
        counts["observed_active"] += rule.active is True
        counts["observed_dormant"] += rule.dormant is True
        counts["observed_violations"] += rule.passed is False
        counts["observed_active_passes"] += rule.active is True and rule.passed is True
        counts["unknown_outcomes"] += rule.passed is None
        counts["unobserved_coverage"] += rule.coverage_status != "observed"
        counts["skipped_variable_checks"] += rule.skipped is True
        counts["completed_decisions"] += 0 if rule.decisions is None else rule.decisions
        if rule.passed is not True or rule.active is not True or rule.skipped:
            problems.append(
                json_object(
                    {
                        "invariant_index": rule.index,
                        "relation": rule.relation,
                        "description": rule.description,
                        "status": rule.status,
                        "coverage_status": rule.coverage_status,
                        "check_passed": rule.passed,
                        "active": rule.active,
                        "dormant": rule.dormant,
                        "variable_check_skipped": rule.skipped,
                        "identity_bearing": rule.identity_bearing,
                        "evidence_pointer": f"per_invariant[{rule.index}]",
                    }
                )
            )
    source_summary = mapping(required(coverage, "summary"), "summary")
    for source_key, derived_key in (
        ("observed_active_invariants", "observed_active"),
        ("observed_dormant_invariants", "observed_dormant"),
        ("observed_violated_invariants", "observed_violations"),
        ("check_outcome_unknown_invariants", "unknown_outcomes"),
        ("coverage_unobserved_invariants", "unobserved_coverage"),
    ):
        if (
            count(required(source_summary, source_key), source_key)
            != counts[derived_key]
        ):
            raise ValueError(
                f"Coverage summary disagrees with its rule records: {source_key}"
            )
    incomplete = any(
        counts[key]
        for key in (
            "checker_error",
            "observer_error",
            "unknown_outcomes",
            "unobserved_coverage",
            "skipped_variable_checks",
            "observed_dormant",
        )
    )
    if counts["observed_violations"]:
        outcome = "violations"
    elif not rules or rows == 0 or counts["completed_decisions"] == 0:
        outcome = "no_evidence"
    elif incomplete:
        outcome = "incomplete"
    else:
        outcome = "active_checks_passed"
    return json_object(
        {
            "schema_version": 1,
            "outcome": outcome,
            "coverage_incomplete": incomplete,
            "trace_rows": rows,
            "invariants": len(rules),
            "counts": {
                key: counts[key]
                for key in (
                    "completed",
                    "checker_error",
                    "observer_error",
                    "unsupported_coverage",
                    "observed_active",
                    "observed_active_passes",
                    "observed_dormant",
                    "observed_violations",
                    "unknown_outcomes",
                    "unobserved_coverage",
                    "skipped_variable_checks",
                    "completed_decisions",
                )
            },
            "problem_rules": problems[:10],
            "additional_problem_rules": max(0, len(problems) - 10),
            "input_sha256": hashes,
            "interpretation": "These are observed rule outcomes, not task success or proof that training is correct.",
        }
    )


def read_artifact(path: Path) -> tuple[JSONObject, JSONObject]:
    contents = path.read_bytes()
    value: object = json.loads(contents)
    return json_object(value, str(path)), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--collection-policy", type=Path)
    parser.add_argument("--violations-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Options())
    coverage, source = read_artifact(args.coverage)
    report = summarize(coverage)
    report["coverage_source"] = source
    report["collection_mode"] = "unknown: no collection policy supplied"
    if args.collection_policy is not None:
        policy, policy_source = read_artifact(args.collection_policy)
        raw_style = required(policy, "model_tracker_style")
        style = None if raw_style is None else string(raw_style, "model_tracker_style")
        if style is not None and style not in {"proxy", "subclass", "sampler"}:
            raise ValueError("Unsupported model_tracker_style in collection policy")
        report["collection_mode"] = (
            "api-only" if style is None else "model-tracker-requested"
        )
        report["configured_model_tracker"] = style
        report["collection_policy_source"] = policy_source
    if args.violations_summary is not None:
        direct, direct_source = read_artifact(args.violations_summary)
        total = count(
            required(direct, "distinct_invariants_violated"),
            "distinct_invariants_violated",
        )
        if total != len(sequence(required(direct, "violations"), "violations")):
            raise ValueError(
                "Direct checker summary has an inconsistent violation count"
            )
        report_counts = mapping(report["counts"], "counts")
        if total != report_counts["observed_violations"]:
            raise ValueError("Direct and observed checker violation counts disagree")
        report["direct_summary_source"] = direct_source
        report["direct_count_check"] = (
            "matched; complete result equivalence not established"
        )
    # Never overwrite an earlier report or either input artifact during a retry.
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outcome": report["outcome"], "counts": report["counts"]}))
    return 0


class Options(argparse.Namespace):
    coverage: Path
    collection_policy: Path | None
    violations_summary: Path | None
    output: Path


if __name__ == "__main__":
    raise SystemExit(main())
