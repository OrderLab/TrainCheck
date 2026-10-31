"""Observe the real offline checker; never replace its predicate or learned rules.

Counters have explicit units. API candidates are calls, DistinctArgument candidates
are within-step argument pairs, and Consistency candidates are overlapping variable
value pairs collected by the real checker. Precondition counters count verification
calls. Each checker stops at its first violation. No wall-time claims are made.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from types import FrameType
from typing import Literal, TypeAlias, TypedDict
from unittest.mock import patch

import numpy as np

from traincheck.invariant import read_inv_file
from traincheck.invariant.base_cls import (
    APIParam,
    GroupedPreconditions,
    Invariant,
    VarTypeParam,
)
from traincheck.invariant.DistinctArgumentRelation import (
    get_event_data_per_function_per_step,
)
from traincheck.trace.trace import Trace
from traincheck.trace.trace_pandas import TracePandas, read_trace_file_Pandas
from traincheck.trace.types import (
    AttrState,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
    MDNONEJSONEncoder,
    VarInstId,
)

# Direct CLI execution must resolve this checkout's harness package even when
# PYTHONPATH intentionally points to a separately composed TrainCheck source.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.fresh_validation.contracts import (  # noqa: E402
    JSONObject,
    JSONValue,
    boolean,
    integer,
    json_list,
    json_object,
    json_value,
    mapping,
    number,
    required,
    sequence,
    string,
)


class ObserverContractError(ValueError):
    """Checker internals no longer satisfy the observer's verified contract."""


class ObserverSites(TypedDict):
    starts: dict[int, str]
    ends: dict[int, int]
    visits: set[int]
    pair_collection: int | None
    variable_requirements: int | None
    method_source_sha256: str


class CallDecision(TypedDict):
    func_call_id: str


class PairDecision(TypedDict):
    pair_func_call_ids: list[str]


class VariableValueIdentity(TypedDict):
    process_id: int | float | str
    thread_id: int | float | str
    var_name: str
    var_type: str
    attribute: str
    event_time: int | float
    liveness_start: int | float
    liveness_end: int | float


class VariablePairDecision(TypedDict):
    variable_value_pair: list[VariableValueIdentity]


class VariableRequirement(TypedDict):
    var_type: str
    attribute: str
    matching_instances: int
    instances_missing_attribute: int


Decision: TypeAlias = CallDecision | PairDecision | VariablePairDecision


class PendingDecision(TypedDict):
    start: int
    end: int
    record: Decision


class GroupCounts(TypedDict):
    attempts: int
    passes: int
    failures: int
    errors: int


class CandidateCounts(TypedDict):
    candidate_calls: int | None
    opportunities: int | None
    opportunity_unit: str
    incomplete_or_exception_calls: int | None


class OptionalReportFields(TypedDict, total=False):
    invariant_index: int
    checker_result: JSONObject
    error: dict[str, str]
    checker_error_after_observer: dict[str, str]
    candidate_calls: int | None
    opportunities: int | None
    opportunity_unit: str
    incomplete_or_exception_calls: int | None
    method_source_sha256: str
    decisions_started: int
    decisions_completed: int
    decisions_not_completed: int
    visited_calls: int | None
    precondition_group_attempts: int
    precondition_group_passes: int
    precondition_group_failures: int
    precondition_group_errors: int
    precondition_groups: dict[str, GroupCounts]
    decision_records: list[Decision]
    unvisited_calls_after_stop: int | None
    unvisited_pairs_after_stop: int | None
    visited_variable_pairs: int
    variable_requirements: list[VariableRequirement]
    variable_precondition_requested: bool
    variable_check_skipped: bool
    fully_validated_active: bool


class ObservationReport(OptionalReportFields):
    # Optional detail fields are present only for the matching status. Core
    # outcome/observation fields are always present; unknown is never false.
    relation: str
    description: str | None
    identity_clauses: list[JSONObject]
    observer_supported: bool
    status: Literal[
        "completed", "checker_error", "observer_error", "unsupported_coverage"
    ]
    coverage_status: Literal["observed", "disabled", "unsupported", "error"]
    coverage_observed: bool
    active: bool | None
    dormant: bool | None
    check_passed: bool | None
    checker_triggered: bool | None


class RuntimeMetadata(TypedDict):
    python: str
    executable: str
    trace_module: str
    packages: dict[str, str]


class CoverageOutput(TypedDict):
    # Version 3 adds variable-pair decisions and nullable call/opportunity
    # counters for variable-only checking and interrupted pair collection.
    schema_version: Literal[3]
    pid: int
    check_relation_first: Literal[False]
    observer_enabled: bool
    runtime: RuntimeMetadata
    harness_sha256: str
    raw_process_ids: list[JSONValue]
    raw_thread_ids: list[JSONValue]
    trace_rows: int
    input_sha256: dict[str, str]
    inputs_unchanged: bool
    summary: dict[str, int | None]
    per_invariant: list[ObservationReport]


TraceCallback: TypeAlias = Callable[[FrameType, str, object], "TraceCallback | None"]


def identity_scalar(value: object, context: str) -> int | float | str:
    """Preserve legacy dataframe identity scalars without numeric coercion."""
    if isinstance(value, str):
        return value
    result = number(value, context)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def dataframe_timestamp(value: object, context: str) -> int | float:
    """Convert observed dataframe time scalars to exact, finite JSON numbers.

    Pandas reductions can leave NumPy scalars in AttrState liveness bounds even
    when source-event dictionaries contain builtin numbers. Keep integer
    nanoseconds as integers, including beyond float64's exact-integer range.
    Extended NumPy floating values are accepted only if conversion to Python's
    float preserves their value. This boundary does not relax JSON validators.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a timestamp number, not a boolean")
    if isinstance(value, np.integer):
        # NumPy also classifies timedelta64 as an integer. A duration's unit
        # cannot be inferred as this trace's numeric nanosecond timestamp.
        if isinstance(value, np.timedelta64):
            raise ValueError(f"{context} must be a timestamp number, not a duration")
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted) or value != converted:
            raise ValueError(
                f"{context} must be finite and exactly representable as a Python float"
            )
        return converted
    result = number(value, context)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def variable_value_identity(state: object, param: object) -> VariableValueIdentity:
    if not isinstance(state, AttrState) or not isinstance(param, VarTypeParam):
        raise ValueError("Consistency decisions require AttrState and VarTypeParam")
    if not state.traces:
        raise ValueError("Consistency value state has no source events")
    record = mapping(state.traces[-1], "variable event")
    return {
        "process_id": identity_scalar(required(record, "process_id"), "process_id"),
        "thread_id": identity_scalar(required(record, "thread_id"), "thread_id"),
        "var_name": string(required(record, "var_name"), "var_name"),
        "var_type": string(required(record, "var_type"), "var_type"),
        "attribute": string(param.attr_name, "attribute"),
        "event_time": dataframe_timestamp(required(record, "time"), "time"),
        "liveness_start": dataframe_timestamp(
            state.liveness.start_time, "liveness.start_time"
        ),
        "liveness_end": dataframe_timestamp(
            state.liveness.end_time, "liveness.end_time"
        ),
    }


def variable_requirements(
    local_values: Mapping[str, object], inv: Invariant
) -> list[VariableRequirement]:
    """Describe missing input from the same index the real checker just read."""
    instances = required(local_values, "all_var_insts")
    if not isinstance(instances, Mapping):
        raise ObserverContractError("all_var_insts must be a mapping")
    requirements: list[VariableRequirement] = []
    for param in inv.params:
        if not isinstance(param, VarTypeParam):
            raise ObserverContractError("Consistency requires VarTypeParam")
        matched = missing = 0
        for identity, raw_attributes in instances.items():
            if not isinstance(identity, VarInstId):
                raise ObserverContractError("Variable index requires VarInstId keys")
            attributes = mapping(raw_attributes, "variable attributes")
            if identity.var_type == param.var_type:
                matched += 1
                missing += param.attr_name not in attributes
        requirements.append(
            {
                "var_type": param.var_type,
                "attribute": param.attr_name,
                "matching_instances": matched,
                "instances_missing_attribute": missing,
            }
        )
    return requirements


def decision_record(family: str, local_values: Mapping[str, object]) -> Decision:
    """Read the exact family-specific identity fields, failing closed on drift."""
    try:
        if family == "ConsistencyRelation":
            return {
                "variable_value_pair": [
                    variable_value_identity(
                        required(local_values, f"attr{index}_val"),
                        required(local_values, f"param{index}"),
                    )
                    for index in (1, 2)
                ]
            }
        if family == "DistinctArgumentRelation":
            ids = []
            for name in ("event1", "event2"):
                event = mapping(required(local_values, name), name)
                ids.append(
                    string(required(event, "func_call_id"), f"{name}.func_call_id")
                )
            return {"pair_func_call_ids": ids}
        name = (
            "parent_func_call_id" if family == "APIContainRelation" else "func_call_id"
        )
        if family not in SUPPORTED:
            raise ValueError(f"Unsupported observer family: {family}")
        return {"func_call_id": string(required(local_values, name), name)}
    except ValueError as exc:
        raise ObserverContractError(str(exc)) from exc


def finalize_report(report: ObservationReport) -> ObservationReport:
    """Validate the conditional schema before exposing a mutable report.

    OptionalReportFields is an internal construction accumulator, not permission
    to omit completed-result or observed-coverage fields in emitted reports.
    """
    status = report["status"]
    coverage_status = report["coverage_status"]
    if status not in {
        "completed",
        "checker_error",
        "observer_error",
        "unsupported_coverage",
    }:
        raise ObserverContractError(f"Unknown checker status: {status}")
    if coverage_status not in {"observed", "disabled", "unsupported", "error"}:
        raise ObserverContractError(f"Unknown coverage status: {coverage_status}")
    try:
        if status == "completed":
            checker = mapping(required(report, "checker_result"), "checker_result")
            passed = boolean(report["check_passed"], "check_passed")
            triggered = boolean(report["checker_triggered"], "checker_triggered")
            if (
                boolean(
                    required(checker, "check_passed"), "checker_result.check_passed"
                )
                != passed
                or boolean(required(checker, "triggered"), "checker_result.triggered")
                != triggered
            ):
                raise ValueError("Checker outcome and report outcome disagree")
            mapping(required(checker, "invariant"), "checker_result.invariant")
        else:
            if (
                report["check_passed"] is not None
                or report["checker_triggered"] is not None
            ):
                raise ValueError("Incomplete checking must have unknown outcome")
            if status in {"checker_error", "observer_error"}:
                error = mapping(required(report, "error"), "error")
                string(required(error, "type"), "error.type")
                string(required(error, "message"), "error.message")
        if coverage_status == "observed":
            if report["coverage_observed"] is not True or status not in {
                "completed",
                "checker_error",
            }:
                raise ValueError("Observed coverage has incompatible checker status")
            boolean(report["active"], "active")
            boolean(report["dormant"], "dormant")
            for key in (
                "decisions_started",
                "decisions_completed",
                "decisions_not_completed",
                "precondition_group_attempts",
                "precondition_group_passes",
                "precondition_group_failures",
                "precondition_group_errors",
            ):
                if integer(required(report, key), key) < 0:
                    raise ValueError(f"{key} cannot be negative")
            opportunities = required(report, "opportunities")
            if opportunities is None:
                if (
                    report["relation"] != "ConsistencyRelation"
                    or status != "checker_error"
                ):
                    raise ValueError(
                        "Only failed variable pair collection has unknown opportunities"
                    )
            elif integer(opportunities, "opportunities") < 0:
                raise ValueError("opportunities cannot be negative")
            candidate_calls = required(report, "candidate_calls")
            if report["relation"] == "ConsistencyRelation":
                if candidate_calls is not None:
                    raise ValueError("Variable coverage must not claim API candidates")
                if (
                    integer(
                        required(report, "visited_variable_pairs"),
                        "visited_variable_pairs",
                    )
                    < 0
                ):
                    raise ValueError("visited_variable_pairs cannot be negative")
                for raw_requirement in sequence(
                    required(report, "variable_requirements"), "variable_requirements"
                ):
                    requirement = mapping(raw_requirement, "variable requirement")
                    for key in ("var_type", "attribute"):
                        string(required(requirement, key), key)
                    for key in ("matching_instances", "instances_missing_attribute"):
                        if integer(required(requirement, key), key) < 0:
                            raise ValueError(f"{key} cannot be negative")
            elif integer(candidate_calls, "candidate_calls") < 0:
                raise ValueError("candidate_calls cannot be negative")
            for key in (
                "variable_precondition_requested",
                "variable_check_skipped",
                "fully_validated_active",
            ):
                boolean(required(report, key), key)
            for key in ("opportunity_unit", "method_source_sha256"):
                string(required(report, key), key)
            groups = mapping(
                required(report, "precondition_groups"), "precondition_groups"
            )
            for name, raw_group in groups.items():
                group = mapping(raw_group, name)
                for key in ("attempts", "passes", "failures", "errors"):
                    if integer(required(group, key), f"{name}.{key}") < 0:
                        raise ValueError(f"{name}.{key} cannot be negative")
            if report["relation"] in {
                "DistinctArgumentRelation",
                "ConsistencyRelation",
            }:
                if (
                    required(report, "visited_calls") is not None
                    or required(report, "unvisited_calls_after_stop") is not None
                ):
                    raise ValueError("Pair coverage must not claim call visit counts")
                unvisited_pairs = required(report, "unvisited_pairs_after_stop")
                if opportunities is None:
                    if unvisited_pairs is not None:
                        raise ValueError(
                            "Unknown opportunities require unknown remainder"
                        )
                else:
                    integer(unvisited_pairs, "unvisited_pairs_after_stop")
            else:
                integer(required(report, "visited_calls"), "visited_calls")
                integer(
                    required(report, "unvisited_calls_after_stop"),
                    "unvisited_calls_after_stop",
                )
            for raw in sequence(
                required(report, "decision_records"), "decision_records"
            ):
                decision = mapping(raw, "decision")
                if report["relation"] == "ConsistencyRelation":
                    pair = sequence(
                        required(decision, "variable_value_pair"), "variable_value_pair"
                    )
                    if len(pair) != 2:
                        raise ValueError(
                            "Variable decisions require exactly two values"
                        )
                    for raw_identity in pair:
                        identity = mapping(raw_identity, "variable value identity")
                        for key in ("process_id", "thread_id"):
                            identity_scalar(required(identity, key), key)
                        for key in ("var_name", "var_type", "attribute"):
                            string(required(identity, key), key)
                        for key in ("event_time", "liveness_start", "liveness_end"):
                            number(required(identity, key), key)
                elif report["relation"] == "DistinctArgumentRelation":
                    pair = sequence(
                        required(decision, "pair_func_call_ids"), "pair_func_call_ids"
                    )
                    if len(pair) != 2:
                        raise ValueError("Pair decisions require exactly two call IDs")
                    for call_identity in pair:
                        string(call_identity, "pair call ID")
                else:
                    string(required(decision, "func_call_id"), "decision.func_call_id")
        elif (
            report["coverage_observed"] is not False
            or report["active"] is not None
            or report["dormant"] is not None
        ):
            raise ValueError(
                "Unobserved coverage must have unknown active/dormant states"
            )
        if status == "unsupported_coverage" and coverage_status != "unsupported":
            raise ValueError("Unsupported checker coverage must be marked unsupported")
        if status == "observer_error" and coverage_status != "error":
            raise ValueError("Observer failure must mark coverage as an error")
    except ValueError as exc:
        raise ObserverContractError(str(exc)) from exc
    return report


def summarize(results: list[ObservationReport]) -> dict[str, int | None]:
    counts: Counter[str] = Counter()
    unobserved = 0
    unknown_outcomes = 0
    for result in results:
        finalize_report(result)
        counts[result["status"]] += 1
        observed = result["coverage_status"] == "observed"
        if observed:
            active, dormant = result["active"], result["dormant"]
            assert active is not None and dormant is not None
            counts["active_invariants"] += active
            counts["dormant_invariants"] += dormant
            counts["identity_bearing_active"] += (
                bool(result["identity_clauses"]) and active
            )
        else:
            unobserved += 1
        unknown_outcomes += result["check_passed"] is None
        counts["violated_invariants"] += result["check_passed"] is False
        counts["identity_bearing_invariants"] += bool(result["identity_clauses"])
    summary: dict[str, int | None] = dict(counts)
    summary["coverage_unobserved_invariants"] = unobserved
    summary["check_outcome_unknown_invariants"] = unknown_outcomes
    summary["observed_violated_invariants"] = counts["violated_invariants"]
    summary["violated_invariants"] = (
        None if unknown_outcomes else counts["violated_invariants"]
    )
    # A disabled/unsupported/error observer cannot establish the total active or
    # dormant population. Keep the observed subset explicitly separate.
    for name in ("active_invariants", "dormant_invariants", "identity_bearing_active"):
        summary["observed_" + name] = counts[name]
        summary[name] = None if unobserved else counts[name]
    return summary


SUPPORTED = {
    "APIContainRelation",
    "ConsistentOutputRelation",
    "ConsistentInputOutputRelation",
    "ThresholdRelation",
    "DistinctArgumentRelation",
    "ConsistencyRelation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observer_sites(function: Callable[..., object], family: str) -> ObserverSites:
    """Locate exact predicate ASTs in the installed method, failing closed on drift."""
    lines, first = inspect.getsourcelines(function)
    # Dedent code by its method indentation. A zero-indented line inside a
    # multiline diagnostic string defeats textwrap.dedent's common prefix.
    indent = len(lines[0]) - len(lines[0].lstrip())
    source = "".join(
        line[indent:] if line.startswith(" " * indent) else line for line in lines
    )
    tree = ast.parse(source)
    starts: dict[int, str] = {}
    ends: dict[int, int] = {}
    visits: set[int] = set()
    pair_collection: int | None = None
    requirements: int | None = None
    predicates: dict[str, Callable[[str], bool]] = {
        "APIContainRelation": lambda s: "skip_var_unchanged_check" in s
        and "found_expected_child_event" in s,
        "ConsistentInputOutputRelation": lambda s: s == "input_value != output_value",
        "ConsistentOutputRelation": lambda s: s == "len(returned_tensors) == 0",
        "ThresholdRelation": lambda s: s
        in ("output_value >= threshold_value", "output_value <= threshold_value"),
        "DistinctArgumentRelation": lambda s: s
        == "is_arguments_list_same(event1['args'], event2['args'])",
    }
    for node in ast.walk(tree):
        if (
            family != "ConsistencyRelation"
            and isinstance(node, ast.If)
            and predicates[family](ast.unparse(node.test))
        ):
            line = first + node.lineno - 1
            starts[line] = ast.unparse(node.test)
            if node.test.end_lineno is None:
                raise ObserverContractError("Predicate has no ending source line")
            ends[line] = first + node.test.end_lineno - 1
        if family == "ConsistencyRelation" and isinstance(node, ast.Assign):
            value_source = ast.unparse(node.value)
            line = first + node.lineno - 1
            if (
                value_source
                == "ConsistencyRelation.evaluate([attr1_val.value, attr2_val.value])"
            ):
                if node.end_lineno is None:
                    raise ObserverContractError(
                        "Consistency comparison has no end line"
                    )
                starts[line] = value_source
                ends[line] = first + node.end_lineno - 1
            elif value_source == "dict(sorted(value_pairs_to_check.items()))":
                if pair_collection is not None:
                    raise ObserverContractError(
                        "Multiple variable pair collection sites"
                    )
                pair_collection = line
            elif value_source == "[attr1_val.traces[-1], attr2_val.traces[-1]]":
                visits.add(line)
        if family == "ConsistencyRelation" and isinstance(node, ast.Try):
            # The first try immediately follows get_var_insts(); observe its
            # required type/attribute availability before the checker catches a
            # missing attribute and returns a legacy vacuous success.
            if any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "type1_attr1"
                    for target in child.targets
                )
                for child in node.body
            ):
                if requirements is not None:
                    raise ObserverContractError("Multiple variable requirement sites")
                requirements = first + node.lineno - 1
        if (
            family != "ConsistencyRelation"
            and isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id in {"func_call_event", "parent_event_type"}
                for t in node.targets
            )
        ):
            visits.add(first + node.lineno - 1)
    expected = (
        2
        if family
        in {"ThresholdRelation", "DistinctArgumentRelation", "ConsistencyRelation"}
        else 1
    )
    if len(starts) != expected:
        raise ObserverContractError(
            f"Unrecognized checker source for {family}: {starts}"
        )
    if family == "ConsistencyRelation" and (
        pair_collection is None or requirements is None or len(visits) != 1
    ):
        raise ObserverContractError("Unrecognized variable pair collection source")
    return {
        "starts": starts,
        "ends": ends,
        "visits": visits,
        "pair_collection": pair_collection,
        "variable_requirements": requirements,
        "method_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def candidate_counts(trace: TracePandas, inv: Invariant) -> CandidateCounts:
    family = inv.relation.__name__
    if family == "ConsistencyRelation":
        # Filled from the real checker's completed collection phase. Do not
        # duplicate its liveness filtering or pre-scan and warm the trace cache.
        return {
            "candidate_calls": None,
            "opportunities": None,
            "opportunity_unit": "overlapping_variable_value_pair",
            "incomplete_or_exception_calls": None,
        }
    api = (
        inv.params[1]
        if family in {"ConsistentInputOutputRelation", "ThresholdRelation"}
        else inv.params[0]
    )
    if not isinstance(api, APIParam):
        raise ObserverContractError(f"{family} expected an APIParam")
    calls = trace.get_func_call_ids(api.api_full_name)
    if family == "DistinctArgumentRelation":
        _, groups = get_event_data_per_function_per_step(trace, {api.api_full_name})
        pairs = sum(
            math.comb(sum(map(len, records.values())), 2)
            for records in groups.get(api.api_full_name, {}).values()
        )
        return {
            "candidate_calls": len(calls),
            "opportunities": pairs,
            "opportunity_unit": "within_step_argument_pair",
            "incomplete_or_exception_calls": None,
        }
    if family == "APIContainRelation":
        eligible = len(calls)
    else:
        eligible = sum(
            not isinstance(
                trace.query_func_call_event(call),
                (FuncCallExceptionEvent, IncompleteFuncCallEvent),
            )
            for call in calls
        )
    return {
        "candidate_calls": len(calls),
        "opportunities": eligible,
        "opportunity_unit": (
            "parent_call"
            if family == "APIContainRelation"
            else "complete_function_call"
        ),
        "incomplete_or_exception_calls": len(calls) - eligible,
    }


def id_clauses(value: JSONValue) -> list[JSONObject]:
    found: list[JSONObject] = []
    if isinstance(value, dict):
        if isinstance(value.get("prop_name"), str) and value.get("prop_name") in {
            "process_id",
            "thread_id",
            "meta_vars.process_id",
            "meta_vars.thread_id",
        }:
            found.append(value)
        for child in value.values():
            found.extend(id_clauses(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(id_clauses(child))
    return found


def observe(
    trace: TracePandas, inv: Invariant, enabled: bool = True
) -> ObservationReport:
    """Run one unchanged inv.check(False), counting only observed completed decisions."""
    family = inv.relation.__name__
    assert inv.precondition is not None, "Invariant precondition must be set"
    report: ObservationReport = {
        "relation": family,
        "description": inv.text_description,
        "identity_clauses": id_clauses(
            json_value(
                json.loads(
                    json.dumps(inv.precondition.to_dict(), cls=MDNONEJSONEncoder)
                )
            )
        ),
        "observer_supported": family in SUPPORTED,
        "status": "unsupported_coverage",
        "coverage_status": "disabled" if not enabled else "unsupported",
        "coverage_observed": False,
        "active": None,
        "dormant": None,
        "check_passed": None,
        "checker_triggered": None,
    }
    if not enabled:
        # Direct oracle: no opportunity pre-scan, source parsing, or monkeypatches.
        try:
            result = inv.check(trace, check_relation_first=False)
            result.calc_and_set_time_precentage(
                trace.get_start_time(), trace.get_end_time()
            )
            report.update(
                {
                    "status": "completed",
                    "checker_result": json_object(
                        json.loads(json.dumps(result.to_dict(), cls=MDNONEJSONEncoder))
                    ),
                    "check_passed": boolean(
                        result.check_passed, "checker.check_passed"
                    ),
                    "checker_triggered": boolean(result.triggered, "checker.triggered"),
                }
            )
        except Exception as exc:
            report.update(
                {
                    "status": "checker_error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "check_passed": None,
                }
            )
        report.update({"coverage_observed": False, "active": None, "dormant": None})
        return finalize_report(report)
    if family not in SUPPORTED:
        return finalize_report(report)
    candidates = candidate_counts(trace, inv)
    report.update(candidates)
    sites = observer_sites(inv.relation.static_check_all, family)
    report["method_source_sha256"] = sites["method_source_sha256"]
    counts: Counter[str] = Counter(
        decisions_started=0,
        decisions_completed=0,
        visited_calls=0,
        visited_variable_pairs=0,
        precondition_group_attempts=0,
        precondition_group_passes=0,
        precondition_group_failures=0,
        precondition_group_errors=0,
    )
    groups: dict[str, GroupCounts] = {}
    decision_records: list[Decision] = []
    requirements: list[VariableRequirement] = []
    variable_precondition_requested = family == "ConsistencyRelation" or (
        family == "APIContainRelation"
        and "var_events" in inv.precondition.get_group_names()
        and len(inv.precondition.get_group("var_events")) > 0
        and not inv.precondition.is_group_unconditional("var_events")
    )
    variable_check_skipped = False
    method_code = inv.relation.static_check_all.__code__
    pending: PendingDecision | None = None
    observer_failure: ObserverContractError | None = None
    original_verify = GroupedPreconditions.verify

    def verify(
        precondition: GroupedPreconditions,
        records: list[object],
        group_name: str,
        trace_for_metavars: Trace | None = None,
    ) -> bool:
        group = groups.setdefault(
            group_name, {"attempts": 0, "passes": 0, "failures": 0, "errors": 0}
        )
        counts["precondition_group_attempts"] += 1
        group["attempts"] += 1
        try:
            result = original_verify(
                precondition, records, group_name, trace_for_metavars
            )
        except Exception:
            counts["precondition_group_errors"] += 1
            group["errors"] += 1
            raise
        label: Literal["passes", "failures"] = "passes" if result else "failures"
        group[label] += 1
        counts["precondition_group_" + label] += 1
        return result

    def trace_event_impl(
        frame: FrameType, event: str, arg: object
    ) -> TraceCallback | None:
        nonlocal pending, variable_check_skipped, requirements
        if frame.f_code is not method_code:
            return None
        if event == "exception":
            # An extraction error can be caught by the real method; it is not a decision.
            pending = None
        if pending is not None:
            done = event == "return"
            if event == "line":
                if family == "ConsistentOutputRelation":
                    done = frame.f_lineno in sites["visits"]
                else:
                    done = not pending["start"] <= frame.f_lineno <= pending["end"]
            if done:
                counts["decisions_completed"] += 1
                decision_records.append(pending["record"])
                pending = None
        if event == "line":
            if frame.f_lineno == sites["variable_requirements"]:
                try:
                    requirements = variable_requirements(frame.f_locals, inv)
                except ValueError as exc:
                    raise ObserverContractError(str(exc)) from exc
                variable_check_skipped = any(
                    item["matching_instances"] == 0
                    or item["instances_missing_attribute"] > 0
                    for item in requirements
                )
            if frame.f_lineno == sites["pair_collection"]:
                try:
                    pairs = integer(
                        required(frame.f_locals, "num_collected_pairs"),
                        "num_collected_pairs",
                    )
                    if pairs < 0:
                        raise ValueError("num_collected_pairs cannot be negative")
                except ValueError as exc:
                    raise ObserverContractError(str(exc)) from exc
                candidates["opportunities"] = pairs
                report["opportunities"] = pairs
            if frame.f_lineno in sites["visits"]:
                counts[
                    (
                        "visited_variable_pairs"
                        if family == "ConsistencyRelation"
                        else "visited_calls"
                    )
                ] += 1
            if frame.f_lineno in sites["starts"] and pending is None:
                if family == "APIContainRelation" and variable_precondition_requested:
                    try:
                        skipped = boolean(
                            required(frame.f_locals, "skip_var_unchanged_check"),
                            "skip_var_unchanged_check",
                        )
                    except ValueError as exc:
                        raise ObserverContractError(str(exc)) from exc
                    if skipped:
                        variable_check_skipped = True
                record = decision_record(family, frame.f_locals)
                counts["decisions_started"] += 1
                pending = {
                    "start": frame.f_lineno,
                    "end": sites["ends"][frame.f_lineno],
                    "record": record,
                }
        return trace_event

    def trace_event(frame: FrameType, event: str, arg: object) -> TraceCallback | None:
        nonlocal observer_failure
        try:
            return trace_event_impl(frame, event, arg)
        except ObserverContractError as exc:
            # A checker may catch an exception raised by sys.settrace. Retain it
            # independently so source drift cannot become a completed check.
            observer_failure = exc
            raise

    previous_trace = sys.gettrace()
    if enabled and previous_trace is not None:
        raise RuntimeError("A pre-existing Python trace observer would be overwritten")
    try:
        with patch.object(GroupedPreconditions, "verify", verify):
            sys.settrace(trace_event)
            result = inv.check(trace, check_relation_first=False)
            if observer_failure is not None:
                raise observer_failure
            result.calc_and_set_time_precentage(
                trace.get_start_time(), trace.get_end_time()
            )
            report.update(
                {
                    "status": "completed",
                    "checker_result": json_object(
                        json.loads(json.dumps(result.to_dict(), cls=MDNONEJSONEncoder))
                    ),
                    "check_passed": boolean(
                        result.check_passed, "checker.check_passed"
                    ),
                    "checker_triggered": boolean(result.triggered, "checker.triggered"),
                }
            )
    except Exception as exc:
        failure = observer_failure if observer_failure is not None else exc
        report.update(
            {
                "status": (
                    "observer_error"
                    if isinstance(failure, ObserverContractError)
                    else "checker_error"
                ),
                "error": {"type": type(failure).__name__, "message": str(failure)},
                "check_passed": None,
            }
        )
        if observer_failure is not None and exc is not observer_failure:
            report["checker_error_after_observer"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
    finally:
        if enabled:
            sys.settrace(previous_trace)
    if family == "ConsistencyRelation" and report["status"] == "completed":
        if candidates["opportunities"] is None:
            if not variable_check_skipped:
                raise ObserverContractError(
                    "Completed variable checker did not collect pairs"
                )
            # The checker returned before collection because an attribute is
            # absent; this is explicitly skipped, never validated as active.
            candidates["opportunities"] = 0
            report["opportunities"] = 0
    report["decisions_started"] = counts["decisions_started"]
    report["decisions_completed"] = counts["decisions_completed"]
    report["visited_calls"] = counts["visited_calls"]
    report["precondition_group_attempts"] = counts["precondition_group_attempts"]
    report["precondition_group_passes"] = counts["precondition_group_passes"]
    report["precondition_group_failures"] = counts["precondition_group_failures"]
    report["precondition_group_errors"] = counts["precondition_group_errors"]
    report["precondition_groups"] = groups
    report["decision_records"] = decision_records
    report["decisions_not_completed"] = (
        counts["decisions_started"] - counts["decisions_completed"]
    )
    stopped = report["status"] != "completed" or report["check_passed"] is False
    if family in {"DistinctArgumentRelation", "ConsistencyRelation"}:
        report["visited_calls"] = None
        report["unvisited_calls_after_stop"] = None
        visited_pairs = (
            counts["visited_variable_pairs"]
            if family == "ConsistencyRelation"
            else counts["decisions_started"]
        )
        report["unvisited_pairs_after_stop"] = (
            None
            if candidates["opportunities"] is None
            else max(0, candidates["opportunities"] - visited_pairs) if stopped else 0
        )
        if family == "ConsistencyRelation":
            report["visited_variable_pairs"] = counts["visited_variable_pairs"]
            report["variable_requirements"] = requirements
    else:
        assert candidates["candidate_calls"] is not None
        report["unvisited_calls_after_stop"] = (
            max(0, candidates["candidate_calls"] - counts["visited_calls"])
            if stopped
            else 0
        )
    report["active"] = counts["decisions_completed"] > 0
    report["dormant"] = report["status"] == "completed" and not report["active"]
    report["coverage_observed"] = report["status"] != "observer_error"
    report["coverage_status"] = "observed" if report["coverage_observed"] else "error"
    if not report["coverage_observed"]:
        report["active"] = None
        report["dormant"] = None
    report["variable_precondition_requested"] = variable_precondition_requested
    report["variable_check_skipped"] = variable_check_skipped
    report["fully_validated_active"] = (
        report["active"] is True
        and not variable_check_skipped
        and report["status"] == "completed"
    )
    return finalize_report(report)


class CoverageOptions(argparse.Namespace):
    traces: list[Path]
    invariants: Path
    output: Path
    observer_off: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--invariants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-off", action="store_true")
    args = parser.parse_args(namespace=CoverageOptions())
    logging.basicConfig(level=logging.ERROR)
    import traincheck.utils

    traincheck.utils._suppress_inner_progress = True
    before = {str(p): sha256(p) for p in [*args.traces, args.invariants]}
    trace = read_trace_file_Pandas(list(map(str, args.traces)))
    invariants = read_inv_file(str(args.invariants))
    results: list[ObservationReport] = []
    for index, inv in enumerate(invariants):
        result = observe(trace, inv, enabled=not args.observer_off)
        result["invariant_index"] = index
        results.append(result)
    summary = summarize(results)
    import traincheck.trace.trace_pandas as trace_module

    output: CoverageOutput = {
        "schema_version": 3,
        "pid": os.getpid(),
        "check_relation_first": False,
        "observer_enabled": not args.observer_off,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "trace_module": str(Path(trace_module.__file__).resolve()),
            "packages": {
                p: importlib.metadata.version(p) for p in ("torch", "pandas", "numpy")
            },
        },
        "harness_sha256": sha256(Path(__file__)),
        "raw_process_ids": json_list(trace.get_process_ids(), "raw_process_ids"),
        "raw_thread_ids": json_list(trace.get_thread_ids(), "raw_thread_ids"),
        "trace_rows": len(trace.events),
        "input_sha256": before,
        "inputs_unchanged": all(sha256(Path(p)) == h for p, h in before.items()),
        "summary": dict(summary),
        "per_invariant": results,
    }
    args.output.write_text(json.dumps(output, cls=MDNONEJSONEncoder, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return int(any(result["status"] != "completed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
