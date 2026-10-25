"""Observe the real offline checker; never replace its predicate or learned rules.

Counters have explicit units. API candidates are calls, DistinctArgument candidates
are within-step argument pairs, and precondition counters are group-verification
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
from pathlib import Path
from unittest.mock import patch

from traincheck.invariant import read_inv_file
from traincheck.invariant.base_cls import GroupedPreconditions
from traincheck.invariant.DistinctArgumentRelation import (
    get_event_data_per_function_per_step,
)
from traincheck.trace.trace_pandas import read_trace_file_Pandas
from traincheck.trace.types import (
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
    MDNONEJSONEncoder,
)

SUPPORTED = {
    "APIContainRelation",
    "ConsistentOutputRelation",
    "ConsistentInputOutputRelation",
    "ThresholdRelation",
    "DistinctArgumentRelation",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observer_sites(function, family):
    """Locate exact predicate ASTs in the installed method, failing closed on drift."""
    lines, first = inspect.getsourcelines(function)
    # Dedent code by its method indentation. A zero-indented line inside a
    # multiline diagnostic string defeats textwrap.dedent's common prefix.
    indent = len(lines[0]) - len(lines[0].lstrip())
    source = "".join(
        line[indent:] if line.startswith(" " * indent) else line for line in lines
    )
    tree = ast.parse(source)
    starts, ends, visits = {}, {}, set()
    predicates = {
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
        if isinstance(node, ast.If) and predicates[family](ast.unparse(node.test)):
            line = first + node.lineno - 1
            starts[line] = ast.unparse(node.test)
            ends[line] = first + node.test.end_lineno - 1
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"func_call_event", "parent_event_type"}
            for t in node.targets
        ):
            visits.add(first + node.lineno - 1)
    expected = 2 if family in {"ThresholdRelation", "DistinctArgumentRelation"} else 1
    if len(starts) != expected:
        raise ValueError(f"Unrecognized checker source for {family}: {starts}")
    return {
        "starts": starts,
        "ends": ends,
        "visits": visits,
        "method_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def candidate_counts(trace, inv):
    family = inv.relation.__name__
    api = (
        inv.params[1]
        if family in {"ConsistentInputOutputRelation", "ThresholdRelation"}
        else inv.params[0]
    )
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


def id_clauses(value):
    found = []
    if isinstance(value, dict):
        if value.get("prop_name") in {
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


def observe(trace, inv, enabled=True):
    """Run one unchanged inv.check(False), counting only observed completed decisions."""
    family = inv.relation.__name__
    report = {
        "relation": family,
        "description": inv.text_description,
        "identity_clauses": id_clauses(inv.precondition.to_dict()),
        "observer_supported": family in SUPPORTED,
    }
    if not enabled:
        # Direct oracle: no opportunity pre-scan, source parsing, or monkeypatches.
        try:
            result = inv.check(trace, check_relation_first=False)
            result.calc_and_set_time_precentage(
                trace.get_start_time(), trace.get_end_time()
            )
            report.update(
                status="completed",
                checker_result=result.to_dict(),
                check_passed=result.check_passed,
                checker_triggered=result.triggered,
            )
        except Exception as exc:
            report.update(
                status="checker_error",
                error={"type": type(exc).__name__, "message": str(exc)},
                check_passed=None,
            )
        report.update(coverage_observed=False, active=None, dormant=None)
        return report
    if family not in SUPPORTED:
        report.update(
            status="unsupported_coverage", check_passed=None, decisions_completed=None
        )
        return report
    report.update(candidate_counts(trace, inv))
    sites = observer_sites(inv.relation.static_check_all, family)
    report["method_source_sha256"] = sites["method_source_sha256"]
    counts = Counter(
        decisions_started=0,
        decisions_completed=0,
        visited_calls=0,
        precondition_group_attempts=0,
        precondition_group_passes=0,
        precondition_group_failures=0,
        precondition_group_errors=0,
    )
    groups = {}
    decision_records = []
    variable_precondition_requested = (
        family == "APIContainRelation"
        and "var_events" in inv.precondition.get_group_names()
        and len(inv.precondition.get_group("var_events")) > 0
        and not inv.precondition.is_group_unconditional("var_events")
    )
    variable_check_skipped = False
    method_code = inv.relation.static_check_all.__code__
    pending = None
    original_verify = GroupedPreconditions.verify

    def verify(precondition, records, group_name, trace_for_metavars=None):
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
        label = "passes" if result else "failures"
        group[label] += 1
        counts["precondition_group_" + label] += 1
        return result

    def trace_event(frame, event, arg):
        nonlocal pending, variable_check_skipped
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
            if frame.f_lineno in sites["visits"]:
                counts["visited_calls"] += 1
            if frame.f_lineno in sites["starts"] and pending is None:
                if variable_precondition_requested and frame.f_locals.get(
                    "skip_var_unchanged_check"
                ):
                    variable_check_skipped = True
                counts["decisions_started"] += 1
                call = frame.f_locals.get(
                    "func_call_id", frame.f_locals.get("parent_func_call_id")
                )
                record = {"func_call_id": call}
                if family == "DistinctArgumentRelation":
                    record = {
                        "pair_func_call_ids": [
                            frame.f_locals[e].get("func_call_id")
                            for e in ("event1", "event2")
                        ]
                    }
                pending = {
                    "start": frame.f_lineno,
                    "end": sites["ends"][frame.f_lineno],
                    "record": record,
                }
        return trace_event

    previous_trace = sys.gettrace()
    if enabled and previous_trace is not None:
        raise RuntimeError("A pre-existing Python trace observer would be overwritten")
    try:
        with (
            patch.object(GroupedPreconditions, "verify", verify)
            if enabled
            else patch.object(GroupedPreconditions, "verify", original_verify)
        ):
            if enabled:
                sys.settrace(trace_event)
            result = inv.check(trace, check_relation_first=False)
            result.calc_and_set_time_precentage(
                trace.get_start_time(), trace.get_end_time()
            )
            report.update(
                status="completed",
                checker_result=result.to_dict(),
                check_passed=result.check_passed,
                checker_triggered=result.triggered,
            )
    except Exception as exc:
        report.update(
            status="checker_error",
            error={"type": type(exc).__name__, "message": str(exc)},
            check_passed=None,
        )
    finally:
        if enabled:
            sys.settrace(previous_trace)
    report.update(counts)
    report["precondition_groups"] = groups
    report["decision_records"] = decision_records
    report["decisions_not_completed"] = (
        counts["decisions_started"] - counts["decisions_completed"]
    )
    stopped = report["status"] != "completed" or report["check_passed"] is False
    if family == "DistinctArgumentRelation":
        report["visited_calls"] = None
        report["unvisited_calls_after_stop"] = None
        report["unvisited_pairs_after_stop"] = (
            max(0, report["opportunities"] - counts["decisions_started"])
            if stopped
            else 0
        )
    else:
        report["unvisited_calls_after_stop"] = (
            max(0, report["candidate_calls"] - counts["visited_calls"])
            if stopped
            else 0
        )
    report["active"] = counts["decisions_completed"] > 0
    report["dormant"] = report["status"] == "completed" and not report["active"]
    report["coverage_observed"] = True
    report["variable_precondition_requested"] = variable_precondition_requested
    report["variable_check_skipped"] = variable_check_skipped
    report["fully_validated_active"] = (
        report["active"]
        and not variable_check_skipped
        and report["status"] == "completed"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--invariants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-off", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    import traincheck.utils

    traincheck.utils._suppress_inner_progress = True
    before = {str(p): sha256(p) for p in [*args.traces, args.invariants]}
    trace = read_trace_file_Pandas(list(map(str, args.traces)))
    invariants = read_inv_file(str(args.invariants))
    results = []
    for index, inv in enumerate(invariants):
        result = observe(trace, inv, enabled=not args.observer_off)
        result["invariant_index"] = index
        results.append(result)
    summary = Counter()
    for result in results:
        summary[result["status"]] += 1
        summary["active_invariants"] += bool(result.get("active"))
        summary["dormant_invariants"] += bool(result.get("dormant"))
        summary["violated_invariants"] += result.get("check_passed") is False
        summary["identity_bearing_invariants"] += bool(result.get("identity_clauses"))
        summary["identity_bearing_active"] += bool(
            result.get("identity_clauses")
        ) and bool(result.get("active"))
    import traincheck.trace.trace_pandas as trace_module

    output = {
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
        "raw_process_ids": trace.get_process_ids(),
        "raw_thread_ids": trace.get_thread_ids(),
        "trace_rows": len(trace.events),
        "input_sha256": before,
        "inputs_unchanged": all(sha256(Path(p)) == h for p, h in before.items()),
        "summary": dict(summary),
        "per_invariant": results,
    }
    args.output.write_text(json.dumps(output, cls=MDNONEJSONEncoder, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return int(bool(summary["checker_error"] or summary["unsupported_coverage"]))


if __name__ == "__main__":
    raise SystemExit(main())
