"""Semantic unit tests for violation summary helpers.

Tests verify the extraction and aggregation logic — pure function tests that
do not depend on trace loading or the inference algorithm.
"""

import pytest

from traincheck.invariant.base_cls import APIParam, CheckerResult, Invariant
from traincheck.invariant.cover_relation import FunctionCoverRelation
from traincheck.reporting.checker_report import (
    _build_violation_entry,
    _extract_violation_steps,
    build_violations_summary,
)

# ---------------------------------------------------------------------------
# _extract_violation_steps
# ---------------------------------------------------------------------------


def test_extract_steps_basic():
    trace = [{"meta_vars.step": 1}, {"meta_vars.step": 3}]
    assert _extract_violation_steps(trace) == [1, 3]


def test_extract_steps_missing_key():
    trace = [{"function": "foo"}, {"meta_vars.step": 5}]
    assert _extract_violation_steps(trace) == [5]


def test_extract_steps_none_trace():
    assert _extract_violation_steps(None) == []


def test_extract_steps_empty_trace():
    assert _extract_violation_steps([]) == []


def test_extract_steps_none_value_skipped():
    trace = [{"meta_vars.step": None}, {"meta_vars.step": 2}]
    assert _extract_violation_steps(trace) == [2]


# ---------------------------------------------------------------------------
# Helper to build minimal CheckerResult fixtures
# ---------------------------------------------------------------------------


def _make_invariant() -> Invariant:
    return Invariant(
        relation=FunctionCoverRelation,
        params=[
            APIParam("torch.distributed.is_initialized"),
            APIParam("torch.nn.modules.module.Module.eval"),
        ],
        precondition=None,
        text_description="test invariant",
    )


def _make_result(steps: list[int] | None, check_passed: bool = False) -> CheckerResult:
    trace = [{"meta_vars.step": s} for s in steps] if steps is not None else None
    return CheckerResult(
        invariant=_make_invariant(),
        trace=trace,
        check_passed=check_passed,
        triggered=True,
    )


# ---------------------------------------------------------------------------
# _build_violation_entry
# ---------------------------------------------------------------------------


def test_build_entry_fields_present():
    result = _make_result(steps=[1, 2, 3])
    entry = _build_violation_entry(result)
    assert "display_name" in entry
    assert "relation_type" in entry
    assert "first_step" in entry
    assert "last_step" in entry
    assert "occurrences" in entry


def test_build_entry_step_values():
    result = _make_result(steps=[1, 5, 3])
    entry = _build_violation_entry(result)
    assert entry["first_step"] == 1
    assert entry["last_step"] == 5
    assert entry["occurrences"] == 3


def test_build_entry_no_steps():
    # trace records that have no meta_vars.step key
    trace = [{"function": "foo"}]
    result = CheckerResult(
        invariant=_make_invariant(),
        trace=trace,
        check_passed=False,
        triggered=True,
    )
    entry = _build_violation_entry(result)
    assert entry["first_step"] is None
    assert entry["last_step"] is None


def test_build_entry_display_name_is_string():
    result = _make_result(steps=[1])
    entry = _build_violation_entry(result)
    assert isinstance(entry["display_name"], str)
    assert len(entry["display_name"]) > 0


def test_build_entry_relation_type():
    result = _make_result(steps=[1])
    entry = _build_violation_entry(result)
    assert entry["relation_type"] == "FunctionCoverRelation"


# ---------------------------------------------------------------------------
# build_violations_summary
# ---------------------------------------------------------------------------


def test_summary_no_failures():
    results = [_make_result(steps=[1], check_passed=True)]
    summary = build_violations_summary(results)
    assert summary["distinct_invariants_violated"] == 0
    assert summary["violations"] == []
    assert summary["first_violation_step"] is None


def test_summary_with_failures():
    results = [
        _make_result(steps=[2, 4], check_passed=False),
        _make_result(steps=[1], check_passed=False),
        _make_result(steps=[5], check_passed=True),
    ]
    summary = build_violations_summary(results)
    assert summary["distinct_invariants_violated"] == 2
    assert summary["first_violation_step"] == 1
    assert len(summary["violations"]) == 2


def test_summary_first_step_across_violations():
    results = [
        _make_result(steps=[10, 20], check_passed=False),
        _make_result(steps=[3], check_passed=False),
    ]
    summary = build_violations_summary(results)
    assert summary["first_violation_step"] == 3


def test_summary_no_step_data():
    # violation with trace records that carry no step information
    result = CheckerResult(
        invariant=_make_invariant(),
        trace=[{"function": "foo"}],
        check_passed=False,
        triggered=True,
    )
    summary = build_violations_summary([result])
    assert summary["distinct_invariants_violated"] == 1
    assert summary["first_violation_step"] is None
    assert summary["violations"][0]["first_step"] is None
