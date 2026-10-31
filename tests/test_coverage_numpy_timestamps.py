"""Coverage observes valid dataframe timestamps without losing nanoseconds."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools.fresh_validation import check_coverage as coverage
from tools.fresh_validation.contracts import mapping, required, sequence
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.invariant.base_cls import (
    GroupedPreconditions,
    Invariant,
    Preconditions,
    UnconditionalPrecondition,
    VarTypeParam,
)
from traincheck.invariant.consistency_relation import ConsistencyRelation
from traincheck.trace.trace_pandas import TracePandas

START = 2**53 + 1
END = START + 10


def integer_timestamp_trace(values: list[int]) -> TracePandas:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(values):
        for timestamp in (START, END):
            rows.append(
                {
                    "type": TraceLineType.STATE_CHANGE,
                    "time": timestamp,
                    "process_id": 1,
                    "thread_id": 2,
                    "var_name": f"weight-{index}",
                    "var_type": "torch.nn.Parameter",
                    "attributes.value": value,
                }
            )
    return TracePandas(pd.DataFrame(rows), truncate_incomplete_func_calls=False)


def consistency_rule() -> Invariant:
    return Invariant(
        ConsistencyRelation,
        [
            VarTypeParam("torch.nn.Parameter", "value"),
            VarTypeParam("torch.nn.Parameter", "value"),
        ],
        GroupedPreconditions({"var": Preconditions([UnconditionalPrecondition()])}),
        "fixture value agreement with integer nanosecond timestamps",
    )


@pytest.mark.parametrize("values,passed", [([7, 7], True), ([7, 8], False)])
def test_real_pandas_final_lifetime_preserves_checker_parity(
    values: list[int], passed: bool
) -> None:
    trace = integer_timestamp_trace(values)
    for attributes in trace.get_var_insts().values():
        state = attributes["value"][-1]
        assert isinstance(state.liveness.end_time, np.int64)
        assert state.liveness.end_time == END
    rule = consistency_rule()
    observed = coverage.observe(trace, rule)
    direct = coverage.observe(integer_timestamp_trace(values), rule, enabled=False)
    assert observed["status"] == "completed", observed
    assert observed["checker_result"] == direct["checker_result"]
    assert observed["check_passed"] is passed
    assert observed["coverage_observed"] is True
    assert observed["active"] is True
    assert observed["opportunities"] == observed["decisions_completed"] == 1
    decisions = observed["decision_records"]
    assert len(decisions) == 1
    decision = mapping(decisions[0], "decision")
    pair = sequence(required(decision, "variable_value_pair"), "variable pair")
    assert len(pair) == 2
    for raw in pair:
        identity = mapping(raw, "identity")
        for key, expected in (
            ("event_time", END),
            ("liveness_start", START),
            ("liveness_end", END),
        ):
            assert isinstance(identity[key], int) and not isinstance(
                identity[key], bool
            )
            assert identity[key] == expected
    # The emitted observation uses ordinary JSON scalars, with exact large ints.
    encoded = json.dumps(decisions, allow_nan=False)
    assert json.loads(encoded) == decisions


@pytest.mark.parametrize(
    "value,expected",
    [
        (START, START),
        (2**100, 2**100),
        (np.int64(START), START),
        (np.int64(2**63 - 1), 2**63 - 1),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (np.int32(-3), -3),
        (0.125, 0.125),
        (np.float16(0.125), 0.125),
        (np.float32(0.1), float(np.float32(0.1))),
        (np.float64(0.125), 0.125),
        (np.longdouble(0.125), 0.125),
    ],
)
def test_timestamp_scalar_conversion_is_exact(
    value: object, expected: int | float
) -> None:
    actual = coverage.dataframe_timestamp(value, "fixture time")
    assert type(actual) is type(expected)
    assert actual == expected
    assert json.loads(json.dumps(actual, allow_nan=False)) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        None,
        "123",
        {},
        [],
        complex(1, 0),
        np.complex64(1),
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32("nan"),
        np.float32("inf"),
        np.float64("-inf"),
        np.timedelta64(1, "ns"),
        np.timedelta64(1, "D"),
        np.datetime64("2026-01-01"),
        pd.NA,
    ],
)
def test_timestamp_rejects_invalid_or_nonfinite_values(value: object) -> None:
    with pytest.raises(ValueError, match="fixture time"):
        coverage.dataframe_timestamp(value, "fixture time")


def test_extended_float_must_be_exactly_representable() -> None:
    # Some platforms implement longdouble as float64. Exercise extra precision
    # only where NumPy actually supplies it; ordinary float64 is covered above.
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        value = np.longdouble(1) + np.finfo(np.longdouble).eps
        assert value != float(value)
        with pytest.raises(ValueError, match="fixture time"):
            coverage.dataframe_timestamp(value, "fixture time")
