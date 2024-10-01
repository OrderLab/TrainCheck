import logging
import polars as pl
import itertools

from mldaikon.config import config
from mldaikon.invariant.base_cls import (
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    APIParam,
    FailedHypothesis,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace


class VarPreserveRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the VarPreserveRelation."""

        logger = logging.getLogger(__name__)

        # Check if the DataFrame is empty
        if trace.events.is_empty() or "func_name" not in trace.events.columns:
            logger.warning("The trace contains no events.")
            return [], []

        # Group events by 'func_name'
        func_names = trace.events["func_name"].unique().to_list()
        if not func_names:
            logger.warning("No 'func_name' found in the events.")
            return [], []

        hypotheses = {}
        group_name = "events"

        func_events_df = trace.events

        func_call_ids = func_events_df["func_call_id"].unique().to_list()

        for func_call_id in func_call_ids:
            # Filter events for this function
            func_events_df = trace.events.filter(pl.col("func_call_id") == func_call_id)
            func_events = [row for row in func_events_df.iter_rows(named=True)]

            if len(func_events) != 2:
                # Not enough events to compare
                continue

            func_name = func_events[0]["func_name"]

            # Initialize hypothesis for this function
            param = APIParam(api_full_name=func_name)

            # Bug:
            if func_name not in hypotheses:
                hypotheses[func_name] = Hypothesis(
                    invariant=Invariant(
                        relation=VarPreserveRelation,
                        params=[param],
                        precondition=None,
                        text_description=f"Events of function {func_name} are similar.",
                    ),
                    positive_examples=ExampleList({group_name}),
                    negative_examples=ExampleList({group_name}),
                )

            event1 = func_events[0]
            event2 = func_events[1]

            if events_are_similar(event1, event2, tolerance=1e-6):
                # Positive example: events are similar
                example = Example({group_name: [event1, event2]})
                hypotheses[func_name].positive_examples.add_example(example)
            else:
                # Negative example: events are not similar
                example = Example({group_name: [event1, event2]})
                hypotheses[func_name].negative_examples.add_example(example)

        # Evaluate hypotheses
        invariants = []
        failed_hypos = []

        for func_name, hypothesis in hypotheses.items():
            pos_count = len(hypothesis.positive_examples.examples)
            neg_count = len(hypothesis.negative_examples.examples)
            total = pos_count + neg_count

            if total == 0:
                continue

            positive_ratio = pos_count / total

            # TODO: replace it with a threshold
            if positive_ratio >= 0.8:
                # Infer preconditions
                preconditions = find_precondition(hypothesis, keys_to_skip=["time"])

                if preconditions is not None:
                    hypothesis.invariant.precondition = preconditions
                invariants.append(hypothesis.invariant)
            else:
                logger.debug(
                    f"Function {func_name}: positive_ratio {positive_ratio} below threshold"
                )
                failed_hypos.append(FailedHypothesis(hypothesis))

        return invariants, failed_hypos


def events_are_similar(event1, event2, tolerance=1e-6):
    """Compare two events for similarity, allowing for small differences in numerical values."""
    # Get the set of all keys in both events
    keys = set(
        [
            "self_stat.min",
            "self_stat.max",
            "self_stat.mean",
            "self_stat.std",
            "self_stat.shape",
        ]
    )

    for key in keys:
        value1 = event1.get(key)
        value2 = event2.get(key)

        # Both values are None or missing
        if value1 is None and value2 is None:
            continue

        # One of the values is None or missing
        if value1 is None or value2 is None:
            return False

        # Compare the values
        if not values_are_equal(value1, value2, tolerance):
            return False

    return True


def values_are_equal(value1, value2, tolerance=1e-6):
    """Compare two values for equality, with tolerance for numerical types."""

    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return abs(value1 - value2) <= tolerance

    elif isinstance(value1, str) and isinstance(value2, str):
        return value1 == value2

    elif isinstance(value1, list) and isinstance(value2, list):
        if len(value1) != len(value2):
            return False
        return all(
            values_are_equal(v1, v2, tolerance) for v1, v2 in zip(value1, value2)
        )

    elif isinstance(value1, dict) and isinstance(value2, dict):
        return events_are_similar(value1, value2, tolerance)

    else:
        # For other types, compare for exact equality
        return value1 == value2
