import logging
import polars as pl
import itertools
from collections import defaultdict

from mldaikon.config import config
from mldaikon.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
    APIParam,
    FailedHypothesis
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Liveness, Trace


class EqualRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the EqualRelation."""

        logger = logging.getLogger(__name__)

        # Check if the DataFrame is empty
        if trace.events.is_empty():
            logger.warning("The trace contains no events.")
            return [], []

        # Group events by 'func_name'
        func_names = trace.events['func_name'].unique().to_list()
        if not func_names:
            logger.warning("No 'func_name' found in the events.")
            return [], []

        hypotheses = {}
        group_name = 'events'

        for func_name in func_names:
            # Filter events for this function
            func_events_df = trace.events.filter(pl.col('func_name') == func_name)
            func_events = [row for row in func_events_df.iter_rows(named=True)]

            if len(func_events) < 2:
                # Not enough events to compare
                continue

            # Initialize hypothesis for this function
            param = APIParam(api_full_name=func_name)
            hypotheses[func_name] = Hypothesis(
                invariant=Invariant(
                    relation=EqualRelation,
                    params=[param],
                    precondition=None,
                    text_description=f"Events of function {func_name} are similar."
                ),
                positive_examples=ExampleList({group_name}),
                negative_examples=ExampleList({group_name})
            )

            # Compare each pair of events
            for event1, event2 in itertools.combinations(func_events, 2):
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

        # TODO: Implement failed hypotheses
        failed_hypos = []

        for hypo, hypothesis in hypotheses.items():
            pos_count = len(hypothesis.positive_examples.examples)
            neg_count = len(hypothesis.negative_examples.examples)
            total = pos_count + neg_count

            if total == 0:
                continue

            positive_ratio = pos_count / total
            
            # TODO: replace it with a threshold
            if positive_ratio >= 0.5:
                # Infer preconditions
                preconditions = find_precondition(
                    hypothesis,
                    keys_to_skip=['time']
                )

                if preconditions is not None:
                    hypothesis.invariant.precondition = preconditions
                invariants.append(hypothesis.invariant)
            else:
                logger.debug(f"Function {hypo}: positive_ratio {positive_ratio} below threshold")
                failed_hypos.append(FailedHypothesis(hypothesis))

        return invariants, failed_hypos


def events_are_similar(event1, event2, tolerance=1e-6):
    """Compare two events for similarity, allowing for small differences in numerical values."""
    # Get the set of all keys in both events
    keys = set(event1.keys()) | set(event2.keys())
    keys.remove('time')
    keys.remove('var_name')

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
        return all(values_are_equal(v1, v2, tolerance) for v1, v2 in zip(value1, value2))

    elif isinstance(value1, dict) and isinstance(value2, dict):
        return events_are_similar(value1, value2, tolerance)

    else:
        # For other types, compare for exact equality
        return value1 == value2
