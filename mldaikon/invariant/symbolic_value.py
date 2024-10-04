import pandas as pd

from mldaikon.trace.types import MD_NONE

ABOVE_ZERO = "above_zero"
BELOW_ZERO = "below_zero"
NON_POSITIVE = "non_positive"
NON_NEGATIVE = "non_negative"
NON_ZERO = "non_zero"
NON_NONE = "non_none"
ANYTHING = "anything"


def is_above_zero(value: int | float) -> bool:
    return value is not None and value > 0


def is_below_zero(value: int | float) -> bool:
    return value is not None and value < 0


def is_non_positive(value: int | float) -> bool:
    return value is not None and value <= 0


def is_non_negative(value: int | float) -> bool:
    return value is not None and value >= 0


def is_non_zero(value: int | float) -> bool:
    return value is not None and value != 0


def is_non_none(value: int | float) -> bool:
    return value is not None


def is_anything(value: int | float) -> bool:
    return True


generalized_value_match = {
    ABOVE_ZERO: is_above_zero,
    BELOW_ZERO: is_below_zero,
    NON_POSITIVE: is_non_positive,
    NON_NEGATIVE: is_non_negative,
    NON_ZERO: is_non_zero,
    NON_NONE: is_non_none,
    ANYTHING: is_anything,
}


def check_generalized_value_match(generalized_type: str, value: int | float) -> bool:
    """Check if a concrete value matches a generalized type."""
    assert (
        generalized_type in generalized_value_match
    ), f"Invalid generalized type: {generalized_type}, expected one of {generalized_value_match.keys()}"
    return generalized_value_match[generalized_type](value)


def generalize_values(values: list[type]) -> None | type | str:
    """Given a list of values, should return a generalized value."""
    if len(values) == 0:
        return None

    if len(set(values)) == 1:
        # no need to generalize
        return values[0]

    all_values = set()
    all_non_none_types = set()
    seen_nan_already = False
    for v in values:
        if pd.isna(v):
            if seen_nan_already:
                continue
            seen_nan_already = True
        all_values.add(v)
        if v is not None and not isinstance(v, MD_NONE):
            all_non_none_types.add(type(v))

    assert (
        len(all_non_none_types) == 1
    ), f"Values should have the same type, got: {set([type(v) for v in values])} ({values})"

    if any(isinstance(v, (int, float)) for v in values):
        all_non_none_values: list[int | float] = [
            v for v in values if isinstance(v, (int, float))
        ]

        min_value = min(all_non_none_values)  # type: ignore
        max_value = max(all_non_none_values)  # type: ignore

        assert (
            min_value != max_value
        ), "Min and max values are the same, you don't need to generalize the values"
        if min_value > 0:
            return ABOVE_ZERO
        elif min_value >= 0:
            return NON_NEGATIVE
        elif max_value < 0:
            return BELOW_ZERO
        elif max_value <= 0:
            return NON_POSITIVE
        elif min_value < 0 and max_value > 0 and 0 not in values:
            return NON_ZERO
        elif (
            min_value < 0 and max_value > 0 and 0 in values and MD_NONE() not in values
        ):
            return NON_NONE
        else:
            # numerical values should always be mergable
            raise ValueError(f"Invalid values: {values}")

    else:
        # for other types, only check if None is in the values
        if MD_NONE() not in values:
            return NON_NONE
        else:
            return ANYTHING
        raise ValueError(f"Cannot generalize, check values: {values}")
