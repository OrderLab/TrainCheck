from typing import Callable, Type

from traincheck.trace.trace import Trace
from traincheck.trace.trace_dict import TraceDict, read_trace_file_dict
from traincheck.trace.trace_pandas import TracePandas, read_trace_file_Pandas
from traincheck.trace.trace_polars import TracePolars, read_trace_file_polars
from traincheck.trace.types import MDNONEJSONDecoder, MDNONEJSONEncoder

__all__ = ["select_trace_implementation", "MDNONEJSONDecoder", "MDNONEJSONEncoder"]


def select_trace_implementation(choice: str) -> tuple[Type[Trace], Callable]:
    """Selects the trace implementation based on the choice.

    Args:
        choice (str): The choice of the trace implementation.
            - "polars": polars pyarrow dataframe based trace implementation (deprecated)
            - "pandas": pandas numpy dataframe based trace implementation (schemaless)
            - "dict": pure python dictionary based trace implementation
    """
    if choice == "polars":
        return TracePolars, read_trace_file_polars
    elif choice == "pandas":
        return TracePandas, read_trace_file_Pandas
    elif choice == "dict":
        return TraceDict, read_trace_file_dict

    raise ValueError(f"Invalid choice: {choice}")
