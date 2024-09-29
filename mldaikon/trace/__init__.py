from typing import Callable, Type

from mldaikon.trace.trace import Trace
from mldaikon.trace.trace_dict import TraceDict, read_trace_file_dict
from mldaikon.trace.trace_pandas import TracePandas, read_trace_file_Pandas
from mldaikon.trace.trace_polars import TracePolars, read_trace_file_polars


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
