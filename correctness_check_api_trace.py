import argparse

from tqdm import tqdm

from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.trace import select_trace_implementation


def check_every_func_pre_has_post(trace) -> bool:
    print("Checking if every function pre has a post call.")
    for row in tqdm(trace.events.rows(named=True)):
        if row["type"] == TraceLineType.FUNC_CALL_PRE:
            try:
                trace.get_post_func_call_record(row["func_call_id"])
            except Exception:
                print(f"Function Pre {row} has no post call.")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Correctness Check for API Trace")
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=True,
        help="Traces files to infer invariants on",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["pandas", "polars", "dict"],
        default="polars",
        help="Specify the backend to use for Trace",
    )

    args = parser.parse_args()
    assert (
        args.backend == "polars"
    ), "Only polars backend is supported for now. See the check_every_func_pre_has_post function."
    _, read_trace_file = select_trace_implementation(args.backend)
    traces = read_trace_file(args.traces)
    assert check_every_func_pre_has_post(traces)


if __name__ == "__main__":
    main()
