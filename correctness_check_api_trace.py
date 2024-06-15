import argparse

from tqdm import tqdm

from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.trace.trace import Trace, read_trace_file


def check_every_func_pre_has_post(trace: Trace):
    print("Checking if every function pre has a post call.")
    for i, row in enumerate(tqdm(trace.events.rows(named=True))):
        if row["type"] == TraceLineType.FUNC_CALL_PRE:
            try:
                trace.get_func_post_call_idx(i)
            except:
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

    args = parser.parse_args()
    traces = read_trace_file(args.traces)
    assert check_every_func_pre_has_post(traces)


if __name__ == "__main__":
    main()
