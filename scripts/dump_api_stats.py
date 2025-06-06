""" Stats to collect from the API trace and the deployed invariants
1. Number of APIs instrumented that got actually executed
3. Top 10 APIs by execution count
4. Top 10 APIs by execution time
"""

import argparse
import json
import logging
import os

import pandas as pd
from traincheck.trace import read_trace_file_Pandas
from traincheck.utils import register_custom_excepthook


def main(trace, instr_opts, iters: None | int = None):
    funcs_instr_opts = instr_opts["funcs_instr_opts"]
    # print(instr_opts)
    # print(f"Total number of APIs instrumented: {len(events)}")
    events: pd.DataFrame = trace.events
    all_executed_funcs = events["function"].unique()
    print(f"Total number of APIs instrumented: {len(all_executed_funcs)}")
    print(f"Total number of APIs to be instrumented: {len(funcs_instr_opts)}")

    # print("TOP 10 APIs:", top_10_count)

    if iters is not None:
        # print the average number of APIs per iteration
        avg_apis_per_iter = len(events) / iters / 2
        print(f"Average number of APIs per iteration: {avg_apis_per_iter}")

        # print per API stats
        per_func_iter_count = events["function"].value_counts().reset_index()
        per_func_iter_count["per_iter"] = per_func_iter_count["count"] / iters / 2
        # convert to int
        per_func_iter_count["per_iter"] = per_func_iter_count["per_iter"].astype(int)

        print("Per iter stats (function name, count, per_iter):")
        print(per_func_iter_count.sort_values(by="per_iter", ascending=False))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-folder", "-f", help="Folder containing the trace files"
    )
    parser.add_argument(
        "--instr-opts",
        "-o",
        help="Instrumentation options file generated by traincheck when doing selective instrumentation",
    )
    parser.add_argument(
        "--iters", "-i", type=int, help="Number of iterations of the experiment"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    if args.debug:
        register_custom_excepthook()
        logging.basicConfig(level=logging.DEBUG)

    trace_files = [
        f"{args.trace_folder}/{file}"
        for file in os.listdir(args.trace_folder)
        if file.startswith("trace_") or file.startswith("proxy_log.json")
    ]
    logger.info("Reading traces from %s", "\n".join(trace_files))
    trace = read_trace_file_Pandas(trace_files)

    instr_opts = json.load(open(args.instr_opts))

    main(trace, instr_opts, args.iters)
