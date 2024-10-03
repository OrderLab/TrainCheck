import argparse
import datetime
import json
import logging
import re

from tqdm import tqdm

from mldaikon.invariant.base_cls import CheckerResult, Invariant, read_inv_file
from mldaikon.trace import Trace, select_trace_implementation


def check_engine(
    traces: list[Trace], invariants: list[Invariant], check_relation_first: bool
) -> list[CheckerResult]:
    logger = logging.getLogger(__name__)
    results: list[CheckerResult] = []
    for trace in tqdm(
        traces, desc="Checking invariants on traces", unit="trace", leave=False
    ):
        for inv in tqdm(
            invariants, desc="Checking invariants", unit="invariant", leave=False
        ):
            assert (
                inv.precondition is not None
            ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
            logger.info("=====================================")
            # logger.debug("Checking invariant %s on trace %s", inv, trace)
            res = inv.check(trace, check_relation_first)
            res.calc_and_set_time_precentage(
                trace.get_start_time(), trace.get_end_time()
            )
            logger.info("Invariant %s on trace %s: %s", inv, trace, res)
            results.append(res)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="(Offline) Invariant Checker for ML Pipelines in Python"
    )
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=True,
        help="Traces files to check invariants on",
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="+",
        required=True,
        help="Invariants files to check on traces",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--report-only-failed",
        action="store_true",
        help="Only report the failed invariants",
    )
    parser.add_argument(
        "--check-relation-first",
        action="store_true",
        help="""Check the relation first, otherwise, the precondition will be checked first. 
            Enabling this flag will make the checker slower, but enables the checker to catch 
            the cases where the invariant still holds even if the precondition is not satisfied, 
            which opens opportunity for precondition refinement. Note that the precondition 
            refinement algorithm is not implemented yet.""",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["pandas", "polars", "dict"],
        default="pandas",
        help="Specify the backend to use for Trace",
    )

    args = parser.parse_args()
    _, read_trace_file = select_trace_implementation(args.backend)
    # read the invariants

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## DEBUG
    time_now = f"{time_now}_relation_first_{args.check_relation_first}"
    # set logging to a file
    logging.basicConfig(
        filename=f"mldaikon_checker_{time_now}.log",
        level=log_level,
    )

    logger = logging.getLogger(__name__)

    # log all the arguments
    logger.info("Checker started with Arguments:")
    for arg, val in vars(args).items():
        logger.info("%s: %s", arg, val)

    logger.info("Reading invaraints from %s", "\n".join(args.invariants))
    invs = read_inv_file(args.invariants)

    logger.info("Reading traces from %s", "\n".join(args.traces))

    traces_string = args.traces[0]

    trace_groups = re.findall(r"\[(.*?)\]", traces_string)

    if len(trace_groups) == 0:
        traces = [read_trace_file(args.traces)]
    else:
        trace_file_groups = [group.split(", ") for group in trace_groups]

        traces = [read_trace_file(group) for group in trace_file_groups]

    results = check_engine(traces, invs, args.check_relation_first)
    results_failed = [res for res in results if not res.check_passed]
    results_not_triggered = [res for res in results if res.triggered is False]

    logger.addHandler(logging.StreamHandler())

    logger.info("Checking finished. %d invariants checked", len(results))
    logger.info(
        "Total failed invariants: %d/%d",
        len(results_failed),
        len(results),
    )
    logger.info(
        "Total passed invariants: %d/%d",
        len(results) - len(results_failed),
        len(results),
    )
    # TODO:
    logger.info(
        "Total invariants that's not triggered: %d/%d",
        len(results_not_triggered),
        len(results),
    )

    # dump the results to a file
    with open(
        f"mldaikon_checker_results_{time_now}.log",
        "w",
    ) as f:
        if args.report_only_failed:
            results = [res for res in results if not res.check_passed]
        res_dicts = [res.to_dict() for res in results]
        json.dump(res_dicts, f, indent=4)
