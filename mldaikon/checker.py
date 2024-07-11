import argparse
import datetime
import json
import logging

from tqdm import tqdm

from mldaikon.invariant.base_cls import CheckerResult, Invariant, read_inv_file
from mldaikon.trace.trace import Trace, read_trace_file


def check_engine(
    traces: list[Trace], invariants: list[Invariant]
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
            res = inv.check(trace)
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

    args = parser.parse_args()

    # read the invariants

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # set logging to a file
    logging.basicConfig(
        filename=f'mldaikon_checker_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
        level=log_level,
    )

    logger = logging.getLogger(__name__)

    logger.info("Reading invaraints from %s", "\n".join(args.invariants))
    invs = read_inv_file(args.invariants)

    logger.info("Reading traces from %s", "\n".join(args.traces))
    traces = [
        read_trace_file(args.traces)
    ]  # TODO: we don't really support multiple traces yet, these are just traces from different processes and they are 'logically' the same trace as they

    results = check_engine(traces, invs)

    # dump the results to a file
    with open(
        f'mldaikon_checker_results_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
        "w",
    ) as f:
        res_dicts = [res.to_dict() for res in results]
        json.dump(res_dicts, f)
