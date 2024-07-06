import argparse
import datetime
import json
import logging

from tqdm import tqdm

from mldaikon.invariant.base_cls import Invariant
from mldaikon.trace.trace import Trace, read_trace_file


def read_inv_file(file_path: str | list[str]):
    if isinstance(file_path, str):
        file_path = [file_path]
    invs = []
    for file in file_path:
        with open(file, "r") as f:
            for line in f:
                inv_dict = json.loads(line)
                inv = Invariant.from_dict(inv_dict)
                invs.append(inv)
    return invs


def check_engine(traces: list[Trace], invariants: list[Invariant]):
    logger = logging.getLogger(__name__)

    for trace in tqdm(
        traces, desc="Checking invariants on traces", unit="trace", leave=False
    ):
        for inv in tqdm(
            invariants, desc="Checking invariants", unit="invariant", leave=False
        ):
            logger.info("=====================================")
            # logger.debug("Checking invariant %s on trace %s", inv, trace)
            res = inv.check(trace)
            logger.info("Invariant %s on trace %s: %s", inv, trace, res)


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

    check_engine(traces, invs)
