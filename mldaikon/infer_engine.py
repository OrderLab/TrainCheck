import argparse
import datetime
import json
import logging
import random
import sys
import threading
import time
import traceback

import mldaikon.config.config as config
from mldaikon.invariant.base_cls import FailedHypothesis, Invariant
from mldaikon.invariant.relation_pool import relation_pool
from mldaikon.trace import select_trace_implementation

# from mldaikon.trace.trace import Trace, read_trace_file

logger = logging.getLogger(__name__)

# set random seed
random.seed(0)


class InferEngine:
    def __init__(self, traces: list):
        self.traces = traces
        pass

    def infer(self):
        all_invs = []
        all_failed_hypos = []
        for trace in self.traces:
            for relation in relation_pool:
                logger.info(f"Infering invariants for relation: {relation.__name__}")
                invs, failed_hypos = relation.infer(trace)
                logger.info(
                    f"Found {len(invs)} invariants for relation: {relation.__name__}"
                )
                all_invs.extend(invs)
                all_failed_hypos.extend(failed_hypos)
        logger.info(
            f"Found {len(all_invs)} invariants, {len(all_failed_hypos)} failed hypotheses due to precondition inference"
        )
        return all_invs, all_failed_hypos


def save_invs(invs: list[Invariant], output_file: str):
    with open(output_file, "w") as f:
        for inv in invs:
            f.write(json.dumps(inv.to_dict()))
            f.write("\n")


def save_failed_hypos(failed_hypos: list[FailedHypothesis], output_file: str):
    with open(output_file, "w") as f:
        for failed_hypo in failed_hypos:
            f.write(json.dumps(failed_hypo.to_dict()))
            f.write("\n")


def handle_excepthook(typ, message, stack):
    """Custom exception handler

    Print detailed stack information with local variables
    """
    logger = logging.getLogger("mldaikon")

    if issubclass(typ, KeyboardInterrupt):
        sys.__excepthook__(typ, message, stack)
        return

    stack_info = traceback.StackSummary.extract(
        traceback.walk_tb(stack), capture_locals=True
    ).format()
    logger.critical("An exception occured: %s: %s.", typ, message)
    for i in stack_info:
        logger.critical(i.encode().decode("unicode-escape"))

    # re-raise the exception so that vscode debugger can catch it and give useful information
    raise typ(message) from None


def thread_excepthook(args):
    """Exception notifier for threads"""
    logger = logging.getLogger("threading")

    exc_type = args.exc_type
    exc_value = args.exc_value
    exc_traceback = args.exc_traceback
    _ = args.thread
    if issubclass(exc_type, KeyboardInterrupt):
        threading.__excepthook__(args)
        return

    stack_info = traceback.StackSummary.extract(
        traceback.walk_tb(exc_traceback), capture_locals=True
    ).format()
    logger.critical("An exception occured: %s: %s.", exc_type, exc_value)
    for i in stack_info:
        logger.critical(i.encode().decode("unicode-escape"))

    # re-raise the exception so that vscode debugger can catch it and give useful information
    raise exc_type(exc_value) from None


sys.excepthook = handle_excepthook
threading.excepthook = thread_excepthook

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python"
    )
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=True,
        help="Traces files to infer invariants on",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="invariants.json",
        help="Output file to save invariants",
    )
    parser.add_argument(
        "--disable-precond-sampling",
        action="store_true",
        help="Disable sampling of positive and negative examples for precondition inference [By default sampling is enabled]",
    )
    parser.add_argument(
        "--precond-sampling-threshold",
        type=int,
        default=config.PRECOND_SAMPLING_THRESHOLD,
        help="The number of samples to take for precondition inference, if the number of samples is larger than this threshold, we will sample this number of samples [Default: 10000]",
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

    Trace, read_trace_file = select_trace_implementation(args.backend)

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # set logging to a file
    logging.basicConfig(
        filename=f'mldaikon_infer_engine_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)20s()] - %(message)s",
    )

    config.ENABLE_PRECOND_SAMPLING = not args.disable_precond_sampling
    config.PRECOND_SAMPLING_THRESHOLD = args.precond_sampling_threshold

    time_start = time.time()
    logger.info("Reading traces from %s", "\n".join(args.traces))
    traces = [read_trace_file(args.traces)]
    time_end = time.time()
    logger.info(f"Traces read successfully in {time_end - time_start} seconds.")

    time_start = time.time()
    engine = InferEngine(traces)
    invs, failed_hypos = engine.infer()
    time_end = time.time()
    logger.info(f"Inference completed in {time_end - time_start} seconds.")

    save_invs(invs, args.output)
    save_failed_hypos(failed_hypos, args.output + ".failed")
