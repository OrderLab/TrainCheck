import argparse
import datetime
import json
import logging
import random
import time

import mldaikon.config.config as config
from mldaikon.invariant.base_cls import FailedHypothesis, Invariant
from mldaikon.invariant.relation_pool import relation_pool
from mldaikon.trace.trace import Trace, read_trace_file

logger = logging.getLogger(__name__)

# set random seed
random.seed(0)


class InferEngine:
    def __init__(self, traces: list[Trace]):
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
    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # set logging to a file
    logging.basicConfig(
        filename=f'mldaikon_infer_engine_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
    invs = engine.infer()
    time_end = time.time()
    logger.info(f"Inference completed in {time_end - time_start} seconds.")

    save_invs(invs, args.output)
    save_failed_hypos(invs, args.output + ".failed")
