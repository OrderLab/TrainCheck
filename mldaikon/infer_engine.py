import argparse
import datetime
import json
import logging
import time

from mldaikon.invariant.base_cls import Invariant
from mldaikon.invariant.relation_pool import relation_pool
from mldaikon.trace.trace import Trace, read_trace_file

logger = logging.getLogger(__name__)


class InferEngine:
    def __init__(self, traces: list[Trace]):
        self.traces = traces
        pass

    def infer(self):
        all_invs = []
        for trace in self.traces:
            for relation in relation_pool:
                logger.info(f"Infering invariants for relation: {relation.__name__}")
                invs = relation.infer(trace)
                logger.info(
                    f"Found {len(invs)} invariants for relation: {relation.__name__}"
                )
                all_invs.extend(invs)
        logger.info(f"Found {len(all_invs)} invariants.")
        return all_invs


def save_invs(invs: list[Invariant], output_file: str):
    with open(output_file, "w") as f:
        for inv in invs:
            f.write(json.dumps(inv.to_dict()))
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
