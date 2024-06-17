import argparse
import logging
import time

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
            for r in relation_pool:
                logger.info(f"Infering invariants for relation: {r}")
                invs = r.infer(trace)
                logger.info(f"Found {len(invs)} invariants for relation: {r}")
                all_invs.extend(invs)
        logger.info(f"Found {len(all_invs)} invariants.")
        return invs


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

    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args()

    # traces = [read_trace_file(t) for t in args.traces]
    time_start = time.time()
    logger.info(f"Reading traces from {args.traces}")
    traces = [read_trace_file(args.traces)]
    time_end = time.time()
    logger.info(f"Traces read successfully in {time_end - time_start} seconds.")

    engine = InferEngine(traces)
    invs = engine.infer()
