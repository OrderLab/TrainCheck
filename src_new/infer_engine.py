import argparse
import logging

from invariant import relation_pool
from trace import read_trace_file

logger = logging.getLogger(__name__)

class InferEngine:
    def __init__(self, traces: list[str]):
        self.traces = traces
        pass

    def infer(self):
        invs = []
        for trace in self.traces:
            for r in relation_pool:
                logger.info(f"Infering invariants for relation: {r}")
                invs.append(r.infer(trace))
                logger.info(f"Found {len(invs)} invariants for relation: {r}")        
        logger.info(f"Found {len(invs)} invariants.")
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

    traces = [read_trace_file(t) for t in args.traces]

    engine = InferEngine(traces)
    invs = engine.infer()
