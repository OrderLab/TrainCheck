import argparse
import datetime
import json
import logging
import os
import random
import time

from tqdm import tqdm

import mldaikon.config.config as config
from mldaikon.invariant.base_cls import (
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.invariant.relation_pool import relation_pool
from mldaikon.trace import MDNONEJSONEncoder, select_trace_implementation
from mldaikon.utils import register_custom_excepthook

register_custom_excepthook()

logger = logging.getLogger(__name__)

# set random seed
random.seed(0)


class InferEngine:
    def __init__(self, traces: list):
        self.traces = traces
        pass

    def infer(self, disabled_relations: list[Relation]):
        all_invs = []
        all_failed_hypos = []
        for trace in self.traces:
            for relation in relation_pool:
                if disabled_relations is not None and relation in disabled_relations:
                    logger.info(
                        f"Skipping relation {relation.__name__} as it is disabled"
                    )
                    continue
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

    def infer_multi_trace(self, disabled_relations: list[Relation]):
        hypotheses = self.generate_hypothesis(disabled_relations)
        self.collect_examples(hypotheses)
        invariants, failed_hypos = self.infer_precondition(hypotheses)
        return invariants, failed_hypos

    def generate_hypothesis(
        self, disabled_relations: list[Relation]
    ) -> list[list[Hypothesis]]:
        logger.info("Generating hypotheses")
        hypotheses = []
        for trace in self.traces:
            current_trace_hypotheses: list[Hypothesis] = []
            for relation in relation_pool:
                if disabled_relations is not None and relation in disabled_relations:
                    logger.info(
                        f"Skipping relation {relation.__name__} as it is disabled"
                    )
                    continue
                logger.info(f"Generating hypotheses for relation: {relation.__name__}")
                inferred_hypos = relation.generate_hypothesis(trace)
                logger.info(
                    f"Found {len(inferred_hypos)} hypotheses for relation: {relation.__name__}"
                )
                current_trace_hypotheses.extend(inferred_hypos)
            hypotheses.append(current_trace_hypotheses)
        return hypotheses

    def collect_examples(self, hypotheses: list[list[Hypothesis]]):
        logger.info("Collecting examples")
        for i, trace in enumerate(
            tqdm(self.traces, desc="Collecting examples on traces")
        ):
            for j, trace_hypotheses in enumerate(hypotheses):
                if j == i:
                    # already collected examples for this hypothesis on the same trace that generated it
                    continue
                for hypothesis in trace_hypotheses:
                    hypothesis.invariant.relation.collect_examples(trace, hypothesis)

    def infer_precondition(self, hypotheses: list[list[Hypothesis]]):
        all_hypotheses: list[Hypothesis] = []
        for trace_hypotheses in hypotheses:
            for hypothesis in trace_hypotheses:
                all_hypotheses.append(hypothesis)

        invariants = []
        failed_hypos = []
        for hypothesis in all_hypotheses:
            hypothesis.invariant.num_positive_examples = len(
                hypothesis.positive_examples
            )
            hypothesis.invariant.num_negative_examples = len(
                hypothesis.negative_examples
            )
            precondition = find_precondition(hypothesis, self.traces)
            if precondition is None:
                failed_hypos.append(FailedHypothesis(hypothesis))
            else:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.invariant)

        return invariants, failed_hypos


def save_invs(invs: list[Invariant], output_file: str):
    with open(output_file, "w") as f:
        for inv in invs:
            f.write(json.dumps(inv.to_dict(), cls=MDNONEJSONEncoder))
            f.write("\n")


def save_failed_hypos(failed_hypos: list[FailedHypothesis], output_file: str):
    with open(output_file, "w") as f:
        for failed_hypo in failed_hypos:
            f.write(json.dumps(failed_hypo.to_dict(), cls=MDNONEJSONEncoder))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python"
    )
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=False,
        help="Traces files to infer invariants on",
    )
    parser.add_argument(
        "-f",
        "--trace-folders",
        nargs="+",
        help='Folders containing traces files to infer invariants on. Trace files should start with "trace_" or "proxy_log.json"',
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
        "--disable-relation",
        nargs="+",
        help="Disable specific relations",
    )
    parser.add_argument(
        "--enable-relation",
        nargs="+",
        help="Enable specific relations, overrides disable-relation",
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

    # check if either traces or trace folders are provided
    if args.traces is None and args.trace_folders is None:
        # print help message if neither traces nor trace folders are provided
        parser.print_help()
        parser.error(
            "Please provide either traces or trace folders to infer invariants"
        )

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

    disabled_relations: list[Relation] = []
    if args.disable_relation is not None:
        name_in_relation_pool = {
            relation.__name__: relation for relation in relation_pool
        }
        for rel_name in args.disable_relation:
            if rel_name not in name_in_relation_pool:
                raise ValueError(f"Relation {rel_name} not found in the relation pool")
            disabled_relations.append(name_in_relation_pool[rel_name])  # type: ignore

    if args.enable_relation is not None:
        name_in_relation_pool = {
            relation.__name__: relation for relation in relation_pool
        }
        disabled_relations = [
            relation  # type: ignore
            for relation in relation_pool
            if relation.__name__ not in args.enable_relation
        ]  # type: ignore
        logger.info(
            f"Enabled relations: {[relation.__name__ for relation in relation_pool if relation not in disabled_relations]}"
        )

    config.ENABLE_PRECOND_SAMPLING = not args.disable_precond_sampling
    config.PRECOND_SAMPLING_THRESHOLD = args.precond_sampling_threshold

    time_start = time.time()

    traces = []
    if args.traces is not None:
        logger.info("Reading traces from %s", "\n".join(args.traces))
        traces.append(read_trace_file(args.traces))
    if args.trace_folders is not None:
        for trace_folder in args.trace_folders:
            # file discovery
            trace_files = [
                f"{trace_folder}/{file}"
                for file in os.listdir(trace_folder)
                if file.startswith("trace_") or file.startswith("proxy_log.json")
            ]
            logger.info("Reading traces from %s", "\n".join(trace_files))
            traces.append(read_trace_file(trace_files))

    time_end = time.time()
    logger.info(f"Traces read successfully in {time_end - time_start} seconds.")

    time_start = time.time()
    engine = InferEngine(traces)
    invs, failed_hypos = engine.infer_multi_trace(disabled_relations=disabled_relations)
    time_end = time.time()
    logger.info(f"Inference completed in {time_end - time_start} seconds.")

    save_invs(invs, args.output)
    save_failed_hypos(failed_hypos, args.output + ".failed")
