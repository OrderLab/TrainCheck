import argparse
import datetime
import json
import logging
import os
import random
import time

from tqdm import tqdm

import mldaikon.config.config as config
from mldaikon.invariant import (
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
    find_precondition,
    relation_pool,
)
from mldaikon.trace import MDNONEJSONEncoder, select_trace_implementation
from mldaikon.utils import register_custom_excepthook

register_custom_excepthook()

logger = logging.getLogger(__name__)

# set random seed
random.seed(0)


class InferEngine:
    def __init__(self, traces: list, disabled_relations: list[Relation] = []):
        self.traces = traces
        self.all_stages = set()
        for trace in traces:
            # FIXME: we don't fully support multi-stage traces with inconsistent stage annotations yet, not sure about the impact to invariant correctness.
            if trace.is_stage_annotated():
                self.all_stages.update(trace.get_all_stages())
        self.disabled_relations = disabled_relations

    def infer_multi_trace(self):
        hypotheses = self.generate_hypothesis()
        hypotheses = self.prune_incorrect_hypos(hypotheses)
        self.collect_examples(hypotheses)
        invariants, failed_hypos = self.infer_precondition(hypotheses)
        return invariants, failed_hypos

    def generate_hypothesis(self) -> dict[Hypothesis, list[int]]:
        """Generate hypotheses for all traces using all relations in the relation pool, excluding disabled relations

        Returns:
            dict[Hypothesis, list[int]]: A dictionary mapping hypotheses to the indices of traces that support them
        """

        logger.info("Generating hypotheses")
        hypotheses_and_trace_idxs: dict[Hypothesis, list[int]] = {}
        for trace_idx, trace in enumerate(tqdm(self.traces, desc="Scanning Traces")):
            for relation in relation_pool:
                if self.disabled_relations and relation in self.disabled_relations:
                    logger.info(
                        f"Skipping relation {relation.__name__} as it is disabled"
                    )
                    continue
                logger.info(f"Generating hypotheses for relation: {relation.__name__}")
                inferred_hypos = relation.generate_hypothesis(trace)
                logger.info(
                    f"Found {len(inferred_hypos)} hypotheses for relation: {relation.__name__}"
                )
                for hypo in tqdm(
                    inferred_hypos, desc="Merging Hypotheses with existing ones"
                ):
                    if hypo not in hypotheses_and_trace_idxs:
                        hypotheses_and_trace_idxs[hypo] = [trace_idx]
                        # print("Already new one")
                    else:
                        # print("Already inside")
                        hypotheses_and_trace_idxs[hypo].append(trace_idx)
                        # get the key in the dictionary
                        original_hypos = [
                            original_hypo
                            for original_hypo in hypotheses_and_trace_idxs.keys()
                            if original_hypo == hypo
                        ]
                        assert len(original_hypos) == 1

                        original_hypo = original_hypos[0]
                        orig_pos_num = len(original_hypo.positive_examples)
                        orig_neg_num = len(original_hypo.negative_examples)

                        original_hypo.positive_examples.examples.extend(
                            hypo.positive_examples.examples
                        )
                        original_hypo.negative_examples.examples.extend(
                            hypo.negative_examples.examples
                        )

                        # sanity check for modification correctness
                        original_hypos = [
                            original_hypo
                            for original_hypo in hypotheses_and_trace_idxs.keys()
                            if original_hypo == hypo
                        ]
                        assert len(original_hypos) == 1

                        original_hypo = original_hypos[0]
                        assert len(
                            original_hypo.positive_examples
                        ) == orig_pos_num + len(
                            hypo.positive_examples
                        ), f"{len(original_hypo.positive_examples)} != {orig_pos_num} + {len(hypo.positive_examples)}"
                        assert len(
                            original_hypo.negative_examples
                        ) == orig_neg_num + len(
                            hypo.negative_examples
                        ), f"{len(original_hypo.negative_examples)} != {orig_neg_num} + {len(hypo.negative_examples)}"

        return hypotheses_and_trace_idxs

    def collect_examples(self, hypotheses: dict[Hypothesis, list[int]]):
        logger.info("Collecting examples")
        for hypo, trace_idxs in hypotheses.items():
            for trace_idx, trace in enumerate(self.traces):
                if trace_idx in trace_idxs:
                    continue
                logger.info(f"Collecting examples for hypothesis: {hypo}")
                hypo.invariant.relation.collect_examples(trace, hypo)

    def prune_incorrect_hypos(self, hypotheses: dict[Hypothesis, list[int]]):
        """Prune incorrect hypotheses based on the collected examples"""

        # rule 1: remove hypotheses with only one positive example
        hypotheses = {
            hypo: trace_idxs
            for hypo, trace_idxs in hypotheses.items()
            if len(hypo.positive_examples) > 1
        }

        return hypotheses

    def infer_precondition(self, hypotheses: dict[Hypothesis, list[int]]):
        """TODO: move the precondition inference driving code into Hypothesis.get_invariant()"""

        all_hypotheses: list[Hypothesis] = []
        for hypo in hypotheses:
            all_hypotheses.append(hypo)

        invariants = []
        failed_hypos = []
        for hypothesis in all_hypotheses:
            precondition = find_precondition(hypothesis, self.traces)
            if precondition is None:
                failed_hypos.append(FailedHypothesis(hypothesis))
            else:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.get_invariant(self.all_stages))
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
    engine = InferEngine(traces, disabled_relations)
    invs, failed_hypos = engine.infer_multi_trace()

    # sort the invariants by the text description
    invs = sorted(invs, key=lambda x: x.text_description)

    time_end = time.time()
    logger.info(f"Inference completed in {time_end - time_start} seconds.")

    save_invs(invs, args.output)
    save_failed_hypos(failed_hypos, args.output + ".failed")
