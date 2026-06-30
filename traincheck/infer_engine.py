import argparse
import datetime
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import traincheck.config.config as config
import traincheck.utils as _tc_utils
from traincheck.invariant import (
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
    find_precondition,
    relation_pool,
)
from traincheck.trace import MDNONEJSONEncoder, select_trace_implementation
from traincheck.utils import register_custom_excepthook

register_custom_excepthook()

logger = logging.getLogger(__name__)

# set random seed
random.seed(0)


# === parallel inference workers ===
# Used by the parallel paths in both generate_hypothesis and infer_precondition.
# Each worker process keeps its own copy of the traces in this module global so that
# tasks reuse them instead of re-pickling the (potentially large) traces per task.
_WORKER_TRACES: list | None = None


def _config_snapshot() -> dict:
    """CLI-overridable config values that must be propagated to workers.

    These are set in main() at runtime; workers started with the 'spawn' method
    begin from a fresh interpreter and would otherwise see only the module
    defaults (harmless under 'fork'). Add any future CLI-overridable config here.
    """
    return {
        "ENABLE_PRECOND_SAMPLING": config.ENABLE_PRECOND_SAMPLING,
        "PRECOND_SAMPLING_THRESHOLD": config.PRECOND_SAMPLING_THRESHOLD,
    }


def _worker_init(traces: list, config_snapshot: dict) -> None:
    """Initializer run once per worker process.

    Stores the traces in a module global (one copy per worker, not per task) and
    restores CLI-overridable config values (see _config_snapshot). Also suppresses
    the per-example inner progress bars, otherwise every worker would fight over
    the terminal.
    """
    global _WORKER_TRACES
    _WORKER_TRACES = traces
    for name, value in config_snapshot.items():
        setattr(config, name, value)
    _tc_utils._suppress_inner_progress = True


def _genhypo_worker_task(task: tuple):
    """Generate hypotheses for one (trace_idx, relation) pair in a worker process.

    Returns (trace_idx, relation, inferred_hypos); the parent merges them in a
    deterministic order so the result matches the serial path.
    """
    trace_idx, relation = task
    assert _WORKER_TRACES is not None, "Worker traces were not initialized"
    return trace_idx, relation, relation.generate_hypothesis(_WORKER_TRACES[trace_idx])


def _precond_worker_task(hypothesis: Hypothesis):
    """Run precondition inference for a single hypothesis in a worker process.

    Returns just the precondition (picklable); the parent applies it to its own
    copy of the hypothesis.
    """
    assert _WORKER_TRACES is not None, "Worker traces were not initialized"
    return find_precondition(hypothesis, _WORKER_TRACES)


class InferEngine:
    def __init__(
        self,
        traces: list,
        disabled_relations: list[Relation] = [],
        num_workers: int = 1,
    ):
        self.traces = traces
        self.num_workers = max(1, num_workers)
        self.all_stages = set()
        for trace in traces:
            # FIXME: we don't fully support multi-stage traces with inconsistent stage annotations yet, not sure about the impact to invariant correctness.
            if trace.is_stage_annotated():
                self.all_stages.update(trace.get_all_stages())
        self.disabled_relations = disabled_relations

    def infer_multi_trace(self):
        hypotheses = self.generate_hypothesis()
        hypotheses, incorrect_hypos = self.prune_incorrect_hypos(hypotheses)
        self.collect_examples(hypotheses)
        invariants, failed_hypos = self.infer_precondition(hypotheses)
        return invariants, failed_hypos + incorrect_hypos

    def generate_hypothesis(self) -> dict[Hypothesis, list[int]]:
        """Generate hypotheses for all traces using all relations in the relation pool, excluding disabled relations

        Returns:
            dict[Hypothesis, list[int]]: A dictionary mapping hypotheses to the indices of traces that support them
        """
        logger.info("============= GENERATING HYPOTHESIS =============")
        hypotheses_and_trace_idxs: dict[Hypothesis, list[int]] = {}
        hypo_lookup: dict[Hypothesis, Hypothesis] = {}
        active_relations = [
            r for r in relation_pool if r not in self.disabled_relations
        ]

        if self.num_workers <= 1:
            self._generate_hypothesis_serial(
                active_relations, hypotheses_and_trace_idxs, hypo_lookup
            )
        else:
            self._generate_hypothesis_parallel(
                active_relations, hypotheses_and_trace_idxs, hypo_lookup
            )

        total = len(hypotheses_and_trace_idxs)
        print(f"\n  {total} hypotheses generated across all relations")
        logger.info(f"Finished generating hypotheses, found {total} hypotheses")
        return hypotheses_and_trace_idxs

    def _merge_hypotheses(
        self,
        inferred_hypos: list[Hypothesis],
        trace_idx: int,
        hypotheses_and_trace_idxs: dict[Hypothesis, list[int]],
        hypo_lookup: dict[Hypothesis, Hypothesis],
    ) -> None:
        """Merge one trace's inferred hypotheses into the running collection.

        Mutates shared state (the dicts and the stored hypotheses' example lists),
        so this is always run in the parent process, never in a worker.
        """
        for hypo in inferred_hypos:
            if hypo not in hypotheses_and_trace_idxs:
                hypotheses_and_trace_idxs[hypo] = [trace_idx]
                hypo_lookup[hypo] = hypo
            else:
                hypotheses_and_trace_idxs[hypo].append(trace_idx)
                original_hypo = hypo_lookup[hypo]
                orig_num_pos_exps = len(original_hypo.positive_examples)
                orig_num_neg_exps = len(original_hypo.negative_examples)
                original_hypo.positive_examples.examples.extend(
                    hypo.positive_examples.examples
                )
                original_hypo.negative_examples.examples.extend(
                    hypo.negative_examples.examples
                )

                assert len(
                    hypo_lookup[hypo].positive_examples
                ) == orig_num_pos_exps + len(
                    hypo.positive_examples
                ), f"Expected {orig_num_pos_exps} + {len(hypo.positive_examples)} positive examples, got {len(hypo_lookup[hypo].positive_examples)}"
                assert len(
                    hypo_lookup[hypo].negative_examples
                ) == orig_num_neg_exps + len(
                    hypo.negative_examples
                ), f"Expected {orig_num_neg_exps} + {len(hypo.negative_examples)} negative examples, got {len(hypo_lookup[hypo].negative_examples)}"

    def _generate_hypothesis_serial(
        self,
        active_relations: list,
        hypotheses_and_trace_idxs: dict[Hypothesis, list[int]],
        hypo_lookup: dict[Hypothesis, Hypothesis],
    ) -> None:
        """Serial hypothesis generation (preserves the original behavior exactly)."""
        n_traces = len(self.traces)
        rel_bar_fmt = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        for trace_idx, trace in enumerate(self.traces):
            tqdm.write(f"\n[Trace {trace_idx + 1}/{n_traces}] Generating hypotheses")
            with tqdm(
                active_relations,
                bar_format=rel_bar_fmt,
                unit="relation",
                leave=True,
            ) as rel_bar:
                for relation in rel_bar:
                    rel_bar.set_description(f"  {relation.__name__}")
                    logger.info(
                        f"Generating hypotheses for relation: {relation.__name__}"
                    )
                    t0 = time.time()
                    inferred_hypos = relation.generate_hypothesis(trace)
                    elapsed = time.time() - t0
                    tqdm.write(
                        f"  {relation.__name__}: {len(inferred_hypos)} hypotheses ({elapsed:.1f}s)"
                    )
                    logger.info(
                        f"Found {len(inferred_hypos)} hypotheses for {relation.__name__} "
                        f"on trace {trace_idx + 1}/{n_traces}"
                    )
                    self._merge_hypotheses(
                        inferred_hypos,
                        trace_idx,
                        hypotheses_and_trace_idxs,
                        hypo_lookup,
                    )

    def _generate_hypothesis_parallel(
        self,
        active_relations: list,
        hypotheses_and_trace_idxs: dict[Hypothesis, list[int]],
        hypo_lookup: dict[Hypothesis, Hypothesis],
    ) -> None:
        """Parallel hypothesis generation.

        Each (trace, relation) pair is an independent task: relation.generate_hypothesis
        is CPU-bound and pure w.r.t. its (per-worker) trace, so we fan them out across
        processes. Workers only PRODUCE hypothesis lists; the merge into shared state
        stays in the parent. Results are merged in trace-then-relation order (identical
        to the serial path) regardless of completion order, so the output is unchanged.
        """
        n_traces = len(self.traces)
        tasks = [
            (trace_idx, relation)
            for trace_idx in range(n_traces)
            for relation in active_relations
        ]
        print(
            f"\nGenerating hypotheses for {n_traces} trace(s) x {len(active_relations)} "
            f"relations using {self.num_workers} worker processes"
        )

        # collect results keyed by (trace_idx, relation name) so we can merge them in a
        # deterministic order afterwards (completion order from as_completed is arbitrary).
        results: dict[tuple[int, str], list[Hypothesis]] = {}
        bar_fmt = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        with tqdm(total=len(tasks), bar_format=bar_fmt, unit="task") as pbar:
            pbar.set_description("generating")
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_worker_init,
                initargs=(self.traces, _config_snapshot()),
            ) as executor:
                future_to_task = {
                    executor.submit(_genhypo_worker_task, task): task for task in tasks
                }
                for future in as_completed(future_to_task):
                    trace_idx, relation, inferred_hypos = future.result()
                    results[(trace_idx, relation.__name__)] = inferred_hypos
                    logger.info(
                        f"Found {len(inferred_hypos)} hypotheses for {relation.__name__} "
                        f"on trace {trace_idx + 1}/{n_traces}"
                    )
                    pbar.update(1)

        # deterministic merge: trace order, then relation order (matching the serial path)
        for trace_idx in range(n_traces):
            for relation in active_relations:
                self._merge_hypotheses(
                    results[(trace_idx, relation.__name__)],
                    trace_idx,
                    hypotheses_and_trace_idxs,
                    hypo_lookup,
                )

    def collect_examples(self, hypotheses: dict[Hypothesis, list[int]]):
        logger.info("============= COLLECTING EXAMPLES =============")
        cross_trace_hypos = [
            (hypo, trace_idxs)
            for hypo, trace_idxs in hypotheses.items()
            if len(set(range(len(self.traces))) - set(trace_idxs)) > 0
        ]
        if cross_trace_hypos:
            print(
                f"\nCollecting examples for {len(cross_trace_hypos)} cross-trace hypotheses"
            )
        for hypo, trace_idxs in cross_trace_hypos:
            for trace_idx, trace in enumerate(self.traces):
                if trace_idx in trace_idxs:
                    continue
                logger.info(
                    f"Collecting examples for hypothesis: {hypo} on trace {trace_idx + 1}/{len(self.traces)}"
                )
                hypo.invariant.relation.collect_examples(trace, hypo)

    def prune_incorrect_hypos(self, hypotheses: dict[Hypothesis, list[int]]):
        """Prune incorrect hypotheses based on the collected examples"""
        incorrect_hypos = []
        correct_hypos = {}
        for hypo, trace_idxs in hypotheses.items():
            if len(hypo.positive_examples) > 1:
                correct_hypos[hypo] = trace_idxs
            else:
                incorrect_hypos.append(
                    FailedHypothesis(hypo, "only one positive example")
                )
        n_pruned = len(incorrect_hypos)
        n_kept = len(correct_hypos)
        print(f"  {n_pruned} pruned (insufficient examples) → {n_kept} remaining")
        return correct_hypos, incorrect_hypos

    def infer_precondition(self, hypotheses: dict[Hypothesis, list[int]]):
        """TODO: move the precondition inference driving code into Hypothesis.get_invariant()"""
        logger.info("============= INFERING PRECONDITIONS =============")
        all_hypotheses = list(hypotheses.keys())
        total = len(all_hypotheses)
        print(f"\nInferring preconditions for {total} hypotheses")

        invariants = []
        failed_hypos = []
        bar_fmt = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        def _record_result(hypothesis: Hypothesis, precondition):
            """Apply a worker/serial precondition result to the parent's hypothesis."""
            if precondition is None:
                failed_hypos.append(
                    FailedHypothesis(hypothesis, "Precondition not found")
                )
            else:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.get_invariant(self.all_stages))

        if self.num_workers <= 1:
            # Serial path (preserves the original behavior exactly).
            _tc_utils._suppress_inner_progress = True
            try:
                with tqdm(total=total, bar_format=bar_fmt, unit="hypo") as pbar:
                    pbar.set_description("0 done · 0 failed")
                    for hypo_idx, hypothesis in enumerate(all_hypotheses):
                        logger.info(
                            f"Inferring precondition for hypothesis {hypo_idx + 1}/{total}: "
                            f"{hypothesis.invariant.text_description}"
                        )
                        precondition = find_precondition(hypothesis, self.traces)
                        _record_result(hypothesis, precondition)
                        pbar.set_description(
                            f"{len(invariants)} done · {len(failed_hypos)} failed"
                        )
                        pbar.update(1)
            finally:
                _tc_utils._suppress_inner_progress = False
        else:
            # Parallel path: each hypothesis's precondition inference is independent and
            # CPU-bound, so we fan it out across processes (threads would serialize on the
            # GIL). Workers load the traces once via the initializer (one copy per worker,
            # not per task); the parent applies each returned precondition to its own
            # hypothesis copy.
            print(f"  Using {self.num_workers} worker processes")
            with tqdm(total=total, bar_format=bar_fmt, unit="hypo") as pbar:
                pbar.set_description("0 done · 0 failed")
                with ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_worker_init,
                    initargs=(self.traces, _config_snapshot()),
                ) as executor:
                    future_to_hypo = {
                        executor.submit(_precond_worker_task, hypothesis): hypothesis
                        for hypothesis in all_hypotheses
                    }
                    for future in as_completed(future_to_hypo):
                        hypothesis = future_to_hypo[future]
                        precondition = future.result()
                        _record_result(hypothesis, precondition)
                        pbar.set_description(
                            f"{len(invariants)} done · {len(failed_hypos)} failed"
                        )
                        pbar.update(1)

        print(f"  {len(invariants)} invariants · {len(failed_hypos)} failed")
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


def main():
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
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for hypothesis generation and precondition "
        "inference. 1 (default) runs serially. Use a value > 1 to parallelize across "
        "cores (0 = use all available cores). Note: each worker keeps its own copy of "
        "the traces, so memory scales with the number of workers.",
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

    # get current process ID
    pid = os.getpid()
    # set logging to a file
    logging.basicConfig(
        filename=f'traincheck_infer_engine_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{pid}.log',
        level=log_level,
        format="%(asctime)s - [TrainCheck] %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)20s()] - %(message)s",
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

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    time_start = time.time()
    engine = InferEngine(traces, disabled_relations, num_workers=num_workers)
    invs, failed_hypos = engine.infer_multi_trace()

    # sort the invariants by the text description
    invs = sorted(invs, key=lambda x: x.text_description)

    time_end = time.time()
    logger.info(f"Inference completed in {time_end - time_start} seconds.")

    save_invs(invs, args.output)
    save_failed_hypos(failed_hypos, args.output + ".failed")


if __name__ == "__main__":
    main()
