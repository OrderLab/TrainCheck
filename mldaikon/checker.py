import argparse
import datetime
import json
import logging
import os

from tqdm import tqdm

from mldaikon.invariant.base_cls import CheckerResult, Invariant, read_inv_file
from mldaikon.trace import MDNONEJSONEncoder, Trace, select_trace_implementation
from mldaikon.utils import register_custom_excepthook

register_custom_excepthook()


def parse_checker_results(file_name: str):
    with open(file_name, "r") as f:
        lines = f.readlines()

    all_results: list[dict] = []
    current_res_str = ""
    for line in lines:
        if line.startswith("{") and current_res_str:
            all_results.append(json.loads(current_res_str))
            current_res_str = ""
        current_res_str += line

    if current_res_str:
        all_results.append(json.loads(current_res_str))
    return all_results


def check_engine(
    traces: list[Trace], invariants: list[Invariant], check_relation_first: bool
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
            res = inv.check(trace, check_relation_first)
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
    parser.add_argument(
        "--check-relation-first",
        action="store_true",
        help="""Check the relation first, otherwise, the precondition will be checked first. 
            Enabling this flag will make the checker slower, but enables the checker to catch 
            the cases where the invariant still holds even if the precondition is not satisfied, 
            which opens opportunity for precondition refinement. Note that the precondition 
            refinement algorithm is not implemented yet.""",
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
        "-o",
        "--output-dir",
        type=str,
        help="Output folder to store the results, defaulted to mldaikon_checker_results_{timestamp}/",
    )

    args = parser.parse_args()
    _, read_trace_file = select_trace_implementation(args.backend)
    # read the invariants

    # check if either traces or trace folders are provided
    if args.traces is None and args.trace_folders is None:
        # print help message if neither traces nor trace folders are provided
        parser.print_help()
        parser.error(
            "Please provide either traces or trace folders to infer invariants"
        )

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## DEBUG
    time_now = f"{time_now}_relation_first_{args.check_relation_first}"
    # set logging to a file
    logging.basicConfig(
        filename=f"mldaikon_checker_{time_now}.log",
        level=log_level,
    )

    logger = logging.getLogger(__name__)

    # log all the arguments
    logger.info("Checker started with Arguments:")
    for arg, val in vars(args).items():
        logger.info("%s: %s", arg, val)

    # create the output folder if not exists
    if not args.output_dir:
        args.output_dir = f"mldaikon_checker_results_{time_now}"
    os.makedirs(args.output_dir, exist_ok=True)

    # copy the invariants to the output folder
    for inv_file in args.invariants:
        os.system(f"cp {inv_file} {args.output_dir}/invariants.json")

    logger.info("Reading invaraints from %s", "\n".join(args.invariants))
    invs = read_inv_file(args.invariants)

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

    results = check_engine(traces, invs, args.check_relation_first)
    results_failed = [res for res in results if not res.check_passed]
    results_not_triggered = [res for res in results if res.triggered is False]

    logger.addHandler(logging.StreamHandler())

    logger.info("Checking finished. %d invariants checked", len(results))
    logger.info(
        "Total failed invariants: %d/%d",
        len(results_failed),
        len(results),
    )
    logger.info(
        "Total passed invariants: %d/%d",
        len(results) - len(results_failed),
        len(results),
    )
    # TODO:
    logger.info(
        "Total invariants that's not triggered: %d/%d",
        len(results_not_triggered),
        len(results),
    )

    # dump the results to a file
    with open(
        os.path.join(args.output_dir, f"failed_{time_now}.log"),
        "w",
    ) as f:
        for res in results:
            if not res.check_passed:
                json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                f.write("\n")

    with open(
        os.path.join(args.output_dir, f"not_triggered_{time_now}.log"),
        "w",
    ) as f:
        for res in results:
            if not res.triggered:
                json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                f.write("\n")

    with open(
        os.path.join(args.output_dir, f"passed_{time_now}.log"),
        "w",
    ) as f:
        for res in results:
            if res.check_passed and res.triggered:
                json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                f.write("\n")
