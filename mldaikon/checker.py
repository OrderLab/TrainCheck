import argparse
import datetime
import logging

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
