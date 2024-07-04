import argparse
import datetime
import json
import logging

from mldaikon.invariant.base_cls import Invariant


def read_inv_file(file_path: str | list[str]):
    if isinstance(file_path, str):
        file_path = [file_path]
    invs = []
    for file in file_path:
        with open(file, "r") as f:
            for line in f:
                inv_dict = json.loads(line)
                inv = Invariant.from_dict(inv_dict)
                invs.append(inv)
    return invs


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

    logger = logging.getLogger(__name__)

    logger.info("Reading invaraints from %s", "\n".join(args.invariants))
    invs = read_inv_file(args.invariants)

    # TODO: make this a test (test whether the invariants are still the same after deserialization and serialization)
    from mldaikon.infer_engine import save_invs

    save_invs(invs, "invariants_deserialized.json")
