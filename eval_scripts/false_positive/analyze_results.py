import os

import yaml
from run_exp_for_class import EXPS, get_checker_output_dir, get_setup_key

from mldaikon.checker import parse_checker_results
from mldaikon.invariant.base_cls import read_inv_file


def discover_checker_results() -> dict:
    """Requires changing to the directory where the checker output files are stored."""

    with open("setups.yml", "r") as f:
        setups = yaml.load(f, Loader=yaml.FullLoader)

    results = {}  # setup: [(program, results), ...]
    valid_programs = [
        f
        for f in os.listdir("validset")
        if os.path.isdir(os.path.join("validset", f)) and f != "data"
    ]
    # for setup in setups['setups']:
    for setup in setups["setups"]:
        for program in valid_programs:
            checker_output_dir = get_checker_output_dir(setup, program)
            if os.path.exists(checker_output_dir):
                if get_setup_key(setup) not in results:
                    results[get_setup_key(setup)] = []
                results[get_setup_key(setup)].append((program, checker_output_dir))
            else:
                print(
                    f"Warning: checker output directory for {program} in {setup} does not exist, skipping. {checker_output_dir}"
                )
    return results


if __name__ == "__main__":
    all_results = {}
    for bench in EXPS:
        os.chdir(bench)
        results = discover_checker_results()
        if results:
            all_results[bench] = results
        os.chdir("..")

    # for each bench, for each setup, for each program, parse the results
    for bench, setups in all_results.items():
        os.chdir(bench)
        for setup, programs in setups.items():
            for program, checker_output_dir in programs:
                # find the results files
                result_and_inv_files = os.listdir(checker_output_dir)
                assert "invariants.json" in result_and_inv_files
                inv_file = os.path.join(checker_output_dir, "invariants.json")
                failed_results = [
                    f for f in result_and_inv_files if f.startswith("failed")
                ][0]
                failed_file = os.path.join(checker_output_dir, failed_results)
                passed_results = [
                    f for f in result_and_inv_files if f.startswith("passed")
                ][0]
                passed_file = os.path.join(checker_output_dir, passed_results)
                not_triggered_results = [
                    f for f in result_and_inv_files if f.startswith("not_triggered")
                ][0]
                not_triggered_file = os.path.join(
                    checker_output_dir, not_triggered_results
                )
                # analyzing the results
                failed = parse_checker_results(failed_file)
                passed = parse_checker_results(passed_file)
                non_triggered = parse_checker_results(not_triggered_file)
                invariants = read_inv_file(inv_file)
                assert len(failed) + len(passed) + len(non_triggered) == len(invariants)

                print(
                    f"Results for {program} in {setup} in {bench}, failed: {len(failed)} ({len(failed) / len(invariants)}), passed: {len(passed)} ({len(passed) / len(invariants)}), non_triggered: {len(non_triggered)} ({len(non_triggered) / len(invariants)})"
                )
        os.chdir("..")
