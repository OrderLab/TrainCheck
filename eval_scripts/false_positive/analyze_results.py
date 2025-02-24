import os
from collections import defaultdict

import pandas as pd
import yaml
from run_exp_for_class import EXPS, get_checker_output_dir, get_setup_key

from mldaikon.checker import parse_checker_results
from mldaikon.invariant.base_cls import Invariant, read_inv_file


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
    global_violated_invs = {}
    global_passed_invs = {}
    global_not_triggered_invs = {}
    global_all_invs = {}
    for bench, setups in all_results.items():
        os.chdir(bench)
        global_violated_invs[bench] = {}
        global_passed_invs[bench] = {}
        global_not_triggered_invs[bench] = {}
        global_all_invs[bench] = {}
        for setup, programs in setups.items():
            global_all_invs[bench][setup] = None
            global_violated_invs[bench][setup] = defaultdict(lambda: 0)
            global_passed_invs[bench][setup] = defaultdict(lambda: 0)
            global_not_triggered_invs[bench][setup] = defaultdict(lambda: 0)
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

                if global_all_invs[bench][setup] is None:
                    global_all_invs[bench][setup] = invariants
                else:
                    assert global_all_invs[bench][setup] == invariants
                assert len(failed) + len(passed) + len(non_triggered) == len(invariants)

                for res in failed:
                    global_violated_invs[bench][setup][
                        (Invariant.from_dict(res["invariant"]))
                    ] += 1
                for res in non_triggered:
                    global_not_triggered_invs[bench][setup][
                        (Invariant.from_dict(res["invariant"]))
                    ] += 1
                for res in passed:
                    global_passed_invs[bench][setup][
                        (Invariant.from_dict(res["invariant"]))
                    ] += 1
            print(f"[{bench}] {setup} statistics:\n", end="")
            print(
                f"\tTotal invariants: \t{len(global_all_invs[bench][setup])}\n", end=""
            )
            print(
                f"\tViolated invariants: \t{len(global_violated_invs[bench][setup])} ({len(global_violated_invs[bench][setup]) / len(global_all_invs[bench][setup]) * 100:.2f}%)\n",
                end="",
            )
            print(
                f"\tPassed invariants: \t{len(global_passed_invs[bench][setup])} ({len(global_passed_invs[bench][setup]) / len(global_all_invs[bench][setup])*100:.2f}%)\n",
                end="",
            )
            print(
                f"\tNot triggered invariants: \t{len(global_not_triggered_invs[bench][setup])} ({len(global_not_triggered_invs[bench][setup]) / len(global_all_invs[bench][setup])*100:.2f}%)\n",
            )
            # for each setup, compute unique invariants being violated, passed, not triggered
            print(
                f"\tUnique violated invariants: \t{len(global_violated_invs[bench][setup])}\n",
                end="",
            )
            # compute the statistics as well, similar to what is done below, but for each setup separately
            fp_row_data = []
            for inv in global_all_invs[bench][setup]:
                if inv not in global_violated_invs[bench][setup]:
                    fp_row_data.append(
                        {
                            "text_description": inv.text_description,
                            "relation": inv.relation.__name__,
                            "status": "passed",
                            "have_precondition": not inv.precondition.is_unconditional(),
                        }
                    )
                else:
                    fp_row_data.append(
                        {
                            "text_description": inv.text_description,
                            "relation": inv.relation.__name__,
                            "status": "violated",
                            "have_precondition": not inv.precondition.is_unconditional(),
                        }
                    )
            df = pd.DataFrame(fp_row_data)
            df.to_csv(f"{bench}_{setup}_fp_stats.csv")

            print("\tDistribution of relation types:")
            print(df["relation"].value_counts(normalize=True) * 100)
            print("\tPrecentage of Conditional Invariants:")
            print(df[df["have_precondition"]].shape[0] / df.shape[0] * 100)
            print("\tPrecentage of Unconditional Invariants that are false positives:")
            print(
                df[(not df["have_precondition"]) & (df["status"] == "violated")].shape[
                    0
                ]
                / df[not df["have_precondition"]].shape[0]
                * 100
            )
            print("\tPrecentage of Conditional Invariants that are false positives:")
            print(
                df[(df["have_precondition"]) & (df["status"] == "violated")].shape[0]
                / df[df["have_precondition"]].shape[0]
                * 100
            )
            print("\tPrecentage of Invariants that are false positives:")
            print(df[df["status"] == "violated"].shape[0] / df.shape[0] * 100)

            print("\n")

        os.chdir("..")

        # now, I want to have some statistics about the invariants
        # 1. distribution of relation types
        # 2. numbers of invariants that are violated, passed, not triggered with/without precondition
        # 2.5 total number of invariants with a precondition

        # step 1, convert the results to a pandas dataframe
        fp_row_data = []
        for setup, fp_invs in global_violated_invs[bench].items():
            for inv in global_all_invs[bench][setup]:
                if inv not in fp_invs:
                    fp_row_data.append(
                        {
                            "text_description": inv.text_description,
                            "relation": inv.relation.__name__,
                            "status": "passed",
                            "have_precondition": not inv.precondition.is_unconditional(),
                        }
                    )
                else:
                    fp_row_data.append(
                        {
                            "text_description": inv.text_description,
                            "relation": inv.relation.__name__,
                            "status": "violated",
                            "have_precondition": not inv.precondition.is_unconditional(),
                        }
                    )
        df = pd.DataFrame(fp_row_data)
        df.to_csv(f"{bench}_fp_stats.csv")

        # step 2, compute the statistics
        print("\tDistribution of relation types:")
        print(df["relation"].value_counts(normalize=True) * 100)
        print("\tPrecentage of Conditional Invariants:")
        print(df[df["have_precondition"]].shape[0] / df.shape[0] * 100)
        print("\tPrecentage of Unconditional Invariants that are false positives:")
        print(
            df[(not df["have_precondition"]) & (df["status"] == "violated")].shape[0]
            / df[not df["have_precondition"]].shape[0]
            * 100
        )
        print("\tPrecentage of Conditional Invariants that are false positives:")
        print(
            df[(df["have_precondition"]) & (df["status"] == "violated")].shape[0]
            / df[df["have_precondition"]].shape[0]
            * 100
        )
        print("\tPrecentage of Invariants that are false positives:")
        print(df[df["status"] == "violated"].shape[0] / df.shape[0] * 100)
