import argparse
import os
import subprocess
import time

import yaml

EXPS = ["CNN", "RNN", "Transformers"]

# get the current time (just date and HH:MM)
READY_TRACES: list[str] = []
READY_INVARIANTS: list[str] = []

PROGRAM_TO_PATH = {}
TRACE_OUTPUT_DIR_PREFIX = "trace_"


def get_trace_collection_dir(program) -> str:
    return f"{TRACE_OUTPUT_DIR_PREFIX}{program}"


def get_inv_file_name(setup: list[str]) -> str:
    setup_names = "_".join(setup["inputs"])
    return f"inv_{setup_names}.json"


def get_checker_output_dir(setup: list[str], program: str) -> str:
    setup_names = "_".join(setup["inputs"])
    return f"checker_{setup_names}_{program}"


def get_setup_key(setup: dict[str, list[str]]) -> tuple[str]:
    return tuple(setup["inputs"])


def get_trace_collection_command(program) -> list[str]:
    global PROGRAM_TO_PATH
    return [
        "python",
        "-m",
        "traincheck.collect_trace",
        "--use-config",
        "--config",
        f"{PROGRAM_TO_PATH[program]}/md-config-var.yml",
        "--output-dir",
        get_trace_collection_dir(program),
    ]


def get_inv_inference_command(setup) -> list[str]:
    cmd = ["python", "-m", "traincheck.infer_engine", "-f"]
    for program in setup["inputs"]:
        cmd.append(get_trace_collection_dir(program))
    cmd.append("-o")
    cmd.append(get_inv_file_name(setup))
    return cmd


def get_inv_checking_command(setup, program) -> list[str]:
    cmd = ["python", "-m", "traincheck.checker", "-f"]
    cmd.append(get_trace_collection_dir(program))
    cmd.append("-i")
    cmd.append(get_inv_file_name(setup))
    cmd.append("-o")
    cmd.append(get_checker_output_dir(setup, program))
    return cmd


def run_command(cmd, block, io_filename) -> subprocess.Popen:
    # run the experiment in a subprocess
    if io_filename:
        with open(io_filename, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if block:
        process.wait()
        if process.returncode != 0:
            raise Exception(
                f"Command failed with return code {process.returncode}, stdout: {process.stdout}, stderr: {process.stderr}"
            )
        return process
    else:
        return process


def run_trace_collection(train_programs, valid_programs, parallelism):
    # run trace collection
    all_programs = train_programs + valid_programs  # prioritize training programs

    running_experiments: dict[str, subprocess.Popen] = {}
    while len(READY_TRACES) < len(all_programs):
        for program in all_programs:
            time.sleep(
                5
            )  # wait for 5 seconds before starting the next experiment or doing any checking
            if (
                program not in READY_TRACES
                and program not in running_experiments
                and parallelism < 0
                or len(running_experiments) < parallelism
            ):
                # run the trace collection
                print("Running trace collection for", program)
                cmd = get_trace_collection_command(program)
                io_filename = f"{program}_trace_collection.log"
                process = run_command(cmd, block=False, io_filename=io_filename)
                running_experiments[program] = process

            # check for failed or completed experiments
            for program, process in running_experiments.copy().items():
                if process.poll() is not None:
                    if process.returncode != 0:
                        raise Exception(
                            f"Trace collection failed for {program} due to an unknown error, aborting"
                        )
                    else:
                        print(f"Trace collection completed for {program}")
                        READY_TRACES.append(program)
                        del running_experiments[program]


def run_invariant_inference(setups):
    # run invariant inference
    running_setups = []
    leftover_setups = setups.copy()
    while len(leftover_setups) > 0 or len(running_setups) > 0:
        for setup in leftover_setups:
            if (
                not any(setup == running_setup for running_setup, _ in running_setups)
            ) and all(program in READY_TRACES for program in setup["inputs"]):
                # run invariant inference
                print("Running invariant inference for", setup)
                cmd = get_inv_inference_command(setup)
                io_filename = "_".join(setup["inputs"]) + "_inference.log"
                process = run_command(cmd, block=False, io_filename=io_filename)
                leftover_setups.remove(setup)
                running_setups.append((setup, process))
                break

        # check for failed or completed experiments
        for setup, process in running_setups.copy():
            if process.poll() is not None:
                if process.returncode != 0:
                    print(f"Invariant inference failed for {setup}")
                    # check for the stderr of this process
                    # if the error is due to cuda memory out of space, we can retry the experiment
                    raise Exception(
                        f"Invariant inference failed for {setup} due to an unknown error, aborting"
                    )
                else:
                    print(f"Invariant inference completed for {setup}")
                    READY_INVARIANTS.append(setup)
                    running_setups.remove((setup, process))

    print("Exiting invariant inference loop")


def run_invariant_checking(valid_programs, setups):
    # run invariant checking for each valid program for each setup
    running_setups = {}
    leftover_setups = {get_setup_key(setup): valid_programs.copy() for setup in setups}

    while len(leftover_setups) > 0 or len(running_setups) > 0:
        for setup in READY_INVARIANTS:
            setup_key = get_setup_key(setup)
            if setup_key not in leftover_setups:
                continue
            for program in leftover_setups[setup_key].copy():
                if program not in READY_TRACES:
                    continue
                # run invariant checking
                print(f"Running invariant checking for {setup_key} on {program}")
                cmd = get_inv_checking_command(setup, program)
                io_filename = f"{program}_invariant_checking.log"
                process = run_command(cmd, block=False, io_filename=io_filename)
                if setup_key not in running_setups:
                    running_setups[setup_key] = []
                running_setups[setup_key].append((program, process))
                leftover_setups[setup_key].remove(program)

            if len(leftover_setups[setup_key]) == 0:
                del leftover_setups[setup_key]

        # check for failed or completed experiments
        for setup_key, processes in running_setups.copy().items():
            for program, process in processes.copy():
                if process.poll() is not None:
                    if process.returncode != 0:
                        print(f"Invariant checking failed for {setup_key} on {program}")
                        # check for the stderr of this process
                        # if the error is due to cuda memory out of space, we can retry the experiment
                        raise Exception(
                            f"Invariant checking failed for {setup_key} on {program} due to an unknown error, aborting"
                        )
                    else:
                        print(
                            f"Invariant checking completed for {setup_key} on {program}"
                        )
                        processes.remove((program, process))
                        if len(processes) == 0:
                            del running_setups[setup_key]
    print("Exiting invariant checking loop")


def cleanup_trace_files():
    # list out all folders with the prefix TRACE_OUTPUT_DIR_PREFIX
    files = os.listdir(".")
    trace_dirs = [file for file in files if file.startswith(TRACE_OUTPUT_DIR_PREFIX)]
    for trace_dir in trace_dirs:
        print(f"Removing {trace_dir}")
        os.system(f"rm -rf {trace_dir}")

    # remove all traincheck logs
    files = os.listdir(".")
    traincheck_logs = [file for file in files if file.startswith("traincheck_")]
    for log in traincheck_logs:
        print(f"Removing {log}")
        os.system(f"rm {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment for a class of models")
    parser.add_argument(
        "--bench", type=str, choices=EXPS, default="CNN", help="Benchmark to run"
    )
    args = parser.parse_args()

    # steps
    """
    1. Create a subprocess for each program in the benchmark
    2. Run collection in parallel, one program per subprocess
    3. Parallelism should be controlled by "trace_collection_parallelism" in the config file
    4. Wait for collection to finish
    4.5 During the wait, if any training setup has been satisfied, start invariant inference.
    5. After completion of inference, start checking and collect results.
    """

    os.chdir(args.bench)
    cleanup_trace_files()
    train_programs = os.listdir("trainset")
    valid_programs = os.listdir("validset")
    # remove non folder and "data"
    train_programs = [
        program
        for program in train_programs
        if os.path.isdir(f"trainset/{program}") and program != "data"
    ]
    valid_programs = [
        program
        for program in valid_programs
        if os.path.isdir(f"validset/{program}") and program != "data"
    ]

    for program in train_programs:
        PROGRAM_TO_PATH[program] = os.path.abspath(f"trainset/{program}")
    for program in valid_programs:
        PROGRAM_TO_PATH[program] = os.path.abspath(f"validset/{program}")

    config = yaml.load(open("setups.yml", "r"), Loader=yaml.FullLoader)
    setups = config["setups"]
    parallelism = config["trace_collection_parallelism"]
    # print(setups)
    import threading

    # start the invariant inference thread
    inference_thread = threading.Thread(target=run_invariant_inference, args=(setups,))
    inference_thread.start()

    # start the invariant checking thread
    checking_thread = threading.Thread(
        target=run_invariant_checking, args=(valid_programs, setups)
    )
    checking_thread.start()
    # start the inference and checking thread
    run_trace_collection(train_programs, valid_programs, parallelism)
