import argparse
import os
import subprocess
import time

import yaml

EXPS = os.listdir(".")
EXPS = [exp for exp in EXPS if os.path.isdir(exp)]

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


def get_trace_collection_command(program) -> list[str]:
    global PROGRAM_TO_PATH
    return [
        "python",
        "-m",
        "mldaikon.collect_trace",
        "--use-config",
        "--config",
        f"{PROGRAM_TO_PATH[program]}/md-config-var.yml",
        "--output-dir",
        get_trace_collection_dir(program),
    ]


def get_inv_inference_command(setup) -> list[str]:
    cmd = ["python", "-m", "mldaikon.infer_engine", "-f"]
    for program in setup["inputs"]:
        cmd.append(get_trace_collection_dir(program))
    cmd.append("-o")
    cmd.append(get_inv_file_name(setup))
    return cmd


def run_command(cmd, block) -> subprocess.Popen:
    # run the experiment in a subprocess
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
                process = run_command(cmd, block=False)
                running_experiments[program] = process

            # check for failed or completed experiments
            for program, process in running_experiments.copy().items():
                if process.poll() is not None:
                    if process.returncode != 0:
                        print(f"Trace collection failed for {program}")
                        # check for the stderr of this process
                        # if the error is due to cuda memory out of space, we can retry the experiment
                        if "CUDA error: out of memory" in process.stderr:
                            print(
                                f"Retrying trace collection for {program} due to CUDA memory error after 1 minute"
                            )
                            process = run_command(program, block=True)
                            running_experiments[program] = process
                            READY_TRACES.append(program)
                        else:
                            raise Exception(
                                f"Trace collection failed for {program} due to an unknown error, aborting, stdout: {process.stdout.read()}, stderr: {process.stderr.read()}"
                            )
                    else:
                        print(f"Trace collection completed for {program}")
                        READY_TRACES.append(program)
                        del running_experiments[program]


def run_invariant_inference(setups):
    # run invariant inference
    running_setups = []
    leftover_setups = setups.copy()
    while len(leftover_setups) > 0:
        for setup in leftover_setups:
            if (
                not any(setup == running_setup for running_setup, _ in running_setups)
            ) and all(program in READY_TRACES for program in setup["inputs"]):
                # run invariant inference
                print("Running invariant inference for", setup)
                cmd = get_inv_inference_command(setup)
                process = run_command(cmd, block=False)
                leftover_setups.remove(setup)
                running_setups.append((setup, process))
                break
        else:
            print("No trace ready for inference yet, waiting for 10 seconds")
            time.sleep(10)

        # check for failed or completed experiments
        for setup, process in running_setups.copy():
            if process.poll() is not None:
                if process.returncode != 0:
                    print(f"Invariant inference failed for {setup}")
                    # check for the stderr of this process
                    # if the error is due to cuda memory out of space, we can retry the experiment
                    if "killed" in process.stderr:
                        print(f"Retrying invariant inference for {setup} due to OOM")
                        cmd = get_inv_inference_command(setup)
                        process = run_command(cmd, block=True)
                        running_setups.remove((setup, process))
                    else:
                        raise Exception(
                            f"Invariant inference failed for {program} due to an unknown error, aborting, stdout: {process.stdout}, stderr: {process.stderr}"
                        )
                else:
                    print(f"Invariant inference completed for {program}")
                    READY_INVARIANTS.append(setup)
                    running_setups.remove((setup, process))


def run_invariant_checking(valid_programs, setups):
    # TBD
    pass


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

    # start the inference and checking thread
    run_trace_collection(train_programs, valid_programs, parallelism)
