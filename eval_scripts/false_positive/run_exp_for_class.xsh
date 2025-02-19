import argparse
import yaml
import os
import time
import subprocess

EXPS = os.listdir(".")
EXPS = [exp for exp in EXPS if os.path.isdir(exp)]

# get the current time (just date and HH:MM)
TIME = $(date "+%m-%d_%H:%M")
READY_TRACES: list[str] = []

def create_tmux_session(name):
    """If the tmux session does not exist, create it."""
    if not $(tmux has-session -t @(name)):
        tmux new-session -d -s @(name)

def run_experiment(experiment, experiment_path, session_name, env_name):
    # run the experiment in a subprocess
    pass

def run_trace_collection(train_programs, valid_programs, parallelism):
    # run trace collection
    all_programs = train_programs + valid_programs # prioritize training programs

    running_experiments: dict[str, subprocess.subprocess] = {}
    for program in all_programs:
        time.sleep(5)  # wait for 5 seconds before starting the next experiment or doing any checking
        if parallelism > 0 and len(running_experiments) < parallelism:
            # run the trace collection
            process = run_experiment(program, session_name, "mldaikon")
            running_experiments[program] = process
    
        # check for failed or completed experiments
        for program, process in running_experiments.items():
            if process.poll() is not None:
                if process.returncode != 0:
                    print(f"Trace collection failed for {program}")
                    # check for the stderr of this process
                    # if the error is due to cuda memory out of space, we can retry the experiment
                    if "CUDA error: out of memory" in process.stderr:
                        print(f"Retrying trace collection for {program} due to CUDA memory error after 1 minute")
                        time.sleep(60)
                        process = run_experiment(program, session_name, "mldaikon", block=True)
                        running_experiments[program] = process
                    else:
                        raise Exception(f"Trace collection failed for {program} due to an unknown error, aborting, stdout: {process.stdout}, stderr: {process.stderr}")
                else:
                    print(f"Trace collection completed for {program}")
                    READY_TRACES.append(program)
                    del running_experiments[program]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment for a class of models')
    parser.add_argument('--bench', type=str, choices=EXPS, default='CNN', help="Benchmark to run")
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


    cd @(args.bench)
    train_programs = os.listdir("trainset")
    valid_programs = os.listdir("validset")

    config = yaml.load(open("setups.yml", "r"))
    setups = config["setups"]
    parallelism = config["trace_collection_parallelism"]

    # use a separate thread to manage parallelism for trace collection
    pass

    