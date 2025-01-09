import argparse
import os
import signal
import subprocess

# configs
$RAISE_SUBPROC_ERROR = True
os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--res_folder", type=str, required=False)
args = parser.parse_args()

SELC_INV_FILE = "sampled_100_invariants.json"
COMMIT = $(git rev-parse --short HEAD).strip()

if args.res_folder:
    RES_FOLDER = args.res_folder
else:
    RES_FOLDER = f"perf_eval_res_{COMMIT}"

MICRO_FOLDER = "overhead-micro"
E2E_FOLDER = "overhead-e2e"


rm -rf @(RES_FOLDER)
mkdir @(RES_FOLDER)

# run microbenchmark
cd @(MICRO_FOLDER)
bash collect_wrapper_overhead.sh
cd ..
mv @(MICRO_FOLDER)/wrapper_overhead_micro.csv @(RES_FOLDER)/

def get_all_GPU_pids():
    pids = $(nvidia-smi | grep 'python' | awk '{ print $5 }').split()
    return pids

def run_cmd(cmd: str, kill_sec: int):
    with open("cmd_output.log", "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        try:
            output, _ = p.communicate(timeout=kill_sec)
        except subprocess.TimeoutExpired:
            print(f"Timeout: {kill_sec} seconds, killing the process {p.pid}")
            # os.kill(
            #     p.pid, signal.SIGTERM
            # )  # send SIGTERM to the process group NOTE: the signal will be delivered here again
            # p.kill()
            p.terminate() # sends SIGTERM

            if str(p.pid + 1) in get_all_GPU_pids():
                print("Found additional plausible GPU process, killing it...")
                kill -9 @(p.pid + 1)
                
            print("Killed the running process...")

# run e2e benchmark
def run_exp(kill_sec: int = 100, workload: str = "mnist"):
    print(f"Running experiments for {workload}")

    ORIG_PY = "main.py"
    SETTRACE_PY = "main_settrace.py"
    RUN_SH = "run.sh"
    CMD_TRAINCHECK = "python -m mldaikon.collect_trace --use-config --config md-config.yml --output-dir traincheck"
    CMD_TRAINCHECK_SELECTIVE = f"python -m mldaikon.collect_trace --use-config --config md-config.yml --output-dir traincheck-selective -i ../{SELC_INV_FILE}"


    with open(f"{E2E_FOLDER}/{workload}/{RUN_SH}", "r") as f:
        cmd = f.read().strip()
    cmd_settrace = cmd.replace(ORIG_PY, SETTRACE_PY)

    cd f"{E2E_FOLDER}/{workload}"

    # run four setups

    # 1. naive running
    print("Running naive setup")
    run_cmd(cmd, kill_sec)
    cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_naive.txt")
    rm iteration_times.txt

    # 2. settrace running
    print("Running settrace setup")
    run_cmd(cmd_settrace, kill_sec)
    rm api_calls.log
    cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_systrace.txt")
    rm iteration_times.txt

    # 3. traincheck proxy instrumentation
    print("Running traincheck instrumentation")
    run_cmd(CMD_TRAINCHECK, kill_sec)
    print("Trying to copy")
    print(os.listdir("traincheck"))
    # shutil.copy("traincheck/iteration_times.txt", f"../../{RES_FOLDER}/e2e_{workload}_monkey-patch.txt")
    cp traincheck/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_monkey-patch.txt")
    print("Copied")
    rm -rf traincheck

    # 4. traincheck selective instrumentation
    print("Running traincheck selective instrumentation")
    run_cmd(CMD_TRAINCHECK_SELECTIVE, kill_sec)
    cp traincheck-selective/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_selective.txt")
    rm -rf traincheck-selective

    cd ../..


# e2e workload
run_exp(kill_sec=15, workload="mnist")
run_exp(kill_sec=15, workload="resnet18")
run_exp(kill_sec=15, workload="transformer")
