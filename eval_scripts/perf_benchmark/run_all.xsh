import os
import subprocess

# configs
$RAISE_SUBPROC_ERROR = True
os.environ["PYTHONUNBUFFERED"] = "1"

SELC_INV_FILE = "sampled_100_invariants.json"
COMMIT = $(git rev-parse --short HEAD)
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

def run_cmd(cmd: str, kill_sec: int):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, _ = p.communicate(timeout=kill_sec)
    except subprocess.TimeoutExpired:
        print(f"Timeout: {kill_sec} seconds, killing the process")
        p.kill()

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

    # run five setups

    # 1. naive running
    print("Running naive setup")
    run_cmd(cmd, kill_sec)
    cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_naive.txt")
    rm iteration_times.txt

    # 2. settrace running
    print("Running settrace setup")
    run_cmd(cmd_settrace, kill_sec)
    rm api_calls.log
    cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_settrace.txt")
    rm iteration_times.txt

    # 3. traincheck proxy instrumentation
    print("Running traincheck proxy instrumentation")
    run_cmd(CMD_TRAINCHECK, kill_sec)
    cp traincheck/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_traincheck.txt")
    rm -rf traincheck
    # rm iteration_times.txt

    # 4. traincheck selective instrumentation
    print("Running traincheck selective instrumentation")
    run_cmd(CMD_TRAINCHECK_SELECTIVE, kill_sec)
    cp traincheck-selective/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_traincheck_selective.txt")
    rm -rf traincheck-selective

    cd ../..


# e2e workload
run_exp(kill_sec=50, workload="mnist")
run_exp(kill_sec=50, workload="resnet18")
run_exp(kill_sec=50, workload="transformer")
