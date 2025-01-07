set -e

# configs
import os
import subprocess

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

def run_cmd_and_get_output(cmd: str, kill_sec: int) -> str:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, _ = p.communicate(timeout=kill_sec)
    except subprocess.TimeoutExpired:
        print(f"Timeout: {kill_sec} seconds, killing the process")
        p.kill()
        output, _ = p.communicate()
    return output.decode("utf-8")

# run e2e benchmark
def run_exp(kill_sec: int = 100, workload: str = "mnist"):
    print(f"Running experiments for {workload}")

    ORIG_PY = "main.py"
    SETTRACE_PY = "main_settrace.py"
    RUN_SH = "run.sh"
    CMD_TRAINCHECK = "python -m mldaikon.collect_trace --use-config --config md-config.yml"
    CMD_TRAINCHECK_SELECTIVE = CMD_TRAINCHECK + f" -i ../{SELC_INV_FILE}"

    with open(f"{E2E_FOLDER}/{workload}/{RUN_SH}", "r") as f:
        cmd = f.read().strip()
    cmd_settrace = cmd.replace(ORIG_PY, SETTRACE_PY)

    cd f"{E2E_FOLDER}/{workload}"

    # run five setups

    # 1. naive running
    print("Running naive setup")
    output = run_cmd_and_get_output(cmd, kill_sec)
    with open(f"../../{RES_FOLDER}/e2e_{workload}_naive.txt", "w") as f:
        f.write(output)

    # 2. settrace running
    print("Running settrace setup")
    output_settrace = run_cmd_and_get_output(cmd_settrace, kill_sec)
    with open(f"../../{RES_FOLDER}/e2e_{workload}_settrace.txt", "w") as f:
        f.write(output_settrace)

    # 3. traincheck proxy instrumentation
    print("Running traincheck proxy instrumentation")
    output_traincheck = run_cmd_and_get_output(CMD_TRAINCHECK, kill_sec)
    with open(f"../../{RES_FOLDER}/e2e_{workload}_traincheck.txt", "w") as f:
        f.write(output_traincheck)

    # 4. traincheck selective instrumentation
    print("Running traincheck selective instrumentation")
    output_traincheck_selective = run_cmd_and_get_output(CMD_TRAINCHECK_SELECTIVE, kill_sec)
    with open(f"../../{RES_FOLDER}/e2e_{workload}_traincheck_selective.txt", "w") as f:
        f.write(output_traincheck_selective)

    cd ../..


# e2e workload
run_exp(kill_sec=100, workload="mnist")
run_exp(kill_sec=100, workload="resnet18")
run_exp(kill_sec=100, workload="transformer")
