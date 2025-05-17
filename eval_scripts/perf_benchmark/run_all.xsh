import argparse
import os
import signal
import subprocess
import time

# configs
$RAISE_SUBPROC_ERROR = True
os.environ["PYTHONUNBUFFERED"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--res_folder", type=str, required=False)
parser.add_argument("-w", "--workloads", type=str, nargs='*', required=False)
parser.add_argument("-v", "--track-variables", action="store_true", required=False)
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

def kill_all_GPU_processes():
    pids = get_all_GPU_pids()
    if len(pids) == 0:
        print("Warning: No GPU processes to kill. Probably the original process crashed somewhere instead of running indefinitely")
        return
    for pid in pids:
        print(f"Killing process {pid}")
        kill -9 @(pid)

def run_cmd(cmd: str, kill_sec: int):
    with open("cmd_output.log", "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=f)
        try:
            if kill_sec >= 0:
                output, _ = p.communicate(timeout=kill_sec)
            else:
                output, _ = p.communicate()
        except subprocess.TimeoutExpired:
            print(f"Timeout: {kill_sec} seconds, killing all GPU processes")
            kill_all_GPU_processes()
    
    # print the output
    with open("cmd_output.log", "r") as f:
        print("Output of the command:")
        print(f.read())
        print("End of the output")
        

# run e2e benchmark
def run_exp(kill_sec: int = 100, workload: str = "mnist", use_proxy: bool = False):
    print(f"Running experiments for {workload}")

    ORIG_PY = "main.py"
    SETTRACE_PY = "main_settrace.py"
    RUN_SH = "run.sh"
    MD_CONFIG_YML = "md-config.yml" if not use_proxy else "md-config-var.yml"
    CMD_TRAINCHECK = f"python -m traincheck.collect_trace --use-config --config {MD_CONFIG_YML} --output-dir traincheck-all"
    CMD_TRAINCHECK_SELECTIVE = f"python -m traincheck.collect_trace --use-config --config {MD_CONFIG_YML} --output-dir traincheck-selective -i ../{SELC_INV_FILE}"

    if not os.path.exists(f"{E2E_FOLDER}/{workload}/{RUN_SH}"):
        cmd = "python3 main.py"
    else:
        with open(f"{E2E_FOLDER}/{workload}/{RUN_SH}", "r") as f:
            cmd = f.read().strip()

    # remove all '\' and linebreaks
    cmd = cmd.replace("\\", "").replace("\n", "")
    cmd_settrace = cmd.replace(ORIG_PY, SETTRACE_PY)

    cd f"{E2E_FOLDER}/{workload}"

    # run four setups

    try:
    # 1. naive running
        print("Running naive setup")
        NAIVE_ENTER_TIME = time.perf_counter()
        run_cmd(cmd, kill_sec)
        NAIVE_EXIT_TIME = time.perf_counter()
        cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_naive.txt")
        rm iteration_times.txt

        # 2. settrace running
        print("Running settrace setup")
        run_cmd(cmd_settrace, kill_sec * 2)
        rm api_calls.log
        cp iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_systrace.txt")
        rm iteration_times.txt

        # 3. traincheck proxy instrumentation
        print("Running traincheck instrumentation")
        run_cmd(CMD_TRAINCHECK, kill_sec)
        cp traincheck-all/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_monkey-patch.txt")
        rm -rf traincheck-all

        # 4. traincheck selective instrumentation
        print("Running traincheck selective instrumentation")
        SEL_ENTER_TIME = time.perf_counter()
        run_cmd(CMD_TRAINCHECK_SELECTIVE, kill_sec)
        SEL_EXIT_TIME = time.perf_counter()
        cp traincheck-selective/iteration_times.txt @(f"../../{RES_FOLDER}/e2e_{workload}_selective.txt")
        rm -rf traincheck-selective


        # write the time
        with open(f"../../{RES_FOLDER}/e2e_{workload}_completion-time.csv", "w") as f:
            f.write(f"naive,{NAIVE_EXIT_TIME - NAIVE_ENTER_TIME}\n")
            f.write(f"selective,{SEL_EXIT_TIME - SEL_ENTER_TIME}\n")
    except Exception as e:
        print(f"Error: {e}, skipping the rest of the experiment")
        kill_all_GPU_processes()

    cd ../..


# e2e workload
# run_exp(kill_sec=60, workload="mnist")
# run_exp(kill_sec=60, workload="resnet18")
# run_exp(kill_sec=60, workload="transformer")

# discover the workload
if args.workloads:
    print("Running selected workloads")
    workloads = args.workloads
else:
    workloads = os.listdir(E2E_FOLDER)

workloads = [w for w in workloads if os.path.isdir(f"{E2E_FOLDER}/{w}") and w != "data"]
print(f"{len(workloads)} workloads to run: ", workloads)
for w in workloads:
    if "ac_bert" in w:
        run_exp(kill_sec=200, workload=w, use_proxy=args.track_variables)
    elif "tf_summarization" in w:
        run_exp(kill_sec=400, workload=w, use_proxy=args.track_variables)
    else:
        run_exp(kill_sec=60, workload=w, use_proxy=args.track_variables)

