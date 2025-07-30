import json
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from traincheck.invariant.base_cls import Invariant
from traincheck.trace import MDNONEJSONDecoder


def collect_trace_dirs(reference_root: str = "./reference_result") -> list[Path]:
    trace_dirs = []
    simulate_trace_dirs = []
    for group_dir in Path(reference_root).iterdir():
        if group_dir.is_dir():
            for trace_dir in group_dir.iterdir():
                if trace_dir.name.endswith("static") and trace_dir.is_dir():
                    trace_dirs.append(trace_dir)
                if trace_dir.name.endswith("simulated") and trace_dir.is_dir():
                    simulate_trace_dirs.append(trace_dir)
    return trace_dirs, simulate_trace_dirs


def find_trace_components(trace_dir: Path):
    trace = None
    reference_out = None
    invariant = None

    for item in trace_dir.iterdir():
        name = item.name
        if name.startswith("trace"):
            trace = item
        elif name.startswith("traincheck_onlinecheck") and name.endswith(".log"):
            reference_out = item
        elif name.startswith("traincheck_onlinechecker") and item.is_dir():
            invariant = item / "invariants.json"

    if not trace or not reference_out or not invariant.exists():
        raise FileNotFoundError(f"Incomplete files in {trace_dir}")
    return trace, reference_out, invariant


def find_trace_components_offline(trace_dir: Path):
    trace = None
    reference_out = None
    invariant = None

    for item in trace_dir.iterdir():
        name = item.name
        if name.startswith("trace"):
            trace = item
        elif name.startswith("traincheck_checker") and item.is_dir():
            invariant = item / "invariants.json"
            for sub_item in item.iterdir():
                if sub_item.name.startswith("trace") and sub_item.is_dir():
                    reference_out = sub_item / "failed.log"

    if not trace or not reference_out or not invariant.exists():
        raise FileNotFoundError(f"Incomplete files in {trace_dir}")
    return trace, reference_out, invariant


def run_online_checker(trace: Path, invariant: Path) -> Path:
    command = ["traincheck-onlinecheck", "-f", str(trace), "-i", str(invariant)]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    try:
        time.sleep(20)
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"Force killing unresponsive process for {trace}")
        process.kill()
        process.wait()

    # Find latest log file
    log_files = sorted(
        Path(".").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not log_files:
        raise FileNotFoundError("No log output found after checker run.")
    return log_files[0]


def run_simulator(trace_dir: Path, invariant: Path) -> Path:
    subprocess.run(
        ["python3", "simulator.py", "-f", str(trace_dir), "-i", str(invariant)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log_files = sorted(
        Path(".").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not log_files:
        raise FileNotFoundError("No log output found after checker run.")
    return log_files[0]


def run_offline_checker(trace: Path, invariant: Path) -> Path:
    command = ["traincheck-check", "-f", str(trace), "-i", str(invariant)]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    process.wait()

    candidates = [p for p in Path(".").glob("traincheck_checker_result*") if p.is_dir()]
    log_files = max(candidates, key=lambda p: p.stat().st_mtime)

    if not log_files:
        raise FileNotFoundError("No log output found after checker run.")
    for item in log_files.iterdir():
        if item.name.startswith("trace") and item.is_dir():
            print(item.name)
            return item / "failed.log"


def extract_summary_info(log_path: Path):
    last_lines = deque(open(log_path), maxlen=3)

    violations_count = None
    invariants_count = None
    for line in last_lines:
        if "violations found" in line:
            match = re.search(r"Total (\d+) violations found", line)
            if match:
                violations_count = int(match.group(1))
        elif "invariants violated" in line:
            match = re.search(r"Total (\d+) invariants violated", line)
            if match:
                invariants_count = int(match.group(1))
    return violations_count, invariants_count


def compare_logs(output_log: Path, reference_log: Path) -> bool:
    result = subprocess.run(
        ["diff", "-w", str(output_log), str(reference_log)], capture_output=True
    )
    if result.returncode == 0:
        return True

    # Soft match check
    v1, i1 = extract_summary_info(output_log)
    v2, i2 = extract_summary_info(reference_log)

    if None in (v1, v2, i1, i2):
        return False

    if i1 != i2:
        return False

    if v1 == 0 and v2 == 0:
        print("soft check pass")
        return True

    diff_ratio = abs(v1 - v2) / max(v1, v2)
    if diff_ratio <= 0.1:
        print("soft check pass")
        return True

    return False


def compare_offline_logs(output_log: Path, reference_log: Path) -> bool:
    failed_invs = read_inv_file(output_log)
    reference_invs = read_inv_file(reference_log)

    if failed_invs == reference_invs:
        return True

    print(str(output_log), str(reference_log))
    return False


def read_inv_file(file_path: Path) -> list[Invariant]:
    with open(file_path, "r") as f:
        content = f.read()
        try:
            all_json_objects = json.loads(
                "[" + content.strip().replace("}\n{", "},\n{") + "]",
                cls=MDNONEJSONDecoder,
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode failed: {e}") from e

    invs = []
    for raw in all_json_objects:
        if "invariant" not in raw:
            raise ValueError("Missing 'invariant' field in one object.")
        inv = Invariant.from_dict(raw["invariant"])
        invs.append(inv)
    return invs


def main():
    static_trace_dirs, simulate_trace_dirs = collect_trace_dirs()
    all_passed = True

    # # online static check
    for trace_dir in static_trace_dirs:
        try:
            trace, ref_log, invariant = find_trace_components(trace_dir)
            out_log = run_online_checker(trace, invariant)
            if compare_logs(out_log, ref_log):
                print(f"Check passed for {trace_dir}")
            else:
                print(f"Check failed for {trace_dir}")
                all_passed = False
                sys.exit(1)
        except Exception as e:
            print(f"Error processing {trace_dir}: {e}")
            all_passed = False
            sys.exit(1)

    # # online simulated check
    for trace_dir in simulate_trace_dirs:
        try:
            trace, ref_log, invariant = find_trace_components(trace_dir)
            out_log = run_simulator(trace, invariant)
            if compare_logs(out_log, ref_log):
                print(f"Check passed for {trace_dir}")
            else:
                print(f"Check failed for {trace_dir}")
                all_passed = False
                sys.exit(1)
        except Exception as e:
            print(f"Error processing {trace_dir}: {e}")
            all_passed = False
            sys.exit(1)

    # offline static check
    for trace_dir in static_trace_dirs:
        if "modified" in str(trace_dir):
            continue
        try:
            trace, ref_log, invariant = find_trace_components_offline(trace_dir)
            out_log = run_offline_checker(trace, invariant)
            if compare_offline_logs(out_log, ref_log):
                print(f"Offline check passed for {trace_dir}")
            else:
                print(f"Offline check failed for {trace_dir}")
                all_passed = False
                sys.exit(1)
        except Exception as e:
            print(f"Error processing {trace_dir}: {e}")
            all_passed = False
            sys.exit(1)

    if all_passed:
        print("All checks passed!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
