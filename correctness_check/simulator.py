import argparse
import json
import os
import subprocess
import threading
import time


def simulate_output_by_time(input_file, output_file):
    time_scale = 1e-9
    with open(input_file, "r") as f:
        lines = f.readlines()

    events = []
    for line in lines:
        try:
            record = json.loads(line)
            events.append((record["time"], line.strip()))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON: {e}")

    events.sort(key=lambda x: x[0])

    with open(output_file, "w") as out:
        last_time = None
        for current_time, line in events:
            if last_time is not None:
                delta = (current_time - last_time) * time_scale
                if delta > 0:
                    time.sleep(delta)
            # print(line + '\n')
            out.write(line + "\n")
            out.flush()
            last_time = current_time


def simulate(dir_path, inv_path):
    files = [
        f for f in os.listdir(dir_path) if f.startswith("trace_") or f.endswith(".json")
    ]
    out_dir = os.path.basename(dir_path) + "_simulated"
    # out_dir = "test"
    os.makedirs(out_dir, exist_ok=True)
    threadlist = []
    for file in files:
        input_path = os.path.join(dir_path, file)
        output_path = os.path.join(out_dir, f"{file}")
        threading_obj = threading.Thread(
            target=simulate_output_by_time, args=(input_path, output_path)
        )
        threadlist.append(threading_obj)
        # simulate_output_by_time(input_path, output_path)

    for thread in threadlist:
        thread.start()

    time.sleep(5)
    command = ["traincheck-onlinecheck", "-f", out_dir, "-i", inv_path]
    process = subprocess.Popen(command)

    for thread in threadlist:
        thread.join()

    time.sleep(20)

    process.terminate()
    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate trace files and simluating online check."
    )
    parser.add_argument(
        "-f",
        "--trace-folders",
        nargs="+",
        help='Folders containing traces files to infer invariants on. Trace files should start with "trace_" or "proxy_log.json"',
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="+",
        required=True,
        help="Invariants files to check on traces",
    )
    args = parser.parse_args()
    simulate(args.trace_folders[0], args.invariants[0])
