import json
import time
import os
import threading

def simulate_output_by_time(input_file, output_file):
    time_scale = 1e-9
    with open(input_file, 'r') as f:
        lines = f.readlines()

    events = []
    for line in lines:
        try:
            record = json.loads(line)
            events.append((record['time'], line.strip()))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON: {e}")

    events.sort(key=lambda x: x[0])

    with open(output_file, 'w') as out:
        last_time = None
        for current_time, line in events:
            if last_time is not None:
                delta = (current_time - last_time) * time_scale
                if delta > 0:
                    time.sleep(delta)
            # print(line + '\n')
            out.write(line + '\n')
            out.flush()
            last_time = current_time


def simulate(dir_path):
    files = [f for f in os.listdir(dir_path) if f.startswith("trace_") or f.endswith(".jsonl")]
    out_dir = os.path.join(dir_path, "simulated")
    os.makedirs(out_dir, exist_ok=True)
    threadlist = []
    for file in files:
        input_path = os.path.join(dir_path, file)
        output_path = os.path.join(out_dir, f"{file}")
        threading_obj = threading.Thread(target=simulate_output_by_time, args=(input_path, output_path))
        threadlist.append(threading_obj)
        # simulate_output_by_time(input_path, output_path)

    for thread in threadlist:
        thread.start()

    for thread in threadlist:
        thread.join()

if __name__ == "__main__":
    simulate("trace_test2")
