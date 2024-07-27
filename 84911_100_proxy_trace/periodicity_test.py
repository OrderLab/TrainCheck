import json

import matplotlib.pyplot as plt
import numpy as np


def process_line(line):
    line = line.strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def read_json_lines(file_path):
    results = []
    with open(file_path, "r") as file:
        for line in file:
            processed = process_line(line)
            if processed:
                results.append(processed)
    return results


data = read_json_lines("proxy_trace_processed_4.json")

t_time = data[0]["time"]
print(t_time)


y = []
time = 0.0
for i, line in enumerate(data):
    new_time = round((line["time"] - t_time), 2)
    while time < new_time:
        y.append(line["meta_vars"]["grad_norm"])
        time += 0.01

# n = len(y)
# # n_padded = n * 5
# # y = np.pad(y, (0, n_padded - n), 'constant')
x = np.arange(len(y))
y_fft = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(y), d=1)

y_fft_shifted = np.fft.fftshift(y_fft)
frequencies_shifted = np.fft.fftshift(frequencies)
print(frequencies_shifted)
# original signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y, marker="o")
plt.grid(True)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.title("Original Signal")

# dft result
plt.subplot(2, 1, 2)
plt.stem(frequencies_shifted, np.abs(y_fft_shifted))
plt.title("DFT of the Signal")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

# save img
plt.savefig("dft_signal_zero_padding_100_it.png")
plt.show()
